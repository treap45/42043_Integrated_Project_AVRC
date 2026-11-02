"""Velocity-level robot I/O for the lab pipeline: the real UR3e (RTDE speedJ), or the simulation standing in for it.

PickPlace was written for a torque-controlled simulation: each control period it reads the encoders, asks the
joint controller for a torque and steps the simulator. On the UR3e the joint loops belong to the robot controller,
which takes joint velocities. Both classes here give PickPlace what it uses (dt, xhat, read_camera, set_gripper)
and turn its joint reference (q_ref, qd_ref) into a joint-velocity command

    qd_cmd = qd_ref + KP (q_ref - q)

that is checked (SafetyLimits) and sent with speedJ. SimIO sends the same command to an emulated speedJ (an
acceleration-limited velocity integrated into a reference that the simulation's LQR tracks) and renders the
scene's fixed camera, so the whole lab pipeline can be run in simulation before the robot.
"""

import collections
import threading
import time
from dataclasses import dataclass

import numpy as np

KP = 4.0               # 1/s, position correction on top of the velocity feed-forward
DT = 0.002             # s, 500 Hz like the simulation (GOAL_UPDATE_PERIOD = 50 gives 10 Hz vision)
QD_MAX = 0.35          # rad/s, every joint; the lab examples use 0.02-0.05, PBVS asks up to about 0.3 when tracking
ACCELERATION = 1.5     # rad/s^2, speedJ acceleration
RELEASE_MARGIN_MM = 4.0  # mm, beyond the cube width: the first opening step when letting go
STALE_FRAME_S = 0.3    # a camera frame older than this is counted as stale
DEPTH_FRAMES = 4       # depth frames averaged per measurement (fixed camera; 0.13 s at 30 fps)
SIM_DEPTH_K = 0.004    # sim depth noise k_axial for the fixed camera (the simulation default 0.01 is the D435 spec
                       # limit at 2 m); at 0.9 m: 3.2 mm. Measure the lab camera with scripts/check_real_depth.py


def average_depth(depths) -> np.ndarray:
    """Mean of the valid (non-zero) readings per pixel; 0 where none of the frames has one."""
    D = np.stack(depths)
    valid = D > 0
    n = valid.sum(axis=0)
    return np.where(n > 0, (D * valid).sum(axis=0) / np.maximum(n, 1), 0.0).astype(np.float32)


class SafetyStop(RuntimeError):
    """Raised when a command would break a safety limit; the caller stops the robot."""


@dataclass
class SafetyLimits:
    table_z: float                 # world z of the table surface
    min_tcp_above_table: float = 0.015   # m; the grasp pose is the cube centre 5 mm down, 25-30 mm above the table
    max_radius: float = 0.50       # m, TCP horizontal distance from the base axis
    max_tracking_error: float = 0.15     # rad, |q_ref - q| on any joint
    qd_max: float = QD_MAX

    def check(self, robot, q, q_ref, qd_cmd, dt) -> np.ndarray:
        """Return the (possibly scaled) command, or raise SafetyStop."""
        if not (np.all(np.isfinite(q_ref)) and np.all(np.isfinite(qd_cmd))):
            raise SafetyStop("non-finite joint command")
        err = float(np.max(np.abs(q_ref - q)))
        if err > self.max_tracking_error:
            raise SafetyStop(f"joint tracking error {err:.3f} rad > {self.max_tracking_error}")
        peak = float(np.max(np.abs(qd_cmd)))
        if peak > self.qd_max:
            qd_cmd = qd_cmd * (self.qd_max / peak)
        # Look one short horizon ahead along the command, so the limit holds before the TCP gets there.
        p, _ = robot.site_pose(q + qd_cmd * 0.1, "tcp")
        if p[2] < self.table_z + self.min_tcp_above_table:
            raise SafetyStop(f"TCP would go to {1000 * (p[2] - self.table_z):.0f} mm above the table")
        if np.hypot(p[0], p[1]) > self.max_radius:
            raise SafetyStop(f"TCP would leave the {self.max_radius} m workspace radius")
        return qd_cmd


class _VelocityIO:
    dt = DT
    grasp_held = None  # result of the last close: True or False on the RG2, None when the gripper cannot tell

    def __init__(self, robot, limits: SafetyLimits, kp: float = KP):
        self.robot = robot
        self.limits = limits
        self.kp = kp
        self._xhat = np.zeros((6, 3))
        self.last_cmd = np.zeros(6)
        self.overruns = 0

    @property
    def xhat(self) -> np.ndarray:
        """[q, qd, 0] per joint, the shape of ArmController.xhat (there is no disturbance estimate here)."""
        return self._xhat

    def velocity_command(self, q_ref, qd_ref) -> np.ndarray:
        q = self._xhat[:, 0]
        qd = np.asarray(qd_ref, dtype=float) + self.kp * (np.asarray(q_ref, dtype=float) - q)
        return self.limits.check(self.robot, q, np.asarray(q_ref, dtype=float), qd, self.dt)


class SimIO(_VelocityIO):
    """The simulation behind the velocity interface, seen through the scene's fixed camera."""

    def __init__(self, robot, limits: SafetyLimits, seed: int = 0, moving_cube: bool = True,
                 camera_noise: bool = True, camera_name: str = "fixed_cam", kp: float = KP,
                 acceleration: float = ACCELERATION, depth_k: float = SIM_DEPTH_K,
                 depth_frames: int = DEPTH_FRAMES):
        from src.control.arm_controller import ArmController
        from src.sim import CameraNoiseConfig, CubeMotionConfig, TaskSim
        from src.sim.sensors import CameraNoise, WristCamera

        super().__init__(robot, limits, kp)
        self.sim = TaskSim(camera_noise=None, cube_motion=CubeMotionConfig() if moving_cube else None)
        self.sim.reset(seed, "L0")
        if abs(self.sim.dt - self.dt) > 1e-12:
            raise ValueError(f"simulation dt {self.sim.dt} differs from {self.dt}")
        self.cam = WristCamera(self.sim.model, camera_name)
        self.K_true = self.cam._K_true.copy()
        self.noise = (CameraNoise(CameraNoiseConfig(k_axial=depth_k), self.K_true, np.random.default_rng(seed))
                      if camera_noise else None)
        self.depth_frames = depth_frames
        y0 = self.sim.read_joint_positions()
        self._ctrl = ArmController(robot=robot, dt=self.dt, y0=y0)  # plant side: the robot's own joint loops
        self._q_int = y0.copy()
        self._qd_sent = np.zeros(6)
        self.acceleration = acceleration
        self._xhat[:, 0] = y0
        self.time = 0.0

    def read_camera(self):
        rgb0, depth0 = self.cam.render(self.sim.data)
        if self.noise is None:
            return rgb0, depth0
        # Independent noise on one render: the cube's motion within the averaging window is ignored here.
        frames = [self.noise.apply(rgb0, depth0) for _ in range(self.depth_frames)]
        return frames[-1][0], average_depth([d for _, d in frames])

    def set_gripper(self, closure: float) -> None:
        self.sim.set_gripper(closure)

    def send(self, q_ref, qd_ref) -> None:
        qd_cmd = self.velocity_command(q_ref, qd_ref)
        self.last_cmd = qd_cmd
        # speedJ: the commanded velocity is reached at the given acceleration.
        step = np.clip(qd_cmd - self._qd_sent, -self.acceleration * self.dt, self.acceleration * self.dt)
        self._qd_sent = self._qd_sent + step
        self._q_int = self._q_int + self._qd_sent * self.dt
        tau = self._ctrl.step(self.sim.read_joint_positions(), self._q_int, self._qd_sent)
        self.sim.step(tau)
        self.time += self.dt
        self._xhat[:, 0] = self.sim.read_joint_positions()
        self._xhat[:, 1] = self._ctrl.xhat[:, 1]

    def stop(self) -> None:
        self._qd_sent[:] = 0.0

    def close(self) -> None:
        self.cam.close()
        self.sim.close()


class _CameraThread(threading.Thread):
    """Keeps the latest D435i frame so the control loop never waits for the camera."""

    def __init__(self, cam, depth_frames: int = DEPTH_FRAMES):
        super().__init__(daemon=True)
        self.cam = cam
        self._lock = threading.Lock()
        self._frame = None
        self._depths = collections.deque(maxlen=depth_frames)
        self._stamp = 0.0
        self._run = True
        self.error = None

    def run(self):
        try:
            while self._run:
                f = self.cam.capture()
                with self._lock:
                    self._frame, self._stamp = f, time.monotonic()
                    self._depths.append(f.depth_m)
        except Exception as exc:  # reported by latest()
            self.error = exc

    def latest(self):
        if self.error is not None:
            raise RuntimeError(f"camera thread stopped: {self.error}")
        with self._lock:
            depth = None if self._frame is None else average_depth(list(self._depths))
            return self._frame, depth, time.monotonic() - self._stamp

    def stop(self):
        self._run = False


class RealIO(_VelocityIO):
    """The lab UR3e over RTDE, the fixed D435i and the RG2.

    dry_run: RTDE control is not opened; commands are computed and checked but not sent, and the gripper is not
    moved. Use it first, with the arm where it would start, to see the camera, the cube estimate and the commands.
    """

    def __init__(self, robot, limits: SafetyLimits, ip: str, dry_run: bool = False, kp: float = KP,
                 acceleration: float = ACCELERATION, cube_side_mm: float = 60.0,
                 depth_frames: int = DEPTH_FRAMES):
        from rtde_receive import RTDEReceiveInterface

        from src.real.camera import D435i
        from src.real.gripper import RG2

        super().__init__(robot, limits, kp)
        self.dry_run = dry_run
        self.acceleration = acceleration
        self.cube_side_mm = cube_side_mm
        self._r = RTDEReceiveInterface(ip, 1.0 / self.dt)
        self._c = None
        if not dry_run:
            from rtde_control import RTDEControlInterface
            self._c = RTDEControlInterface(ip, 1.0 / self.dt)
        self._cam = D435i()
        self._cam.__enter__()
        self.K = self._cam.K
        self._thread = _CameraThread(self._cam, depth_frames)
        self._thread.start()
        self.gripper = None if dry_run else RG2(ip)
        self.stale_frames = 0
        self._read_state()
        self._deadline = time.monotonic()
        self.time = 0.0

    def _read_state(self):
        self._xhat[:, 0] = self._r.getActualQ()
        self._xhat[:, 1] = self._r.getActualQd()

    def read_camera(self):
        frame, depth, age = self._thread.latest()
        while frame is None:  # first frames after start-up
            time.sleep(0.01)
            frame, depth, age = self._thread.latest()
        if age > STALE_FRAME_S:
            self.stale_frames += 1
        return frame.rgb, depth

    def set_gripper(self, closure: float) -> None:
        self._halt()
        if self.gripper is None:
            print(f"  [dry run] gripper {'close' if closure >= 0.5 else 'open'}")
            return
        if closure >= 0.5:
            w, self.grasp_held = self.gripper.close_on(self.cube_side_mm)
            print(f"  gripper closed at {w:.1f} mm, held={self.grasp_held}")
        else:
            if self.grasp_held:
                # Let go first: the fingertips rise as they open, and while still gripping would lift the cube with them.
                self.gripper.grip(self.cube_side_mm + RELEASE_MARGIN_MM)
            self.grasp_held = None
            print(f"  gripper open at {self.gripper.open():.1f} mm")
        self._deadline = time.monotonic()

    def send(self, q_ref, qd_ref) -> None:
        self._read_state()
        qd_cmd = self.velocity_command(q_ref, qd_ref)
        self.last_cmd = qd_cmd
        if self._c is not None and not self._c.speedJ(qd_cmd.tolist(), self.acceleration, self.dt):
            raise SafetyStop("speedJ was refused (protective stop, or the program is not running)")
        # Absolute deadlines: a slow vision period is made up by the following ones, so the dt that PickPlace
        # and the EKF count matches the wall clock on average.
        self._deadline += self.dt
        delay = self._deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elif delay < -10 * self.dt:
            self.overruns += 1
            self._deadline = time.monotonic()
        self.time += self.dt

    def _halt(self) -> None:
        if self._c is not None:
            self._c.speedStop(2.0)
        time.sleep(0.2)
        self._read_state()

    def stop(self) -> None:
        if self._c is not None:
            try:
                self._c.speedStop(2.0)
            except Exception:
                pass

    def close(self) -> None:
        self.stop()
        self._thread.stop()
        self._thread.join(timeout=1.0)
        self._cam.__exit__(None, None, None)
        if self._c is not None:
            try:
                self._c.stopScript()
            except Exception:
                pass
            self._c.disconnect()
        self._r.disconnect()
