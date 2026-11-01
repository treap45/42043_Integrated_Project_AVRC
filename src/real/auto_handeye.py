"""Automatic eye-to-hand data collection: the arm moves itself through varied views of the marker on the gripper.

Starting from a pose where the camera sees the marker, the arm goes through two phases of slow joint moves (moveJ):
    1. small joint offsets around the start, which keep the marker in view, until a coarse calibration is possible;
    2. larger offsets chosen with that coarse calibration: the marker must stay in the image and tilted to the
       camera, and each view is the candidate most different from those already saved.
It then returns to the start pose. Views are saved in the format of scripts/capture_handeye.py.

Every target and the straight joint path to it are checked first (ViewLimits). Each move is asynchronous and watched,
so a protective stop, a timeout or a stop request ends the run with the arm stopped.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.real.aruco import (DICTIONARY, MARKER_ID, detect, make_detector, marker_depth, marker_object_points,
                            marker_pose, tilt_deg)
from src.real.frames import invert_T, make_T, rotation_angle_deg, rtde_pose_to_T, sim_world_from_rtde
from src.real.handeye import evaluate, refine, solve_T_base_cam

PHASE1_SPREAD_DEG = np.array([6.0, 5.0, 7.0, 12.0, 12.0, 12.0])
PHASE2_SPREAD_DEG = np.array([20.0, 15.0, 20.0, 35.0, 35.0, 40.0])
PHASE1_MAX_VIEWS = 14
PHASE1_MIN_VIEWS = 6
COARSE_MAX_REPROJ_PX = 5.0
COARSE_MIN_TILT_DEG = 12.0
CANDIDATES = 1500
MIN_TILT_DEG, MAX_TILT_DEG = 20.0, 55.0
MIN_NEW_DEG = 10.0            # a planned view differs at least this much from every saved one
MIN_SIDE_PX = 25.0
IMAGE_MARGIN_PX = 30.0
AVERAGE_FRAMES = 10
STILL_QD = 1e-3               # rad/s
STILL_TIME = 0.3              # s
MOVE_TIMEOUT = 30.0           # s
AT_TARGET_RAD = 0.005         # rad, every joint
MAX_MISSED = 5                # consecutive moves without a detection before giving up


class AutoStop(RuntimeError):
    """The run must end. halt=True: something is wrong or a stop was asked for, so the arm stays where it stopped.
    halt=False: collection cannot go on but the arm is fine, so it may return to the start pose."""

    def __init__(self, message: str, halt: bool = True):
        super().__init__(message)
        self.halt = halt


@dataclass
class ViewLimits:
    table_z: float = 0.0          # m, table surface in the RTDE base frame
    clearance: float = 0.05       # m, flange and fingertip stay this far above the table
    tool_length: float = 0.23     # m, flange to beyond the RG2 fingertips, along the flange z
    max_radius: float = 0.50      # m, flange horizontal distance from the base axis
    max_tcp_travel: float = 0.15  # m, flange distance from its start position
    speed: float = 0.20           # rad/s, leading joint
    acceleration: float = 0.40    # rad/s^2


class Kinematics:
    """Planning FK in the RTDE base frame from the controller-side model (matches the robot to 0.8 mm, 0.4 deg)."""

    def __init__(self, robot, q0, T_base_tcp0):
        self.robot = robot
        self._T_b_w = invert_T(sim_world_from_rtde())
        # The pendant TCP may or may not be the flange; keep what the robot reports as a constant offset.
        self.T_flange_tcp = invert_T(self.flange(q0)) @ T_base_tcp0

    def flange(self, q) -> np.ndarray:
        p, R = self.robot.site_pose(np.asarray(q, dtype=float), "attachment_site")
        return self._T_b_w @ make_T(R, p)

    def tcp(self, q) -> np.ndarray:
        return self.flange(q) @ self.T_flange_tcp


def path_is_clear(kin: Kinematics, q_from, q_to, p_start, limits: ViewLimits, samples: int = 12) -> bool:
    """Check the flange and fingertip along the straight joint path that moveJ follows."""
    for s in np.linspace(0.0, 1.0, samples):
        T = kin.flange(q_from + s * (q_to - q_from))
        p, tip = T[:3, 3], T[:3, 3] + limits.tool_length * T[:3, 2]
        if min(p[2], tip[2]) < limits.table_z + limits.clearance:
            return False
        if np.hypot(p[0], p[1]) > limits.max_radius:
            return False
        if np.linalg.norm(p - p_start) > limits.max_tcp_travel:
            return False
    return True


def predict_marker(T_base_tcp, T_base_cam, T_ee_marker, obj, K):
    """Predicted corners, R_cam_marker and tilt; None if the marker is behind the camera or faces away."""
    T_cm = invert_T(T_base_cam) @ T_base_tcp @ T_ee_marker
    t = T_cm[:3, 3]
    if t[2] <= 0.05:
        return None
    normal, ray = T_cm[:3, 2], t / np.linalg.norm(t)
    if normal @ ray >= 0.0:  # the printed side must face the camera
        return None
    pts = (T_cm[:3, :3] @ obj.T).T + t
    uv = (K @ pts.T).T
    corners = uv[:, :2] / uv[:, 2:]
    tilt = float(np.degrees(np.arccos(np.clip(-(normal @ ray), 0.0, 1.0))))
    return corners, T_cm[:3, :3], tilt


class AutoCollector:
    def __init__(self, ctrl, recv, cam, robot, side: float, limits: ViewLimits, out: Path, n_views: int = 25,
                 seed: int = 0, confirm=None, on_frame=None, log=print):
        self.ctrl, self.recv, self.cam, self.robot = ctrl, recv, cam, robot
        self.side, self.limits, self.out, self.n_views = side, limits, Path(out), n_views
        self.rng = np.random.default_rng(seed)
        self.confirm = confirm      # called before each move with its description; raise AutoStop to refuse
        self.on_frame = on_frame    # called with (rgb, corners or None) while moving; True asks to stop
        self.log = log
        self.detector = make_detector()
        self.obj = marker_object_points(side)
        self.views = []
        self.T_base_cam = None
        self.T_ee_marker = None

    # ---- robot ------------------------------------------------------------------------------------------

    def _check_robot(self) -> None:
        if self.recv.isEmergencyStopped():
            raise AutoStop("emergency stop is active")
        if self.recv.isProtectiveStopped():
            raise AutoStop("protective stop: check the arm and clear it on the pendant")

    def _q(self) -> np.ndarray:
        return np.asarray(self.recv.getActualQ(), dtype=float)

    def _move(self, q_target, label: str) -> None:
        q_target = np.asarray(q_target, dtype=float)
        self._check_robot()
        if not self.ctrl.isJointsWithinSafetyLimits(list(map(float, q_target))):
            raise AutoStop(f"{label}: target outside the robot's safety limits")
        if self.confirm is not None:
            self.confirm(label)
        if not self.ctrl.moveJ(list(map(float, q_target)), self.limits.speed, self.limits.acceleration, True):
            raise AutoStop(f"{label}: moveJ refused (is the pendant in Remote?)")
        t0 = time.monotonic()
        try:
            while True:
                time.sleep(0.02)
                self._check_robot()
                if self.on_frame is not None:
                    f = self.cam.capture()
                    if self.on_frame(f.rgb, detect(self.detector, f.rgb)):
                        raise AutoStop("stop requested")
                if time.monotonic() - t0 > MOVE_TIMEOUT:
                    raise AutoStop(f"{label}: move timed out")
                # Negative progress means no operation is running, which is also true just after the command, so
                # the move only counts as finished at its target; otherwise the timeout above ends it.
                at_target = np.max(np.abs(self._q() - q_target)) < AT_TARGET_RAD
                if at_target and self.ctrl.getAsyncOperationProgress() < 0:
                    break
        except BaseException:  # including Ctrl+C
            self.ctrl.stopJ(1.0)
            raise
        still_since = None
        while True:
            self._check_robot()
            moving = np.max(np.abs(self.recv.getActualQd())) > STILL_QD
            now = time.monotonic()
            still_since = None if moving else (still_since if still_since is not None else now)
            if still_since is not None and now - still_since >= STILL_TIME:
                break
            if now - t0 > MOVE_TIMEOUT:
                raise AutoStop(f"{label}: arm did not settle")
            time.sleep(0.02)

    # ---- views ------------------------------------------------------------------------------------------

    def _capture_view(self) -> dict | None:
        stack, frame = [], None
        for _ in range(AVERAGE_FRAMES):
            frame = self.cam.capture()
            c = detect(self.detector, frame.rgb)
            if c is not None:
                stack.append(c)
        if len(stack) < AVERAGE_FRAMES // 2:
            return None
        corners = np.mean(stack, axis=0)
        obs = marker_pose(corners, self.side, self.cam.K, self.cam.dist)
        i = len(self.views)
        cv2.imwrite(str(self.out / f"{i:03d}.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        np.save(self.out / f"{i:03d}_depth.npy", frame.depth_m)
        view = {"index": i, "time": time.time(), "corners_px": corners.tolist(),
                "corner_std_px": float(np.std(stack, axis=0).mean()), "frames_averaged": len(stack),
                "marker_depth_m": marker_depth(frame.depth_m, corners),
                "pnp_t_cam_marker": obs.t_cam_marker.tolist(), "pnp_reproj_px": obs.reproj_px,
                "tilt_deg": tilt_deg(obs), "q": self._q().tolist(),
                "tcp_pose": np.asarray(self.recv.getActualTCPPose(), dtype=float).tolist(),
                "R_cam_marker": obs.R_cam_marker.tolist()}
        self.views.append(view)
        (self.out / "views.json").write_text(json.dumps(self.views, indent=1))
        self.log(f"  view {i:2d}: dist {np.linalg.norm(obs.t_cam_marker):.3f} m, side {obs.side_px:.0f} px, "
                 f"tilt {view['tilt_deg']:.0f} deg")
        return view

    def _coarse_solve(self) -> float | None:
        """Update T_base_cam and T_ee_marker from the saved views; returns the mean reprojection error in px."""
        views = [v for v in self.views if v["tilt_deg"] >= COARSE_MIN_TILT_DEG]
        if len(views) < 4:
            return None
        Tb = [rtde_pose_to_T(v["tcp_pose"]) for v in views]
        corners = [np.asarray(v["corners_px"]) for v in views]
        Tc = []
        for c in corners:
            obs = marker_pose(c, self.side, self.cam.K, self.cam.dist)
            Tc.append(make_T(obs.R_cam_marker, obs.t_cam_marker))
        try:
            T0 = solve_T_base_cam(Tb, Tc, "Park")
            T_em0 = evaluate(T0, Tb, Tc, corners, self.obj, self.cam.K, self.cam.dist).T_ee_marker
            T_bc, T_em = refine(T0, T_em0, Tb, corners, self.obj, self.cam.K, self.cam.dist)
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            return None
        T_cb = invert_T(T_bc)
        errs = []
        for T, c in zip(Tb, corners):
            T_cm = T_cb @ T @ T_em
            proj, _ = cv2.projectPoints(self.obj, cv2.Rodrigues(T_cm[:3, :3])[0], T_cm[:3, 3], self.cam.K, self.cam.dist)
            errs.append(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - c) ** 2, axis=1))))
        self.T_base_cam, self.T_ee_marker = T_bc, T_em
        return float(np.mean(errs))

    def _plan_next(self, kin: Kinematics, q0, p_start, q_now) -> np.ndarray | None:
        """The reachable candidate whose marker orientation differs most from every saved view."""
        saved = [np.asarray(v["R_cam_marker"]) for v in self.views]
        spread = np.radians(PHASE2_SPREAD_DEG)
        best, best_score = None, MIN_NEW_DEG
        w, h = self.cam.width, self.cam.height
        for _ in range(CANDIDATES):
            q = q0 + self.rng.uniform(-spread, spread)
            pred = predict_marker(kin.tcp(q), self.T_base_cam, self.T_ee_marker, self.obj, self.cam.K)
            if pred is None:
                continue
            corners, R_cm, tilt = pred
            if not MIN_TILT_DEG <= tilt <= MAX_TILT_DEG:
                continue
            if (np.any(corners < IMAGE_MARGIN_PX) or np.any(corners[:, 0] > w - IMAGE_MARGIN_PX)
                    or np.any(corners[:, 1] > h - IMAGE_MARGIN_PX)):
                continue
            if np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1).mean() < MIN_SIDE_PX:
                continue
            score = min(rotation_angle_deg(R.T @ R_cm) for R in saved)
            if score > best_score and path_is_clear(kin, q_now, q, p_start, self.limits):
                best, best_score = q, score
        return best

    # ---- run --------------------------------------------------------------------------------------------

    def run(self) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "session.json").write_text(json.dumps({
            "date": datetime.now().isoformat(timespec="seconds"), "camera_serial": self.cam.serial,
            "K": self.cam.K.tolist(), "dist": self.cam.dist.tolist(), "depth_scale_m": self.cam.depth_scale,
            "aruco_dictionary": "DICT_4X4_50", "aruco_dictionary_id": int(DICTIONARY), "marker_id": MARKER_ID,
            "marker_side_m": self.side, "pose_source": "RTDE getActualTCPPose (pendant TCP, RTDE base frame)",
            "collection": "automatic"}, indent=2))
        self._check_robot()
        q0 = self._q()
        kin = Kinematics(self.robot, q0, rtde_pose_to_T(self.recv.getActualTCPPose()))
        p_start = kin.flange(q0)[:3, 3]
        if not path_is_clear(kin, q0, q0, p_start, self.limits, samples=1):
            raise AutoStop("the start pose is already outside the limits (table clearance, radius)")
        self.log(f"saving to {self.out}")
        if self._capture_view() is None:
            raise AutoStop("the camera does not see the marker at the start pose: jog the arm so that it does")

        # A halting stop or Ctrl+C leaves this method here, with the arm where it stopped.
        soft_stop = None
        try:
            self._phase1(kin, q0, p_start)
            self._phase2(kin, q0, p_start)
        except AutoStop as exc:
            if exc.halt:
                raise
            soft_stop = exc
        self.log("returning to the start pose")
        self._move(q0, "return to the start pose")
        if soft_stop is not None:
            raise soft_stop
        self.log(f"{len(self.views)} views saved")
        return self.out

    def _phase1(self, kin, q0, p_start) -> None:
        self.log("phase 1: small moves around the start")
        spread = np.radians(PHASE1_SPREAD_DEG)
        missed = 0
        while len(self.views) < PHASE1_MAX_VIEWS:
            q_now = self._q()
            for _ in range(200):
                q = q0 + self.rng.uniform(-spread, spread)
                if path_is_clear(kin, q_now, q, p_start, self.limits):
                    break
            else:
                raise AutoStop("no clear small move from the start pose: raise the arm or check --table-z-mm", halt=False)
            self._move(q, f"small move {len(self.views)}")
            view = self._capture_view()
            missed = 0 if view is not None else missed + 1
            if missed >= MAX_MISSED:
                raise AutoStop("the marker keeps leaving the view; start with it nearer the image centre", halt=False)
            if view is not None and len(self.views) >= PHASE1_MIN_VIEWS:
                err = self._coarse_solve()
                if err is not None and err < COARSE_MAX_REPROJ_PX:
                    self.log(f"  coarse calibration: {err:.2f} px, camera at {np.round(self.T_base_cam[:3, 3], 3)} m")
                    return
        raise AutoStop("no coarse calibration after the small moves: is the marker rigid and --side-mm right?",
                       halt=False)

    def _phase2(self, kin, q0, p_start) -> None:
        self.log("phase 2: moves chosen for varied views")
        missed = 0
        while len(self.views) < self.n_views:
            q = self._plan_next(kin, q0, p_start, self._q())
            if q is None:
                self.log("  no further view differs enough; stopping collection")
                return
            self._move(q, f"view {len(self.views)}")
            view = self._capture_view()
            missed = 0 if view is not None else missed + 1
            if missed >= MAX_MISSED:
                self.log("  the marker keeps leaving the view; stopping collection")
                return
            if view is not None and len(self.views) % 3 == 0:
                self._coarse_solve()
