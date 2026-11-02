"""The pick-and-place state machine on the lab setup: the simulation's PickPlace on a velocity interface.

RealPickPlace reuses PickPlace's states as they are (TRACK with the cube EKF and PBVS, APPROACH, the Cartesian and
joint moves, the holds), with two changes:
    _control_step   sends the joint reference to the robot through src/real/io.py instead of stepping a simulator
    run()           the lab sequence: the camera is fixed, so there is no OBSERVE pose to move to and no zone to
                    search for; the arm starts where it is (out of the camera's view of the table), the cube is
                    found in the first frames, and the place position is given.

The cube rests on the table, so its centre height is known (table + half the edge). Every cube estimate takes
that height instead of the camera's, which with a fixed camera about 0.9 m away carries the calibration error
(0.5 deg is about 8 mm); the camera's height is kept in z_offsets as a check of the calibration and table height.

A cube pushed by hand can turn on its way, while the simulation's cube only translates. After TRACK the arm holds
still, the cube's yaw is measured again, and if it changed the wrist turns at the hover height (ALIGN) before the
approach, so the fingers meet the faces of the cube as it now lies.

mode "track": find the cube, follow it until it stops, and hover there (no descent, gripper untouched).
mode "pick":  the full task: track, approach, descend, grasp, lift, place at place_xy, release, retreat, return.
mode "static": a cube at rest, without vision once the arm has moved. With the arm above the cube its fingers get
              into the cube's mask and depth (on the lab camera 6 of 21 estimates pass, against 21 of 21 with the arm
              clear), so the cube is measured once from the start pose, grasp_offset is added, and the arm goes to the
              pregrasp, straight down, grasps and places, with no further correction. The wrist turns on the way to the
              place so that the cube's faces end up square to the base axes.
"""

import dataclasses

import numpy as np

from src.perception import cube_pose
from src.planning import grasp
from src.planning.grasp import grasp_poses, place_poses
from src.planning.ik import solve_ik
from src.planning.trajectory import QuinticJoint
from src.task.pick_place import PickPlace

JOINT_V_PEAK = 0.25  # rad/s, peak joint speed of the joint-space moves (the simulation uses 0.5)
FIND_CUBE_FRAMES = 30  # vision attempts at the start (3 s at 10 Hz)
FIND_PERIOD = 50       # control periods between attempts
YAW_FRAMES = 7         # frames looked at for the yaw after the stop
YAW_MIN_FRAMES = 3     # valid frames needed to trust them over the EKF yaw
ALIGN_MIN_DEG = 2.0    # a smaller change of grasp yaw is left alone
STATIC_FRAMES = 21     # frames looked at for the cube in mode "static"
STATIC_MIN_FRAMES = 18 # valid frames needed: on the lab camera 21 of 21 with the arm clear, 6 of 21 with it above the cube
STATIC_MOVE_S = 6.0    # s, each vertical move in mode "static" (17 mm/s over the 100 mm hover; the simulation uses 2 s)


class RealPickPlace(PickPlace):
    def __init__(self, io, robot, camera, mode: str = "track", place_xy=None, on_step=None, grasp_offset=(0.0, 0.0)):
        if mode not in ("track", "pick", "static"):
            raise ValueError("mode must be 'track', 'pick' or 'static'")
        if mode in ("pick", "static") and place_xy is None:
            raise ValueError(f"mode '{mode}' needs place_xy")
        super().__init__(io, robot, io, on_step=on_step, camera=camera, approach="track")
        self.io = io
        self.mode = mode
        self.place_xy = None if place_xy is None else np.asarray(place_xy, dtype=float)
        self.q_start = io.xhat[:, 0].copy()
        self.z_offsets = []  # camera cube height minus the known one, per accepted estimate
        self.yaw_at_stop = None  # rad, the cube's yaw measured after TRACK
        self.align_deg = None    # deg, how far that moved the grasp yaw
        self.grasp_offset = np.array([*grasp_offset, 0.0], dtype=float)  # m, world, added to the measured cube (static)
        self.static_cube = None  # the cube measured in mode "static", before the offset

    def _known_height(self, cube):
        z = grasp.TABLE_Z + cube_pose.CUBE_HALF_HEIGHT
        self.z_offsets.append(float(cube.pos[2] - z))
        return dataclasses.replace(cube, pos=np.array([cube.pos[0], cube.pos[1], z]))

    def _capture_cube(self):
        cube = super()._capture_cube()
        return None if cube is None else self._known_height(cube)

    def _control_step(self, q_ref: np.ndarray, qd_ref: np.ndarray) -> None:
        self.io.send(q_ref, qd_ref)
        self.elapsed += self.dt
        if self.on_step is not None:
            self.on_step(self)

    def _move_joint(self, name: str, q_target: np.ndarray) -> bool:
        traj = QuinticJoint(self.ctrl.xhat[:, 0].copy(), q_target, v_max=JOINT_V_PEAK)
        return self._execute_trajectory(name, traj)

    def _find_cube(self):
        """Hold still and try a few frames until the cube is seen with a valid estimate."""
        self.state = "FIND"
        q_hold = self.ctrl.xhat[:, 0].copy()
        for _ in range(FIND_CUBE_FRAMES):
            cube = self._capture_cube()
            if cube is not None:
                self.cube_estimate = cube
                self.records.append({"state": "FIND", "ok": True, "elapsed": 0.0, "reason": None})
                return cube
            for _ in range(FIND_PERIOD):
                self._control_step(q_hold, np.zeros(6))
        self.records.append({"state": "FIND", "ok": False, "elapsed": 0.0, "reason": "cube not detected"})
        return None

    def _yaw_at_stop(self, ekf_yaw: float) -> float:
        """The cube's yaw from a few frames with the arm still, or the EKF's if too few frames are usable.

        A square's yaw repeats every 90 deg, so the frames are averaged on the angle multiplied by four."""
        self.state = "MEASURE_YAW"
        q_hold = self.ctrl.xhat[:, 0].copy()
        yaws = []
        for _ in range(YAW_FRAMES):
            cube = self._capture_cube()
            if cube is not None:
                yaws.append(cube.yaw)
            for _ in range(FIND_PERIOD):
                self._control_step(q_hold, np.zeros(6))
        if len(yaws) < YAW_MIN_FRAMES:
            return ekf_yaw
        return float(np.arctan2(np.mean(np.sin(4.0 * np.array(yaws))), np.mean(np.cos(4.0 * np.array(yaws)))) / 4.0)

    def _realign_grasp(self, cube, psi: float):
        """Choose the grasp yaw for the cube as it lies now and turn the wrist to it. Returns the yaw, or None."""
        yaw = self._yaw_at_stop(cube.yaw)
        self.yaw_at_stop = yaw
        q_now = self.ctrl.xhat[:, 0].copy()
        selected = self._select_grasp(cube.pos, yaw, q_now)
        if selected is None:
            return None
        psi_new = selected[0]
        change = abs(np.degrees(np.arctan2(np.sin(psi_new - psi), np.cos(psi_new - psi))))
        self.align_deg = change
        if change < ALIGN_MIN_DEG:
            return psi
        p_now, _ = self.robot.site_pose(q_now, "tcp")
        q_align, ok = solve_ik(self.robot, p_now, grasp_poses(cube.pos, psi_new)["pregrasp"][1], q_now)
        if not ok or not self._move_joint("ALIGN", q_align):
            return None
        return psi_new

    def _measure_static(self):
        """The cube from several frames with the arm still: median position, yaw averaged on four times the angle."""
        self.state = "MEASURE"
        q_hold = self.ctrl.xhat[:, 0].copy()
        cubes = []
        for _ in range(STATIC_FRAMES):
            cube = self._capture_cube()
            if cube is not None:
                cubes.append(cube)
            for _ in range(FIND_PERIOD):
                self._control_step(q_hold, np.zeros(6))
        ok = len(cubes) >= STATIC_MIN_FRAMES
        self.records.append({"state": "MEASURE", "ok": ok, "elapsed": 0.0,
                             "reason": None if ok else f"{len(cubes)} of {STATIC_FRAMES} frames"})
        if not ok:
            return None
        yaws = 4.0 * np.array([c.yaw for c in cubes])
        return dataclasses.replace(cubes[-1], pos=np.median([c.pos for c in cubes], axis=0),
                                   yaw=float(np.arctan2(np.sin(yaws).mean(), np.cos(yaws).mean()) / 4.0))

    def _square_place(self, q_seed: np.ndarray):
        """The IK above the place for the axis-aligned grasp yaw needing the least joint motion: (yaw, q), or None."""
        best = None
        for k in range(4):
            yaw = k * np.pi / 2.0
            p_above, rotation = place_poses(self.place_xy, yaw)["above"]
            q, ok = solve_ik(self.robot, p_above, rotation, q_seed)
            if ok and (best is None or np.sum(np.abs(q - q_seed)) < np.sum(np.abs(best[1] - q_seed))):
                best = (yaw, q)
        return best

    def _grasp_and_place(self, cube, psi: float, move_s: float = 2.0, square_place: bool = False) -> dict:
        """From above the cube: down, grasp, lift, place, release, retreat, back to the start pose.

        square_place: the fingers, and so the faces of the cube they hold, end up along the base axes."""
        poses = grasp_poses(cube.pos, psi)
        rotation = poses["grasp"][1]
        p_grasp, p_lift = poses["grasp"][0], poses["lift"][0]
        p_pre, _ = self.robot.site_pose(self.ctrl.xhat[:, 0], "tcp")

        if not self._move_cartesian("DESCEND", p_pre, p_grasp, rotation, move_s):
            return self._fail("DESCEND failed")
        if not self._hold("GRASP", 0.5, gripper_closure=1.0):
            return self._fail("GRASP timed out")
        if self.io.grasp_held is False:
            return self._fail("the gripper closed on nothing; the arm stays at the grasp height")
        if not self._move_cartesian("LIFT", p_grasp, p_lift, rotation, move_s):
            return self._fail("LIFT failed")

        q_now = self.ctrl.xhat[:, 0].copy()
        if square_place:
            selected = self._square_place(q_now)
            if selected is None:
                return self._fail("no axis-aligned placement IK solution")
            psi_place, q_above = selected
        else:
            psi_place = psi
            q_above, ok = solve_ik(self.robot, place_poses(self.place_xy, psi)["above"][0],
                                   place_poses(self.place_xy, psi)["above"][1], q_now)
            if not ok:
                return self._fail("placement-above IK did not converge")
        poses = place_poses(self.place_xy, psi_place)
        p_above, place_rotation = poses["above"]
        p_place, p_retreat = poses["place"][0], poses["retreat"][0]
        if not self._move_joint("TRANSPORT", q_above):
            return self._fail("TRANSPORT failed")
        if not self._move_cartesian("PLACE", p_above, p_place, place_rotation, move_s):
            return self._fail("PLACE failed")
        if not self._hold("RELEASE", 0.5, gripper_closure=0.0):
            return self._fail("RELEASE timed out")
        if not self._move_cartesian("RETREAT", p_place, p_retreat, place_rotation, move_s):
            return self._fail("RETREAT failed")
        if not self._move_joint("RETURN", self.q_start):
            return self._fail("return to the start pose failed")
        return {"ok": True, "elapsed": self.elapsed, "states": self.records, "reason": None}

    def _run_static(self) -> dict:
        cube = self._measure_static()
        if cube is None:
            return self._fail("the cube was not measured in enough frames; is the arm in the camera's view of it?")
        self.static_cube = cube
        q_now = self.ctrl.xhat[:, 0].copy()
        target = dataclasses.replace(cube, pos=cube.pos + self.grasp_offset)
        selected = self._select_grasp(target.pos, target.yaw, q_now)
        if selected is None:
            return self._fail("no feasible pregrasp IK solution")
        psi, q_pregrasp = selected
        if not self._move_joint("PREGRASP", q_pregrasp):
            return self._fail("PREGRASP failed")
        return self._grasp_and_place(target, psi, STATIC_MOVE_S, square_place=True)

    def run(self, timeout: float = 120.0) -> dict:
        self.timeout = float(timeout)
        # The RG2 stays at its last width between runs, and the simulation's gripper starts open.
        gripper_closure = 0.0 if self.mode in ("pick", "static") else None
        if not self._hold("START", 0.2, q_hold=self.q_start, gripper_closure=gripper_closure):
            return self._fail("timeout at start")
        if self.mode == "static":
            return self._run_static()

        cube = self._find_cube()
        if cube is None:
            return self._fail("cube was not detected with sufficient quality")

        selected = self._select_grasp(cube.pos, cube.yaw, self.ctrl.xhat[:, 0].copy())
        if selected is None:
            return self._fail("no feasible pregrasp IK solution")
        psi, _ = selected

        cube = self._track(cube, psi)
        if cube is None:
            return self._fail("TRACK failed")

        if self.mode == "track":
            self._hold("HOVER", 2.0)
            return {"ok": True, "elapsed": self.elapsed, "states": self.records, "reason": None}

        psi = self._realign_grasp(cube, psi)
        if psi is None:
            return self._fail("no feasible grasp for the cube's yaw at the stop")
        cube = self._approach_pbvs(cube, psi)
        if cube is None:
            return self._fail("APPROACH failed")
        return self._grasp_and_place(cube, psi)
