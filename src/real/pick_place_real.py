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

mode "track": find the cube, follow it until it stops, and hover there (no descent, gripper untouched).
mode "pick":  the full task: track, approach, descend, grasp, lift, place at place_xy, release, retreat, return.
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


class RealPickPlace(PickPlace):
    def __init__(self, io, robot, camera, mode: str = "track", place_xy=None, on_step=None):
        if mode not in ("track", "pick"):
            raise ValueError("mode must be 'track' or 'pick'")
        if mode == "pick" and place_xy is None:
            raise ValueError("mode 'pick' needs place_xy")
        super().__init__(io, robot, io, on_step=on_step, camera=camera, approach="track")
        self.io = io
        self.mode = mode
        self.place_xy = None if place_xy is None else np.asarray(place_xy, dtype=float)
        self.q_start = io.xhat[:, 0].copy()
        self.z_offsets = []  # camera cube height minus the known one, per accepted estimate

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

    def run(self, timeout: float = 120.0) -> dict:
        self.timeout = float(timeout)
        if not self._hold("START", 0.2, q_hold=self.q_start):
            return self._fail("timeout at start")

        cube = self._find_cube()
        if cube is None:
            return self._fail("cube was not detected with sufficient quality")

        selected = self._select_grasp(cube.pos, cube.yaw, self.ctrl.xhat[:, 0].copy())
        if selected is None:
            return self._fail("no feasible pregrasp IK solution")
        psi, _ = selected
        rotation = grasp_poses(cube.pos, psi)["pregrasp"][1]

        cube = self._track(cube, psi)
        if cube is None:
            return self._fail("TRACK failed")

        if self.mode == "track":
            self._hold("HOVER", 2.0)
            return {"ok": True, "elapsed": self.elapsed, "states": self.records, "reason": None}

        cube = self._approach_pbvs(cube, psi)
        if cube is None:
            return self._fail("APPROACH failed")
        poses = grasp_poses(cube.pos, psi)
        p_grasp, p_lift = poses["grasp"][0], poses["lift"][0]
        p_pre, _ = self.robot.site_pose(self.ctrl.xhat[:, 0], "tcp")

        if not self._move_cartesian("DESCEND", p_pre, p_grasp, rotation, 2.0):
            return self._fail("DESCEND failed")
        if not self._hold("GRASP", 0.5, gripper_closure=1.0):
            return self._fail("GRASP timed out")
        if not self._move_cartesian("LIFT", p_grasp, p_lift, rotation, 2.0):
            return self._fail("LIFT failed")

        poses = place_poses(self.place_xy, psi)
        p_above, place_rotation = poses["above"]
        p_place, p_retreat = poses["place"][0], poses["retreat"][0]
        q_above, ok = solve_ik(self.robot, p_above, place_rotation, self.ctrl.xhat[:, 0].copy())
        if not ok:
            return self._fail("placement-above IK did not converge")
        if not self._move_joint("TRANSPORT", q_above):
            return self._fail("TRANSPORT failed")
        if not self._move_cartesian("PLACE", p_above, p_place, place_rotation, 2.0):
            return self._fail("PLACE failed")
        if not self._hold("RELEASE", 0.5, gripper_closure=0.0):
            return self._fail("RELEASE timed out")
        if not self._move_cartesian("RETREAT", p_place, p_retreat, place_rotation, 2.0):
            return self._fail("RETREAT failed")
        if not self._move_joint("RETURN", self.q_start):
            return self._fail("return to the start pose failed")
        return {"ok": True, "elapsed": self.elapsed, "states": self.records, "reason": None}
