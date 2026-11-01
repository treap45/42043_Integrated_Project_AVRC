import dataclasses

import numpy as np

from src.control.pbvs import TRANSIT_V_MAX, TRANSIT_W_MAX, joint_velocity, pbvs_twist, pose_error
from src.estimation.cube_ct_ekf import CubeCTEKF
from src.estimation.cube_ekf import CubeEKF
from src.perception.cube_pose import cube_measurement, estimate_cube
from src.perception.place_zone import estimate_zone
from src.perception.camera_model import CameraModel
from src.perception.projection import backproject, camera_to_world
from src.perception.segmentation import (
    CUBE_HSV,
    ZONE_HSV,
    color_mask,
    largest_component,
)
from src.planning.grasp import grasp_poses, place_poses, yaw_candidates
from src.planning.ik import solve_ik
from src.planning.trajectory import CartesianLine, GoalRelativeQuintic, QuinticJoint

SETTLE_WINDOW = 50  # control periods (0.1 s at 500 Hz)
ZONE_VIEW_MIN_RADIUS = 0.30  # m, keeps the zone-view pose reachable for zones near the base
OBSERVE_TARGET_X = 0.33  # m, table point the camera looks at from the observe pose
GOAL_ALPHA = 0.4  # TRANSIT goal filter gain per vision update
GOAL_GATE = 0.04  # m, a larger jump in the measured goal is treated as an outlier
GOAL_AGREE = 0.01  # m, two consecutive outliers this close are accepted as a real move of the cube
GOAL_UPDATE_PERIOD = 50  # control periods between vision updates (10 Hz at 500 Hz)
TRACK_TIMEOUT = 30.0  # s, TRACK waits this long for the cube to stop, however long it keeps moving
STOP_SPEED = 0.005  # m/s, estimated cube speed below which it may have stopped
STOP_CHI2 = 5.99  # chi-square, 2 degrees of freedom, 95 %: velocity statistically indistinguishable from zero
STOP_POS_TOL = 0.005  # m, TCP to its goal above the cube (horizontal)
STOP_COUNT = 20  # consecutive vision updates (2 s) that must all pass the stop test
_MJ_FROM_CV = np.diag([1.0, -1.0, -1.0])  # MuJoCo camera frame <-> OpenCV camera frame


class PickPlace:
    """Estimate-driven pick-and-place state machine."""

    def __init__(self, sim, robot, ctrl, on_step=None, camera: CameraModel | None = None,
                 approach: str = "track", cube_filter: str = "cv"):
        """on_step(pick_place) is called after every control period, for display and logging only.
        camera: intrinsics and hand-eye transform; defaults to the nominal (uncalibrated) values.
        approach: "track" (default, see below); "open" descends straight from the pregrasp pose planned at OBSERVE
        (look-then-move);
        "pbvs" moves to that pregrasp pose, then servos the TCP on the cube before the final descent;
        "pbvs_full" also servos the move from OBSERVE to the pregrasp pose (TRANSIT) on the cube, meant for
        the occluded levels; "track" follows a moving cube above it with PBVS on the cube EKF and starts the
        approach once the EKF says the cube has stopped (TRACK).
        cube_filter: the TRACK estimator, "cv" (CubeEKF, default) or "ct" (CubeCTEKF, with a turn rate)."""
        if approach not in ("pbvs_full", "pbvs", "open", "track"):
            raise ValueError("approach must be 'pbvs_full', 'pbvs', 'open' or 'track'")
        if cube_filter not in ("cv", "ct"):
            raise ValueError("cube_filter must be 'cv' or 'ct'")
        self.sim = sim
        self.robot = robot
        self.ctrl = ctrl
        self.camera = camera if camera is not None else CameraModel.nominal(robot)
        self.approach = approach
        self.dt = sim.dt
        self.q_observe = ctrl.xhat[:, 0].copy()
        self.elapsed = 0.0
        self.records = []
        self.timeout = 60.0
        self.on_step = on_step
        self.state = "START"
        self.progress = 0.0  # fraction of the current trajectory, for disturbance timing in experiments
        self.transit_log = []
        self.track_log = []
        self.track_stop_time = None  # elapsed time at which TRACK declared the cube stopped
        self.cube_ekf = None
        self.cube_estimate = None
        self.zone_estimate = None
        self.cube_filter = cube_filter

    def _control_step(self, q_ref: np.ndarray, qd_ref: np.ndarray) -> None:
        y = self.sim.read_joint_positions()
        tau = self.ctrl.step(y, q_ref, qd_ref)
        self.sim.step(tau)
        self.elapsed += self.dt
        if self.on_step is not None:
            self.on_step(self)

    def _hold(
        self,
        name: str,
        duration: float,
        q_hold: np.ndarray | None = None,
        gripper_closure: float | None = None,
    ) -> bool:
        """Hold a joint reference while continuing the feedback loop."""
        self.state = name
        start_time = self.elapsed
        if q_hold is None:
            q_hold = self.ctrl.xhat[:, 0].copy()
        q_hold = np.asarray(q_hold, dtype=float)

        if gripper_closure is not None:
            self.sim.set_gripper(gripper_closure)

        steps = int(np.ceil(duration / self.dt))
        for _ in range(steps):
            if self.elapsed >= self.timeout:
                self.records.append({
                    "state": name,
                    "ok": False,
                    "elapsed": self.elapsed - start_time,
                    "reason": "overall timeout",
                })
                return False

            self._control_step(q_hold, np.zeros(6))

        self.records.append({
            "state": name,
            "ok": True,
            "elapsed": self.elapsed - start_time,
            "reason": None,
        })
        return True

    def _execute_trajectory(
        self,
        name: str,
        traj,
        timeout_extra: float = 3.0,
    ) -> bool:
        """Execute one trajectory and wait for estimated settling."""
        self.state = name
        start_time = self.elapsed
        traj_time = 0.0
        state_timeout = traj.T + timeout_extra
        # A single velocity estimate is too noisy at rest (std about 0.045 rad/s), so settling is
        # judged on its mean over the last SETTLE_WINDOW samples.
        qd_window = []

        while traj_time <= state_timeout and self.elapsed < self.timeout:
            q_ref, qd_ref = traj(traj_time)
            # Time fraction; equals the quintic path fraction s at the midpoint, so "progress >= 0.5" is the
            # same instant here and in TRANSIT.
            self.progress = min(traj_time / traj.T, 1.0)
            self._control_step(q_ref, qd_ref)
            traj_time += self.dt

            xhat = self.ctrl.xhat
            qd_window.append(xhat[:, 1].copy())
            qd_window = qd_window[-SETTLE_WINDOW:]
            settled = (
                traj_time >= traj.T
                and np.max(np.abs(xhat[:, 0] - q_ref)) < 2e-3
                and np.max(np.abs(np.mean(qd_window, axis=0))) < 0.02
            )
            if settled:
                self.records.append({
                    "state": name,
                    "ok": True,
                    "elapsed": self.elapsed - start_time,
                    "reason": None,
                })
                return True

        reason = "overall timeout" if self.elapsed >= self.timeout else "state timeout"
        self.records.append({
            "state": name,
            "ok": False,
            "elapsed": self.elapsed - start_time,
            "reason": reason,
        })
        return False

    def _capture_estimates(self):
        """Capture one stopped-arm frame and estimate the cube and placement zone."""
        rgb, depth = self.sim.read_camera()
        qhat = self.ctrl.xhat[:, 0].copy()
        cam_pos, cam_rot = self.camera.pose(self.robot, qhat)

        cube_mask = largest_component(
            color_mask(rgb, **CUBE_HSV),
            min_pixels=100,
        )
        zone_mask = largest_component(
            color_mask(rgb, **ZONE_HSV),
            min_pixels=100,
        )

        cube_estimate = None
        if cube_mask is not None:
            points_cv = backproject(cube_mask, depth, self.camera.K)
            points_w = camera_to_world(points_cv, cam_pos, cam_rot)
            cube_estimate = estimate_cube(points_w)

        zone_estimate = None
        if zone_mask is not None:
            points_cv = backproject(zone_mask, depth, self.camera.K)
            points_w = camera_to_world(points_cv, cam_pos, cam_rot)
            zone_estimate = estimate_zone(points_w, zone_mask)

        if cube_estimate is not None:
            self.cube_estimate = cube_estimate
        if zone_estimate is not None:
            self.zone_estimate = zone_estimate
        return cube_estimate, zone_estimate

    def _capture_cube(self):
        """Capture one frame and return a valid cube estimate, or None."""
        rgb, depth = self.sim.read_camera()
        qhat = self.ctrl.xhat[:, 0].copy()

        mask = largest_component(
            color_mask(rgb, **CUBE_HSV),
            min_pixels=100,
        )
        if mask is None:
            return None

        points_cv = backproject(mask, depth, self.camera.K)
        if len(points_cv) == 0:
            return None

        cam_pos, cam_rot = self.camera.pose(self.robot, qhat)
        estimate = estimate_cube(camera_to_world(points_cv, cam_pos, cam_rot))
        return estimate if estimate.ok else None

    def _measure_cube(self):
        """Capture one frame for the cube EKF.

        Returns (cube estimate, measurement [u, v, Z, psi], camera position, OpenCV camera rotation), or None.
        """
        rgb, depth = self.sim.read_camera()
        qhat = self.ctrl.xhat[:, 0].copy()

        mask = largest_component(
            color_mask(rgb, **CUBE_HSV),
            min_pixels=100,
        )
        if mask is None:
            return None

        points_cv = backproject(mask, depth, self.camera.K)
        if len(points_cv) == 0:
            return None

        cam_pos, cam_rot = self.camera.pose(self.robot, qhat)
        points_w = camera_to_world(points_cv, cam_pos, cam_rot)
        cube = estimate_cube(points_w)
        z = cube_measurement(points_cv, points_w, cube, self.camera.K)
        if z is None:
            return None
        return cube, z, cam_pos, cam_rot @ _MJ_FROM_CV

    def _move_joint(self, name: str, q_target: np.ndarray) -> bool:
        q_start = self.ctrl.xhat[:, 0].copy()
        traj = QuinticJoint(q_start, q_target)
        return self._execute_trajectory(name, traj)

    def _move_cartesian(
        self,
        name: str,
        p0: np.ndarray,
        p1: np.ndarray,
        R: np.ndarray,
        duration: float = 2.0,
    ) -> bool:
        q_start = self.ctrl.xhat[:, 0].copy()
        try:
            traj = CartesianLine(
                self.robot,
                q_start,
                p0,
                p1,
                R,
                T=duration,
            )
        except ValueError as exc:
            self.records.append({
                "state": name,
                "ok": False,
                "elapsed": 0.0,
                "reason": str(exc),
            })
            return False
        return self._execute_trajectory(name, traj)

    def _update_goal(self, g_hat, psi, pending):
        """One vision update of the filtered pregrasp goal g_hat.

        Returns (g_hat, cube or None, pending, status). A measurement more than GOAL_GATE from g_hat is held
        in `pending` and accepted only if the next measurement agrees with it within GOAL_AGREE.
        """
        cube = self._capture_cube()

        if cube is None:
            return g_hat, None, None, "none"

        g_meas = grasp_poses(cube.pos, psi)["pregrasp"][0]

        if np.linalg.norm(g_meas - g_hat) <= GOAL_GATE:
            g_new = g_hat + GOAL_ALPHA * (g_meas - g_hat)
            return g_new, cube, None, "accepted"

        if pending is not None and np.linalg.norm(g_meas - pending) < GOAL_AGREE:
            return g_meas, cube, None, "reacquired"

        return g_hat, None, g_meas, "rejected"

    def _servo_step(self, q_ref, q_hat, twist):
        """Integrate the DLS joint velocity of a twist into q_ref; freeze q_ref while the joints lag by 0.05 rad.

        Returns (q_ref, qd_ref, frozen).
        """
        J = self.robot.site_jacobian(q_hat, "tcp")
        qd_cmd = joint_velocity(J, twist)

        candidate = q_ref + qd_cmd * self.dt

        ref_close = np.max(np.abs(q_ref - q_hat)) < 0.05
        candidate_close = np.max(np.abs(candidate - q_hat)) < 0.05

        if ref_close and candidate_close:
            return candidate, qd_cmd, False

        return q_ref.copy(), np.zeros_like(q_ref), True

    def _transit_pbvs(self, cube, psi, enable_goal_updates=True):
        """Servo the TCP from its current pose to the pregrasp pose along a reference that follows the goal.

        Returns the latest valid cube estimate, or None on timeout.
        """
        self.state = "TRANSIT"
        self.progress = 0.0
        start_time = self.elapsed

        q_ref = self.ctrl.xhat[:, 0].copy()
        p0, R0 = self.robot.site_pose(q_ref, "tcp")
        g_hat, R_goal = grasp_poses(cube.pos, psi)["pregrasp"]

        traj = GoalRelativeQuintic(p0, R0, g_hat, R_goal)
        last_good = cube
        pending = None

        self.transit_log = []
        t = 0.0
        k = 0
        settled_steps = 0
        last_status = "not_sampled"

        while t < traj.T + 3.0 and self.elapsed < self.timeout:
            vision_attempt = enable_goal_updates and k % GOAL_UPDATE_PERIOD == 0

            if vision_attempt:
                g_hat, detected_cube, pending, last_status = self._update_goal(
                    g_hat, psi, pending
                )
                if detected_cube is not None:
                    last_good = detected_cube
                    self.cube_estimate = detected_cube

            q_hat = self.ctrl.xhat[:, 0].copy()
            p, R = self.robot.site_pose(q_hat, "tcp")
            p_ref, R_ref, feedforward, s = traj(t, g_hat)
            self.progress = s

            error = pose_error(p, R, p_ref, R_ref)
            twist = pbvs_twist(
                error,
                feedforward=feedforward,
                v_max=TRANSIT_V_MAX,
                w_max=TRANSIT_W_MAX,
            )

            q_ref, qd_ref, frozen = self._servo_step(q_ref, q_hat, twist)

            e_p = np.linalg.norm(error[:3])
            e_R = np.linalg.norm(error[3:])

            self.transit_log.append({
                "t": t,
                "s": s,
                "p": p.copy(),
                "p_ref": p_ref.copy(),
                "g_hat": g_hat.copy(),
                "status": last_status if vision_attempt else "not_sampled",
                "vision_attempt": vision_attempt,
                "e_p": e_p,
                "e_R": e_R,
                "frozen": frozen,
            })

            in_tolerance = (
                t >= traj.T
                and e_p < 0.005
                and e_R < 0.05
            )
            settled_steps = settled_steps + 1 if in_tolerance else 0

            self._control_step(q_ref, qd_ref)

            if settled_steps >= 5:
                self.progress = 1.0
                self.records.append({
                    "state": "TRANSIT",
                    "ok": True,
                    "elapsed": self.elapsed - start_time,
                    "reason": None,
                })
                return last_good

            t += self.dt
            k += 1

        self.records.append({
            "state": "TRANSIT",
            "ok": False,
            "elapsed": self.elapsed - start_time,
            "reason": "overall timeout" if self.elapsed >= self.timeout else "state timeout",
        })
        return None


    def _track(self, cube, psi):
        """Follow the cube above it until it stops: PBVS on the cube EKF, with the cube velocity fed forward.

        The reference starts as a quintic from the current TCP pose to the pregrasp pose above the EKF estimate
        and moves with that estimate; once the quintic has finished, the TCP hovers above the cube. The cube is
        declared stopped when, for STOP_COUNT consecutive vision updates, its estimated speed is below STOP_SPEED,
        the velocity is within the 95 % chi-square bound of zero and the TCP is on its goal.

        Returns the cube estimate at the stop (EKF position, height from OBSERVE), or None on timeout.
        """
        self.state = "TRACK"
        self.progress = 0.0
        start_time = self.elapsed

        filter_class = CubeCTEKF if self.cube_filter == "ct" else CubeEKF
        self.cube_ekf = filter_class(np.array([cube.pos[0], cube.pos[1], cube.yaw, 0.0, 0.0]), self.dt)
        z_cube = cube.pos[2]  # the cube slides on the table, so its height stays at the OBSERVE estimate

        def pregrasp_goal():
            return grasp_poses(np.array([*self.cube_ekf.position, z_cube]), psi)["pregrasp"][0]

        q_ref = self.ctrl.xhat[:, 0].copy()
        p0, R0 = self.robot.site_pose(q_ref, "tcp")
        # The orientation goal stays the grasp yaw chosen at OBSERVE: the EKF yaw is wrapped to +/-45 deg and
        # could flip by 90 deg near the wrap, which would swing the wrist for an equivalent grasp.
        R_goal = grasp_poses(cube.pos, psi)["pregrasp"][1]
        traj = GoalRelativeQuintic(p0, R0, pregrasp_goal(), R_goal)

        last_cube = cube
        self.track_log = []
        still = 0
        t = 0.0
        k = 0

        while t < TRACK_TIMEOUT and self.elapsed < self.timeout:
            self.cube_ekf.predict()
            vision_update = k % GOAL_UPDATE_PERIOD == 0
            status = None
            if vision_update:
                m = self._measure_cube()
                status = "none" if m is None else self.cube_ekf.update(m[1], self.camera.K, m[2], m[3])
                if m is not None and m[0].ok:
                    last_cube = m[0]
                # Shown on screen as the cube estimate: the EKF position, not the single frame.
                self.cube_estimate = dataclasses.replace(
                    last_cube, pos=np.array([*self.cube_ekf.position, z_cube]), yaw=self.cube_ekf.yaw)

            q_hat = self.ctrl.xhat[:, 0].copy()
            p, R = self.robot.site_pose(q_hat, "tcp")
            p_ref, R_ref, feedforward, s = traj(t, pregrasp_goal())
            self.progress = s

            # The reference moves with the cube: add its estimated velocity to the quintic's feed-forward.
            feedforward = feedforward + np.array([*self.cube_ekf.velocity, 0.0, 0.0, 0.0, 0.0])
            error = pose_error(p, R, p_ref, R_ref)
            twist = pbvs_twist(
                error,
                feedforward=feedforward,
                v_max=TRANSIT_V_MAX,
                w_max=TRANSIT_W_MAX,
            )
            q_ref, qd_ref, frozen = self._servo_step(q_ref, q_hat, twist)

            if vision_update:
                speed = float(np.linalg.norm(self.cube_ekf.velocity))
                stopped = (
                    s >= 1.0
                    and speed < STOP_SPEED
                    and self.cube_ekf.velocity_test() < STOP_CHI2
                    and np.linalg.norm(error[:2]) < STOP_POS_TOL
                )
                still = still + 1 if stopped else 0
                self.track_log.append({
                    "t": t,
                    "status": status,
                    "speed": speed,
                    "e_p": float(np.linalg.norm(error[:3])),
                    "frozen": frozen,
                    "still": still,
                })

            self._control_step(q_ref, qd_ref)

            if still >= STOP_COUNT:
                self.progress = 1.0
                self.track_stop_time = self.elapsed
                self.records.append({
                    "state": "TRACK",
                    "ok": True,
                    "elapsed": self.elapsed - start_time,
                    "reason": None,
                })
                return self.cube_estimate

            t += self.dt
            k += 1

        self.records.append({
            "state": "TRACK",
            "ok": False,
            "elapsed": self.elapsed - start_time,
            "reason": "overall timeout" if self.elapsed >= self.timeout else "cube did not stop in time",
        })
        return None

    def _approach_pbvs(
        self,
        cube,
        psi: float,
        h_start: float = 0.095,
        h_end: float = 0.060,
        timeout: float = 5.0,
    ):
        """Servo the TCP above the cube while reducing the target height.

        Returns the latest valid cube estimate, or None on timeout.
        """
        self.state = "APPROACH"
        start_time = self.elapsed

        q_ref = self.ctrl.xhat[:, 0].copy()
        last_good = cube
        _, R_des = grasp_poses(cube.pos, psi)["grasp"]

        h = h_start
        t = 0.0
        k = 0
        stable_count = 0

        while t < timeout and self.elapsed < self.timeout:
            if k % 50 == 0:
                new_estimate = self._capture_cube()
                if new_estimate is not None:
                    last_good = new_estimate
                    self.cube_estimate = new_estimate

            h = max(h_end, h - 0.02 * self.dt)
            target_z_velocity = -0.02 if h > h_end else 0.0

            p_des = last_good.pos + np.array([0.0, 0.0, h])

            qhat = self.ctrl.xhat[:, 0].copy()
            p, R = self.robot.site_pose(qhat, "tcp")
            error = pose_error(p, R, p_des, R_des)

            feedforward = np.array([
                0.0,
                0.0,
                target_z_velocity,
                0.0,
                0.0,
                0.0,
            ])
            twist = pbvs_twist(error, feedforward=feedforward)
            q_ref, qd_ref, _ = self._servo_step(q_ref, qhat, twist)
            self._control_step(q_ref, qd_ref)

            height_reached = h <= h_end + 1e-12
            position_ok = np.linalg.norm(error[:3]) < 0.003
            orientation_ok = np.linalg.norm(error[3:]) < 0.03

            if height_reached and position_ok and orientation_ok:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= 5:
                self.records.append({
                    "state": "APPROACH",
                    "ok": True,
                    "elapsed": self.elapsed - start_time,
                    "reason": None,
                })
                return last_good

            t += self.dt
            k += 1

        self.records.append({
            "state": "APPROACH",
            "ok": False,
            "elapsed": self.elapsed - start_time,
            "reason": "overall timeout" if self.elapsed >= self.timeout else "state timeout",
        })
        return None

    def _select_grasp(self, cube_pos, cube_yaw, q_seed):
        """Select the feasible yaw candidate requiring the least joint motion."""
        candidates = []

        for psi in yaw_candidates(cube_yaw):
            pregrasp_pos, rotation = grasp_poses(cube_pos, psi)["pregrasp"]
            q_solution, ok = solve_ik(
                self.robot,
                pregrasp_pos,
                rotation,
                q_seed,
            )
            if ok:
                candidates.append((
                    np.sum(np.abs(q_solution - q_seed)),
                    psi,
                    q_solution,
                ))

        if not candidates:
            return None

        _, psi, q_pregrasp = min(candidates, key=lambda item: item[0])
        return psi, q_pregrasp

    def _fail(self, reason: str) -> dict:
        return {
            "ok": False,
            "elapsed": self.elapsed,
            "states": self.records,
            "reason": reason,
        }

    def run(self, timeout: float = 60.0) -> dict:
        """Run one pick-and-place attempt without consulting simulator ground truth."""
        self.timeout = float(timeout)

        if not self._hold("OBSERVE", 0.2, q_hold=self.q_observe):
            return self._fail("timeout while stabilising at observe pose")

        cube, zone = self._capture_estimates()
        if cube is None or not cube.ok:
            return self._fail("cube was not detected with sufficient quality")
        if zone is None or not np.all(np.isfinite(zone.xy)):
            return self._fail("placement zone was not detected")

        # Re-observe a partly seen zone before grasping: once the cube is held it covers the
        # bottom centre of the wrist image, which is where the zone would appear.
        if not zone.complete:
            q_current = self.ctrl.xhat[:, 0].copy()
            p_observe, R_observe = self.robot.site_pose(self.q_observe)

            # Rotate the observe pose about the base axis so that the camera looks at the zone.
            phi = np.arctan2(zone.xy[1], zone.xy[0])
            radius = max(np.hypot(zone.xy[0], zone.xy[1]), ZONE_VIEW_MIN_RADIUS)
            Rz = np.array([
                [np.cos(phi), -np.sin(phi), 0.0],
                [np.sin(phi), np.cos(phi), 0.0],
                [0.0, 0.0, 1.0],
            ])
            p_zone_view = Rz @ (p_observe + np.array([radius - OBSERVE_TARGET_X, 0.0, 0.0]))
            R_zone_view = Rz @ R_observe

            q_zone_view, ok = solve_ik(
                self.robot,
                p_zone_view,
                R_zone_view,
                q_current,
            )
            if not ok:
                return self._fail("ZONE_VIEW IK did not converge")
            if not self._move_joint("ZONE_VIEW", q_zone_view):
                return self._fail("ZONE_VIEW failed")
            if not self._hold("ZONE_CAPTURE", 0.1):
                return self._fail("ZONE_CAPTURE timed out")

            _, zone = self._capture_estimates()
            if zone is None or not zone.complete:
                return self._fail("placement zone remains incomplete after ZONE_VIEW")

        q_now = self.ctrl.xhat[:, 0].copy()
        selected = self._select_grasp(cube.pos, cube.yaw, q_now)
        if selected is None:
            return self._fail("no feasible pregrasp IK solution")

        psi, q_pregrasp = selected
        poses = grasp_poses(cube.pos, psi)
        p_pre, rotation = poses["pregrasp"]
        p_grasp = poses["grasp"][0]
        p_lift = poses["lift"][0]

        q_grasp, ok = solve_ik(self.robot, p_grasp, rotation, q_pregrasp)
        if not ok:
            return self._fail("grasp pose IK did not converge")

        if self.approach == "pbvs_full":
            cube = self._transit_pbvs(cube, psi)
            if cube is None:
                return self._fail("TRANSIT failed")
        elif self.approach == "track":
            cube = self._track(cube, psi)
            if cube is None:
                return self._fail("TRACK failed")
        elif not self._move_joint("PREGRASP", q_pregrasp):
            return self._fail("PREGRASP failed")

        if self.approach in ("pbvs", "pbvs_full", "track"):
            cube = self._approach_pbvs(cube, psi)
            if cube is None:
                return self._fail("APPROACH failed")
            # Final descent from where PBVS stopped, to the grasp pose of the latest estimate.
            poses = grasp_poses(cube.pos, psi)
            p_grasp = poses["grasp"][0]
            p_lift = poses["lift"][0]
            p_pre, _ = self.robot.site_pose(self.ctrl.xhat[:, 0], "tcp")

        if not self._move_cartesian("DESCEND", p_pre, p_grasp, rotation, 1.5):
            return self._fail("DESCEND failed")

        if not self._hold(
            "GRASP",
            0.8,
            q_hold=self.ctrl.xhat[:, 0],
            gripper_closure=1.0,
        ):
            return self._fail("GRASP timed out")

        if not self._move_cartesian("LIFT", p_grasp, p_lift, rotation, 1.5):
            return self._fail("LIFT failed")

        poses = place_poses(zone.xy, psi)
        p_above, place_rotation = poses["above"]
        p_place = poses["place"][0]
        p_retreat = poses["retreat"][0]

        q_current = self.ctrl.xhat[:, 0].copy()
        q_above, ok = solve_ik(
            self.robot,
            p_above,
            place_rotation,
            q_current,
        )
        if not ok:
            return self._fail("placement-above IK did not converge")
        if not self._move_joint("TRANSPORT", q_above):
            return self._fail("TRANSPORT failed")

        if not self._move_cartesian(
            "PLACE",
            p_above,
            p_place,
            place_rotation,
            1.5,
        ):
            return self._fail("PLACE failed")

        if not self._hold(
            "RELEASE",
            0.5,
            q_hold=self.ctrl.xhat[:, 0],
            gripper_closure=0.0,
        ):
            return self._fail("RELEASE timed out")

        if not self._move_cartesian(
            "RETREAT",
            p_place,
            p_retreat,
            place_rotation,
            1.5,
        ):
            return self._fail("RETREAT lift failed")

        if not self._move_joint("RETURN_TO_OBSERVE", self.q_observe):
            return self._fail("return to observe pose failed")

        return {
            "ok": True,
            "elapsed": self.elapsed,
            "states": self.records,
            "reason": None,
        }
