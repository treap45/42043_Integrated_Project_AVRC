"""Calibrate the wrist camera in simulation: intrinsics, then the hand-eye transform.

Collects checkerboard images (src/sim TaskSim.place_calib_board), runs src/perception/calibration.py, compares
the results with ground truth and saves them to configs/camera_calib.json. Ground truth is read only in the
evaluate_* functions.

1. Intrinsics: the arm stays at the observe pose and the board is placed at varied positions, tilts and yaws,
   like moving a printed board by hand in front of a fixed camera.
2. Hand-eye: the board lies flat on the table and the arm is driven to varied poses by the joint controller.
   The end-effector pose comes from FK of the encoder readings averaged once the arm has settled, never from
   the simulator state.

Run from the repo root with the venv active:
    python scripts/calibrate_camera.py
    python scripts/calibrate_camera.py --camera-noise
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.control.arm_controller import ArmController  # noqa: E402
from src.model.ur3e import UR3eModel  # noqa: E402
from src.perception import calibration  # noqa: E402
from src.planning.trajectory import QuinticJoint  # noqa: E402
from src.sim import CameraNoiseConfig, TaskSim  # noqa: E402
from src.sim.env import CALIB_SQUARE, CALIB_SQUARES, CALIB_THICKNESS  # noqa: E402
from src.sim.randomize import TABLE_Z  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "configs/camera_calib.json"
BOARD_Z = TABLE_Z + CALIB_THICKNESS / 2  # board centre when lying flat on the table
BOARD_HALF = np.array([(CALIB_SQUARES[0] + 2) * CALIB_SQUARE / 2, (CALIB_SQUARES[1] + 2) * CALIB_SQUARE / 2])
HAND_EYE_BOARD_POS = np.array([0.36, 0.0, BOARD_Z])
EE_SITE = "tcp"  # any frame rigidly attached to the flange works; the evaluation uses the same one
MJ_FROM_CV = np.diag([1.0, -1.0, -1.0])

N_INTRINSIC_VIEWS = 20
N_HAND_EYE_VIEWS = 15
SEED = 0

# Intrinsic views: board centre and orientation ranges around the point the observe pose looks at.
BOARD_X = (0.31, 0.41)  # m
BOARD_Y = (-0.04, 0.04)  # m
BOARD_LIFT = (0.0, 0.04)  # m, on top of the clearance needed for the tilt
BOARD_TILT_DEG = 30.0  # about the world x and y axes
BOARD_YAW_DEG = 30.0

# Hand-eye views: joint offsets from the observe pose (rad). Wrist offsets give the rotation variety
# calibrateHandEye needs; the shoulder and elbow move the camera around the board.
JOINT_OFFSET = np.array([0.15, 0.10, 0.10, 0.30, 0.30, 0.50])
VIEW_MARGIN_PX = 25
VIEW_MAX_V = 400  # gripper fingers occupy the bottom of the image

# Settling (same idea as src/task/pick_place.py): reference reached and mean velocity estimate small.
SETTLE_WINDOW = 50
SETTLE_POS = 2e-3  # rad
SETTLE_VEL = 0.02  # rad/s
SETTLE_TIMEOUT = 3.0  # s after the trajectory ends
N_AVERAGE = 50  # encoder samples averaged for the FK of each view


def quat_from_euler_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.deg2rad([rx, ry, rz]), "XYZ")
    return q


def board_corners_world(pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    """The four outer corners of the board's printed face in the world frame."""
    sx, sy = BOARD_HALF
    local = np.array([[sx, sy, 0], [sx, -sy, 0], [-sx, sy, 0], [-sx, -sy, 0]]) + [0, 0, CALIB_THICKNESS / 2]
    return local @ R.T + pos


# ---- 1. intrinsics -------------------------------------------------------------------------------

def intrinsic_board_poses(rng: np.random.Generator, n: int = 3 * N_INTRINSIC_VIEWS) -> list[tuple]:
    """Board poses (pos, quat (w, x, y, z)) seen from the observe pose, varied in position, tilt and yaw.

    More poses than needed are drawn; collect_intrinsic_views keeps the first N_INTRINSIC_VIEWS in which the
    whole board is detected. Each board is lifted just enough that no part of it is below the table top.
    """
    poses = []
    for _ in range(n):
        quat = quat_from_euler_deg(*rng.uniform(-BOARD_TILT_DEG, BOARD_TILT_DEG, 2),
                                   rng.uniform(-BOARD_YAW_DEG, BOARD_YAW_DEG))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        pos = np.array([rng.uniform(*BOARD_X), rng.uniform(*BOARD_Y), BOARD_Z])
        drop = BOARD_Z - board_corners_world(pos, R.reshape(3, 3))[:, 2].min()
        pos[2] += max(drop, 0.0) + 0.002 + rng.uniform(*BOARD_LIFT)
        poses.append((pos, quat))
    return poses


def collect_intrinsic_views(sim: TaskSim, poses: list[tuple],
                            n_views: int = N_INTRINSIC_VIEWS) -> tuple[list[np.ndarray], tuple[int, int]]:
    """Place the board at each pose, capture a frame, detect the corners.

    Returns the detected corner sets and the image size (width, height).
    """
    corner_sets = []
    tried = 0
    for pos, quat in poses:
        if len(corner_sets) == n_views:
            break
        tried += 1
        sim.place_calib_board(pos, quat)
        rgb, _ = sim.read_camera()
        corners = calibration.detect_corners(rgb)
        if corners is not None:
            corner_sets.append(corners)
    print(f"intrinsics: board detected in {len(corner_sets)} of {tried} placements")
    if len(corner_sets) < n_views:
        raise RuntimeError(f"only {len(corner_sets)} usable views, need {n_views}")
    return corner_sets, (sim.camera.width, sim.camera.height)


# ---- 2. hand-eye ---------------------------------------------------------------------------------

def board_in_view(robot: UR3eModel, q: np.ndarray, K: np.ndarray) -> bool:
    """Predict whether the whole board (at the position we placed it) is inside the image at joint pose q.

    Uses the controller-side model's nominal camera mount and the calibrated K, like aiming a real camera
    from CAD data; the hand-eye result itself does not depend on this prediction.
    """
    cam_pos, cam_rot = robot.camera_pose(q)
    p_cv = (board_corners_world(HAND_EYE_BOARD_POS, np.eye(3)) - cam_pos) @ cam_rot @ MJ_FROM_CV
    if np.any(p_cv[:, 2] <= 0.05):
        return False
    uv = (K @ (p_cv / p_cv[:, 2:]).T).T[:, :2]
    w, h = 2 * K[0, 2] + 1, 2 * K[1, 2] + 1
    return bool(np.all(uv[:, 0] > VIEW_MARGIN_PX) and np.all(uv[:, 0] < w - VIEW_MARGIN_PX)
                and np.all(uv[:, 1] > VIEW_MARGIN_PX) and np.all(uv[:, 1] < min(h - VIEW_MARGIN_PX, VIEW_MAX_V)))


def hand_eye_joint_targets(q_observe: np.ndarray, rng: np.random.Generator, robot: UR3eModel, K: np.ndarray,
                           n: int = N_HAND_EYE_VIEWS, max_tries: int = 5000) -> list[np.ndarray]:
    """Joint targets around the observe pose with the board predicted fully in view."""
    targets = []
    for _ in range(max_tries):
        q = q_observe + rng.uniform(-1.0, 1.0, 6) * JOINT_OFFSET
        if board_in_view(robot, q, K):
            targets.append(q)
            if len(targets) == n:
                return targets
    raise RuntimeError(f"found only {len(targets)} joint targets with the board in view")


def move_and_settle(sim: TaskSim, ctrl: ArmController, q_target: np.ndarray) -> np.ndarray | None:
    """Drive the arm to q_target with the joint controller and wait until it has settled.

    Returns the encoder readings averaged over N_AVERAGE control periods once settled (the arm is held at
    q_target meanwhile), or None if it does not settle in time.
    """
    traj = QuinticJoint(ctrl.xhat[:, 0].copy(), q_target)
    t, qd_window = 0.0, []
    while True:
        q_ref, qd_ref = traj(t)
        ctrl_y = sim.read_joint_positions()
        sim.step(ctrl.step(ctrl_y, q_ref, qd_ref))
        t += sim.dt
        qd_window = (qd_window + [ctrl.xhat[:, 1].copy()])[-SETTLE_WINDOW:]
        if (t >= traj.T and np.max(np.abs(ctrl.xhat[:, 0] - q_ref)) < SETTLE_POS
                and np.max(np.abs(np.mean(qd_window, axis=0))) < SETTLE_VEL):
            break
        if t > traj.T + SETTLE_TIMEOUT:
            return None

    readings = []
    for _ in range(N_AVERAGE):
        y = sim.read_joint_positions()
        readings.append(y)
        sim.step(ctrl.step(y, q_target, np.zeros(6)))
    return np.mean(readings, axis=0)


def collect_hand_eye_views(sim: TaskSim, robot: UR3eModel, ctrl: ArmController, targets: list[np.ndarray],
                           K: np.ndarray, dist: np.ndarray) -> dict:
    """At each target: settle, FK of the averaged encoder reading for the end-effector pose, board pose from
    the image. Returns the lists needed by calibration.calibrate_hand_eye."""
    views = {"R_base_ee": [], "t_base_ee": [], "R_cam_board": [], "t_cam_board": []}
    skipped = 0
    for q_target in targets:
        q_hat = move_and_settle(sim, ctrl, q_target)
        if q_hat is None:
            skipped += 1
            continue
        rgb, _ = sim.read_camera()
        corners = calibration.detect_corners(rgb)
        if corners is None:
            skipped += 1
            continue
        R_cam_board, t_cam_board = calibration.board_pose(corners, K, dist)
        t_base_ee, R_base_ee = robot.site_pose(q_hat, EE_SITE)
        views["R_base_ee"].append(R_base_ee)
        views["t_base_ee"].append(t_base_ee)
        views["R_cam_board"].append(R_cam_board)
        views["t_cam_board"].append(np.ravel(t_cam_board))
    n = len(views["R_base_ee"])
    print(f"hand-eye: {n} usable views, {skipped} skipped (not settled or board not detected)")
    if n < 3:
        raise RuntimeError("calibrateHandEye needs at least 3 views")
    return views


# ---- 3. evaluation against ground truth (only here) ----------------------------------------------

def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def evaluate_intrinsics(sim: TaskSim, K: np.ndarray, dist: np.ndarray, rms: float) -> dict:
    """Compare K with sim.ground_truth()["K"] (principal point (319.5, 239.5)) and report the RMS error."""
    K_true = sim.ground_truth()["K"]
    est = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}
    true = {"fx": K_true[0, 0], "fy": K_true[1, 1], "cx": K_true[0, 2], "cy": K_true[1, 2]}
    return {
        "estimate_px": {k: float(v) for k, v in est.items()},
        "true_px": {k: float(v) for k, v in true.items()},
        "error_px": {k: float(est[k] - true[k]) for k in est},
        "dist": [float(v) for v in np.ravel(dist)],
        "rms_reprojection_px": float(rms),
    }


def evaluate_hand_eye(sim: TaskSim, robot: UR3eModel, R_ee_cam: np.ndarray, t_ee_cam: np.ndarray) -> dict:
    """Compare with the true transform: inv(T_base_ee(q_true)) @ T_base_cam, with the camera frame converted
    from MuJoCo to OpenCV by diag(1, -1, -1). Report the rotation error (deg) and translation error (mm).

    The controller-side model's kinematics equal the simulator's (scripts/check_controller_model.py), so its
    FK of the true joint angles gives the true end-effector pose."""
    gt = sim.ground_truth()
    p_ee, R_ee = robot.site_pose(gt["q"], EE_SITE)
    R_true = R_ee.T @ gt["cam_mat"] @ MJ_FROM_CV
    t_true = R_ee.T @ (gt["cam_pos"] - p_ee)
    t_est = np.ravel(t_ee_cam)
    return {
        "rotation_error_deg": rotation_angle_deg(R_true.T @ R_ee_cam),
        "translation_error_mm": float(np.linalg.norm(t_est - t_true) * 1000),
        "t_estimate_mm": [float(v) for v in t_est * 1000],
        "t_true_mm": [float(v) for v in t_true * 1000],
    }


# ---- main ----------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera-noise", action="store_true", help="noisy wrist RGB-D")
    args = p.parse_args()

    sim = TaskSim(camera_noise=CameraNoiseConfig() if args.camera_noise else None)
    robot = UR3eModel()
    rng = np.random.default_rng(SEED)

    # 1. Intrinsics: arm at the observe pose, board moved around.
    sim.reset(SEED)
    corner_sets, image_size = collect_intrinsic_views(sim, intrinsic_board_poses(rng))
    K, dist, rms = calibration.calibrate_intrinsics(corner_sets, image_size)
    intrinsic_report = evaluate_intrinsics(sim, K, dist, rms)

    # 2. Hand-eye: board fixed flat on the table, arm moved by the joint controller.
    sim.reset(SEED)
    sim.place_calib_board(HAND_EYE_BOARD_POS)
    ctrl = ArmController(robot=robot, dt=sim.dt, y0=sim.read_joint_positions(), mode="lqr")
    q_observe = ctrl.xhat[:, 0].copy()
    targets = hand_eye_joint_targets(q_observe, rng, robot, K)
    views = collect_hand_eye_views(sim, robot, ctrl, targets, K, dist)
    R_ee_cam, t_ee_cam = calibration.calibrate_hand_eye(**views)
    hand_eye_report = evaluate_hand_eye(sim, robot, R_ee_cam, t_ee_cam)

    # 3. Save the calibration (not the ground-truth comparison) for the pipeline to load.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "K": np.asarray(K).tolist(),
        "dist": np.ravel(dist).tolist(),
        "ee_site": EE_SITE,
        "R_ee_cam": np.asarray(R_ee_cam).tolist(),
        "t_ee_cam": np.ravel(t_ee_cam).tolist(),
        "camera_frame": "OpenCV (+z forward, +y down)",
        "camera_noise": args.camera_noise,
    }, indent=2))
    print(json.dumps({"intrinsics": intrinsic_report, "hand_eye": hand_eye_report}, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
