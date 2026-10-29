"""Calibration board check: the board is parked during trials, renders as specified, and its detected corners
match the true camera model (which fixes the pixel convention of the true intrinsics).

Test script: projects the true board corners with ground truth, which the pipeline must not do.

Run from the repo root with the venv active:
    python scripts/check_calib_board.py
"""

import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim import CameraNoiseConfig, TaskSim  # noqa: E402
from src.sim.env import CALIB_SQUARE, CALIB_SQUARES, CALIB_THICKNESS  # noqa: E402
from src.sim.randomize import TABLE_Z  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results/check_calib_board"
PATTERN = (CALIB_SQUARES[0] - 1, CALIB_SQUARES[1] - 1)
BOARD_Z = TABLE_Z + CALIB_THICKNESS / 2
MJ_FROM_CV = np.diag([1.0, -1.0, -1.0])


def quat_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Quaternion (w, x, y, z) of rotations about the world x, then y, then z axes (degrees)."""
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.deg2rad([rx, ry, rz]), "XYZ")
    return q


def true_corners_px(sim: TaskSim, K: np.ndarray) -> np.ndarray:
    """Project the inner corners (board frame: centred grid on the printed face) with the true camera pose."""
    gt = sim.ground_truth()
    nx, ny = PATTERN
    i, j = np.meshgrid(np.arange(nx) - (nx - 1) / 2, np.arange(ny) - (ny - 1) / 2)
    p_board = np.column_stack([i.ravel() * CALIB_SQUARE, j.ravel() * CALIB_SQUARE,
                               np.full(i.size, CALIB_THICKNESS / 2)])
    p_world = p_board @ gt["board_mat"].T + gt["board_pos"]
    p_cv = (p_world - gt["cam_pos"]) @ gt["cam_mat"] @ MJ_FROM_CV
    uv = p_cv[:, :2] / p_cv[:, 2:]
    return uv * [K[0, 0], K[1, 1]] + K[:2, 2]


def corner_error(sim: TaskSim, K: np.ndarray, name: str, save: bool = False) -> float | None:
    rgb, _ = sim.read_camera()
    if save:
        cv2.imwrite(str(OUT / f"{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    found, corners = cv2.findChessboardCornersSB(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), PATTERN)
    if not found:
        return None
    detected = corners.reshape(-1, 2)
    truth = true_corners_px(sim, K)
    nearest = np.linalg.norm(detected[:, None] - truth[None], axis=2).min(axis=1)
    return float(nearest.mean())


def check_parked() -> bool:
    print("== 1. Board parked during trials ==")
    sim = TaskSim()
    z = []
    for level in ["L0", "L1"]:
        for seed in range(20):
            sim.reset(seed, level)
            z.append(sim.ground_truth()["board_pos"][2])
    ok = max(z) < 0
    print(f"  board z after reset ({len(z)} trials): max {max(z):.2f} m  {'PASS' if ok else 'FAIL'}")
    return ok


def check_corners() -> bool:
    print("\n== 2. Detected corners vs true projection (findChessboardCornersSB) ==")
    sim = TaskSim()
    K = sim.ground_truth()["K"]
    K_centre = K.copy()
    K_centre[:2, 2] = [sim.camera.width / 2, sim.camera.height / 2]
    poses = {
        "flat": ([0.36, 0.0, BOARD_Z], quat_xyz(0, 0, 0)),
        "flat_rotated": ([0.34, 0.02, BOARD_Z], quat_xyz(0, 0, 20)),
        "tilted_x": ([0.36, 0.0, BOARD_Z + 0.03], quat_xyz(25, 0, 0)),
        "tilted_y": ([0.36, 0.0, BOARD_Z + 0.04], quat_xyz(0, -25, 10)),
    }
    ok = True
    for name, (pos, quat) in poses.items():
        sim.reset(0)
        sim.place_calib_board(pos, quat)
        err = corner_error(sim, K, name, save=True)
        err_centre = corner_error(sim, K_centre, name)
        pose_ok = err is not None and err < 0.15
        ok &= pose_ok
        print(f"  {name:13s} found {err is not None}, mean error {err if err is None else round(err, 3)} px with "
              f"c = (319.5, 239.5), {err_centre if err_centre is None else round(err_centre, 3)} px with "
              f"c = (320, 240)  {'PASS' if pose_ok else 'FAIL'}")
    return ok


def check_depth_convention() -> bool:
    print("\n== 3. Depth back-projection with the true K vs mj_ray ==")
    sim = TaskSim()
    sim.reset(0)
    sim.place_calib_board([0.36, 0.0, BOARD_Z], quat_xyz(0, 0, 0))
    gt = sim.ground_truth()
    K, R, c = gt["K"], gt["cam_mat"], gt["cam_pos"]
    _, depth = sim.read_camera()
    errors = []
    for u, v in [(5, 5), (634, 474), (319, 239), (100, 400), (600, 50), (450, 300)]:
        ray = R @ MJ_FROM_CV @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        point = c + ray * depth[v, u]
        direction = ray / np.linalg.norm(ray)
        start = c + 0.05 * direction  # start outside the camera housing
        dist = mujoco.mj_ray(sim.model, sim.data, start, direction, None, 1, -1, np.array([-1], dtype=np.int32))
        errors.append(np.linalg.norm(point - (start + dist * direction)))
    ok = max(errors) < 1e-5
    print(f"  max distance between back-projected point and ray hit: {max(errors) * 1000:.4f} mm  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_noisy() -> bool:
    print("\n== 4. Detection with camera noise on ==")
    sim = TaskSim(camera_noise=CameraNoiseConfig())
    K = sim.ground_truth()["K"]
    errors = []
    for seed in range(5):
        sim.reset(seed)
        sim.place_calib_board([0.36, 0.0, BOARD_Z + 0.03], quat_xyz(20, 0, 0))
        errors.append(corner_error(sim, K, "noisy"))
    ok = all(e is not None and e < 0.3 for e in errors)
    print(f"  5 seeds, mean corner error {[None if e is None else round(e, 3) for e in errors]} px  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {
        "parked": check_parked(),
        "corners": check_corners(),
        "depth": check_depth_convention(),
        "noisy": check_noisy(),
    }
    print("\n== Summary ==")
    for name, ok in results.items():
        print(f"  {name:8s} {'PASS' if ok else 'FAIL'}")
    print(f"  images in {OUT}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
