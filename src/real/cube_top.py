"""estimate_cube and cube_measurement of src/perception/cube_pose.py with the top-face band as a setting.

The simulation keeps a top-face band of 5 mm below the 98th percentile of the cube's heights. With the D435i at about
0.9 m the top-face points spread further than that: a band of 5 mm cuts a 60 mm top down to about 52 mm and the size
check fails (docs/worklog.md, 2026-11-02). With the arm above the cube, the mask also picks up depth readings of the
fingers (up to 50 mm above the cube), which move a percentile away from the top face. The cube rests on the table, so
here the top face is the band around its known height. Apart from that band and SIZE_TOL these are the same
computations as in cube_pose.py; src/real/config.py puts them in place of the simulation's in src/task/pick_place.py.

LabCubeEKF is the cube EKF with the yaw noise measured on the lab camera, which is larger than in the simulation.
"""

import cv2
import numpy as np

from src.estimation import cube_ekf
from src.estimation.cube_ekf import CubeEKF
from src.perception import cube_pose

TOP_BAND = 0.010  # m, either side of the known top-face height
YAW_SIGMA = 0.03  # rad, measurement noise of the cube yaw (simulation 0.01; a static cube on the lab camera: 0.026)
SIZE_TOL = 0.008  # m, of the fitted top against the cube edge (cube_pose.py: 0.005); the 60 mm top measures 53-59 mm


def estimate_cube(points_w: np.ndarray, min_top_points: int = 200) -> cube_pose.CubeEstimate:
    points_w = np.asarray(points_w, dtype=float)
    if points_w.ndim != 2 or points_w.shape[1] != 3:
        raise ValueError("points_w must have shape (N, 3)")
    points_w = points_w[np.all(np.isfinite(points_w), axis=1)]
    if len(points_w) < min_top_points:
        return cube_pose._invalid_estimate()

    top = points_w[np.abs(points_w[:, 2] - cube_ekf.Z_TOP) <= TOP_BAND]
    if len(top) < min_top_points:
        return cube_pose._invalid_estimate()

    xy = top[:, :2]
    _, _, angle0 = cv2.minAreaRect(xy.astype(np.float32))
    c, s = np.cos(np.deg2rad(angle0)), np.sin(np.deg2rad(angle0))
    local = xy @ np.array([[c, -s], [s, c]])
    lo, hi = np.percentile(local, [cube_pose.EDGE_TRIM_PCT, 100.0 - cube_pose.EDGE_TRIM_PCT], axis=0)
    inliers = np.all((local >= lo) & (local <= hi), axis=1)
    if np.count_nonzero(inliers) < min_top_points:
        return cube_pose._invalid_estimate()

    (cx, cy), (width, height), angle_deg = cv2.minAreaRect(xy[inliers].astype(np.float32))
    yaw = ((np.deg2rad(angle_deg) + np.pi / 4.0) % (np.pi / 2.0)) - np.pi / 4.0
    center_z = float(np.mean(top[:, 2]) - cube_pose.CUBE_HALF_HEIGHT)
    size_ok = abs(width - cube_pose.CUBE_SIDE) <= SIZE_TOL and abs(height - cube_pose.CUBE_SIDE) <= SIZE_TOL
    return cube_pose.CubeEstimate(pos=np.array([cx, cy, center_z], dtype=float), yaw=float(yaw), n_points=len(top),
                                  top_size=(float(width), float(height)), ok=bool(size_ok))


def cube_measurement(points_cv, points_w, cube, K_cam):
    """[u, v, Z, yaw] from matched OpenCV and world point clouds, or None."""
    if cube is None or not cube.ok:
        return None
    points_cv = np.asarray(points_cv, dtype=float)
    points_w = np.asarray(points_w, dtype=float)
    if points_cv.ndim != 2 or points_cv.shape[1] != 3 or points_w.shape != points_cv.shape:
        raise ValueError("points_cv and points_w must both have shape (N, 3)")
    valid = np.all(np.isfinite(points_cv), axis=1) & np.all(np.isfinite(points_w), axis=1)
    points_cv, points_w = points_cv[valid], points_w[valid]
    if len(points_cv) == 0:
        return None

    top = np.abs(points_w[:, 2] - cube_ekf.Z_TOP) <= TOP_BAND
    if np.count_nonzero(top) < 20:
        return None
    X, Y, Z = np.mean(points_cv[top], axis=0)
    if Z <= 0.0:
        return None
    return np.array([K_cam[0, 0] * X / Z + K_cam[0, 2], K_cam[1, 1] * Y / Z + K_cam[1, 2], Z, cube.yaw], dtype=float)


class LabCubeEKF(CubeEKF):
    def __init__(self, x0, dt, **kwargs):
        kwargs.setdefault("R", np.diag([0.5**2, 0.5**2, 0.001**2, YAW_SIGMA**2]))
        super().__init__(x0, dt, **kwargs)
