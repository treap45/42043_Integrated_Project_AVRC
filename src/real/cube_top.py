"""estimate_cube and cube_measurement of src/perception/cube_pose.py with the top-face band as a setting.

The simulation keeps a top-face band of 5 mm below the 98th percentile of the cube's heights. With the D435i at about
0.9 m the top-face points spread further than that: a band of 5 mm cuts a 60 mm top down to about 52 mm and the size
check fails (docs/worklog.md, 2026-11-02). With the arm above the cube, the mask also picks up depth readings of the
fingers (up to 50 mm above the cube), which move a percentile away from the top face. The cube rests on the table, so
here the top face is the band around its known height. Apart from that band and SIZE_TOL these are the same
computations as in cube_pose.py; src/real/config.py puts them in place of the simulation's in src/task/pick_place.py.

LabCubeEKF is the cube EKF with the yaw noise measured on the lab camera, which is larger than in the simulation.
"""

import collections

import cv2
import numpy as np

from src.estimation import cube_ekf
from src.estimation.cube_ekf import CubeEKF
from src.perception import cube_pose

TOP_BAND = 0.010  # m, either side of the known top-face height
YAW_SIGMA = 0.03  # rad, measurement noise of the cube yaw (simulation 0.01; a static cube on the lab camera: 0.026)
camera = None     # FixedCameraModel, set by RealPickPlace
counts = collections.Counter()  # why estimates failed, for the run's log
sizes = []        # mm, the fitted tops that failed the size check
COAST_S = 0.5     # s, without an accepted measurement before the EKF stops extrapolating
SIZE_TOL = 0.008  # m, of the fitted top against the cube edge (cube_pose.py: 0.005); the 60 mm top measures 53-59 mm


def estimate_cube(points_w: np.ndarray, min_top_points: int = 200) -> cube_pose.CubeEstimate:
    points_w = np.asarray(points_w, dtype=float)
    if points_w.ndim != 2 or points_w.shape[1] != 3:
        raise ValueError("points_w must have shape (N, 3)")
    points_w = points_w[np.all(np.isfinite(points_w), axis=1)]
    if len(points_w) < min_top_points:
        counts["too few points"] += 1
        return cube_pose._invalid_estimate()

    top = points_w[np.abs(points_w[:, 2] - cube_ekf.Z_TOP) <= TOP_BAND]
    if len(top) < min_top_points:
        counts["too few top-face points"] += 1
        return cube_pose._invalid_estimate()

    xy = top[:, :2]
    _, _, angle0 = cv2.minAreaRect(xy.astype(np.float32))
    c, s = np.cos(np.deg2rad(angle0)), np.sin(np.deg2rad(angle0))
    local = xy @ np.array([[c, -s], [s, c]])
    lo, hi = np.percentile(local, [cube_pose.EDGE_TRIM_PCT, 100.0 - cube_pose.EDGE_TRIM_PCT], axis=0)
    inliers = np.all((local >= lo) & (local <= hi), axis=1)
    if np.count_nonzero(inliers) < min_top_points:
        counts["too few top-face points"] += 1
        return cube_pose._invalid_estimate()

    (cx, cy), (width, height), angle_deg = cv2.minAreaRect(xy[inliers].astype(np.float32))
    yaw = ((np.deg2rad(angle_deg) + np.pi / 4.0) % (np.pi / 2.0)) - np.pi / 4.0
    center_z = float(np.mean(top[:, 2]) - cube_pose.CUBE_HALF_HEIGHT)
    size_ok = abs(width - cube_pose.CUBE_SIDE) <= SIZE_TOL and abs(height - cube_pose.CUBE_SIDE) <= SIZE_TOL
    counts["ok" if size_ok else "size"] += 1
    if not size_ok:
        sizes.append((round(1000.0 * float(width), 1), round(1000.0 * float(height), 1)))
    return cube_pose.CubeEstimate(pos=np.array([cx, cy, center_z], dtype=float), yaw=float(yaw), n_points=len(top),
                                  top_size=(float(width), float(height)), ok=bool(size_ok))


def cube_measurement(points_cv, points_w, cube, K_cam):
    """[u, v, Z, yaw] of the centre of the fitted top face, or None.

    cube_pose.py takes the mean of the top-face points. On the lab camera those lack the near edge and include some of
    the side face, so the mean sits 5 px and 5 mm from the rectangle's centre, whichever the position, and the EKF
    rejects most measurements of a cube at rest (a bias of NIS 140 against a gate of 13). The centre is projected instead,
    through the camera pose set in `camera` by RealPickPlace, at the known height the EKF's model uses.
    """
    if cube is None or not cube.ok:
        return None
    p_cv = camera.R_world_cv.T @ (np.array([cube.pos[0], cube.pos[1], cube_ekf.Z_TOP]) - camera.p_world_cam)
    if p_cv[2] <= 0.0:
        return None
    return np.array([K_cam[0, 0] * p_cv[0] / p_cv[2] + K_cam[0, 2], K_cam[1, 1] * p_cv[1] / p_cv[2] + K_cam[1, 2],
                     p_cv[2], cube.yaw], dtype=float)


class LabCubeEKF(CubeEKF):
    """The cube EKF with the lab's yaw noise, which does not coast without measurements.

    The arm follows this filter's position, so after a run of frames with no valid measurement (the pusher hiding the
    cube, say) the constant-velocity prediction would lead it along a path the cube is not on: 23 cm in one lab run.
    After COAST_S without an accepted measurement the velocity is taken as zero, so the arm waits where it is, and
    velocity_test() fails, so that a cube the camera cannot see is not declared stopped.
    """

    def __init__(self, x0, dt, **kwargs):
        kwargs.setdefault("R", np.diag([0.5**2, 0.5**2, 0.001**2, YAW_SIGMA**2]))
        super().__init__(x0, dt, **kwargs)
        self._blind_steps = 0

    @property
    def blind(self) -> bool:
        return self._blind_steps * self.dt > COAST_S

    def predict(self):
        self._blind_steps += 1
        if self.blind:
            self.x[3:5] = 0.0
        return super().predict()

    def update(self, *args, **kwargs):
        status = super().update(*args, **kwargs)
        if status in ("accepted", "reinit"):
            self._blind_steps = 0
        return status

    def velocity_test(self) -> float:
        return float("inf") if self.blind else super().velocity_test()
