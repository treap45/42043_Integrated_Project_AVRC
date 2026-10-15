import numpy as np


# Nominal intrinsics from the MuJoCo camera fovy and resolution.
# Replace this matrix with the calibrated intrinsics when calibration is complete.
_HEIGHT = 480
_WIDTH = 640
_FOVY_DEG = 42.0
_FOCAL_LENGTH = 0.5 * _HEIGHT / np.tan(np.deg2rad(_FOVY_DEG) / 2.0)

K_NOMINAL = np.array([
    [_FOCAL_LENGTH, 0.0, _WIDTH / 2.0],
    [0.0, _FOCAL_LENGTH, _HEIGHT / 2.0],
    [0.0, 0.0, 1.0],
])

_MJ_FROM_CV = np.diag([1.0, -1.0, -1.0])


def backproject(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray = K_NOMINAL,
) -> np.ndarray:
    """Back-project masked metric depth pixels into the OpenCV camera frame."""
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth)

    if mask.ndim != 2 or depth.ndim != 2:
        raise ValueError("mask and depth must both have shape (H, W)")
    if mask.shape != depth.shape:
        raise ValueError("mask and depth must have the same shape")

    v, u = np.nonzero(mask)
    d = depth[v, u].astype(float)

    valid = np.isfinite(d) & (d > 0.0)
    u = u[valid].astype(float)
    v = v[valid].astype(float)
    d = d[valid]

    if d.size == 0:
        return np.empty((0, 3), dtype=float)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d

    return np.column_stack((x, y, z))


def camera_to_world(
    p_cv: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
) -> np.ndarray:
    """Transform OpenCV camera-frame points into the world frame."""
    p_cv = np.asarray(p_cv, dtype=float)
    cam_pos = np.asarray(cam_pos, dtype=float)
    cam_rot = np.asarray(cam_rot, dtype=float)

    if p_cv.ndim != 2 or p_cv.shape[1] != 3:
        raise ValueError("p_cv must have shape (N, 3)")
    if cam_pos.shape != (3,) or cam_rot.shape != (3, 3):
        raise ValueError("cam_pos and cam_rot must have shapes (3,) and (3, 3)")

    p_mj = p_cv @ _MJ_FROM_CV
    return p_mj @ cam_rot.T + cam_pos