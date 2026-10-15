from dataclasses import dataclass

import cv2
import numpy as np


CUBE_SIDE = 0.050
CUBE_HALF_HEIGHT = 0.025
EDGE_TRIM_PCT = 0.5  # percent of top-face points trimmed on each side before the rectangle fit


@dataclass(frozen=True)
class CubeEstimate:
    """Estimated cube pose and geometric quality indicators."""

    pos: np.ndarray
    yaw: float
    n_points: int
    top_size: tuple[float, float]
    ok: bool


def _invalid_estimate() -> CubeEstimate:
    return CubeEstimate(
        pos=np.full(3, np.nan),
        yaw=float("nan"),
        n_points=0,
        top_size=(float("nan"), float("nan")),
        ok=False,
    )


def estimate_cube(
    points_w: np.ndarray,
    min_top_points: int = 200,
) -> CubeEstimate:
    """Estimate cube center and yaw from world-frame points."""
    points_w = np.asarray(points_w, dtype=float)

    if points_w.ndim != 2 or points_w.shape[1] != 3:
        raise ValueError("points_w must have shape (N, 3)")

    points_w = points_w[np.all(np.isfinite(points_w), axis=1)]
    if len(points_w) < min_top_points:
        return _invalid_estimate()

    # Use a high percentile to reduce sensitivity to isolated high outliers.
    z_ref = np.percentile(points_w[:, 2], 98.0)
    top = points_w[points_w[:, 2] >= z_ref - 0.005]

    if len(top) < min_top_points:
        return _invalid_estimate()

    # minAreaRect encloses every point, so a few outliers can inflate it. Depth noise moves points along the
    # viewing ray, which is slanted away from the image centre, so some top-face points land millimetres
    # outside the true edge. Trim the extreme EDGE_TRIM_PCT percent on each side in the frame of a first
    # fit, then fit again on the points that remain.
    xy = top[:, :2]
    _, _, angle0 = cv2.minAreaRect(xy.astype(np.float32))
    c, s = np.cos(np.deg2rad(angle0)), np.sin(np.deg2rad(angle0))
    local = xy @ np.array([[c, -s], [s, c]])
    lo, hi = np.percentile(local, [EDGE_TRIM_PCT, 100.0 - EDGE_TRIM_PCT], axis=0)
    inliers = np.all((local >= lo) & (local <= hi), axis=1)
    if np.count_nonzero(inliers) < min_top_points:
        return _invalid_estimate()

    (cx, cy), (width, height), angle_deg = cv2.minAreaRect(
        xy[inliers].astype(np.float32)
    )

    yaw = (
        (np.deg2rad(angle_deg) + np.pi / 4.0)
        % (np.pi / 2.0)
    ) - np.pi / 4.0

    center_z = float(np.mean(top[:, 2]) - CUBE_HALF_HEIGHT)
    size = (float(width), float(height))

    size_ok = (
        abs(width - CUBE_SIDE) <= 0.005
        and abs(height - CUBE_SIDE) <= 0.005
    )

    return CubeEstimate(
        pos=np.array([cx, cy, center_z], dtype=float),
        yaw=float(yaw),
        n_points=len(top),
        top_size=size,
        ok=bool(size_ok and len(top) >= min_top_points),
    )