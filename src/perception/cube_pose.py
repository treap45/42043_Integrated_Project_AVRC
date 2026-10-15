from dataclasses import dataclass

import cv2
import numpy as np


CUBE_SIDE = 0.050
CUBE_HALF_HEIGHT = 0.025


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

    (cx, cy), (width, height), angle_deg = cv2.minAreaRect(
        top[:, :2].astype(np.float32)
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