from dataclasses import dataclass

import cv2
import numpy as np


ZONE_SIDE = 0.100
IMAGE_MARGIN_PX = 3


@dataclass(frozen=True)
class ZoneEstimate:
    """Estimated placement-zone center and completeness indicator."""

    xy: np.ndarray
    size: tuple[float, float]
    complete: bool


def _invalid_estimate() -> ZoneEstimate:
    return ZoneEstimate(
        xy=np.full(2, np.nan),
        size=(float("nan"), float("nan")),
        complete=False,
    )


def _touches_image_margin(mask: np.ndarray, margin: int) -> bool:
    """Return True when foreground pixels enter the image-edge margin."""
    edge = np.zeros_like(mask, dtype=bool)
    edge[:margin, :] = True
    edge[-margin:, :] = True
    edge[:, :margin] = True
    edge[:, -margin:] = True
    return bool(np.any(mask & edge))


def estimate_zone(
    points_w: np.ndarray,
    mask: np.ndarray,
    min_points: int = 100,
) -> ZoneEstimate:
    """Estimate the placement-zone center and whether it was fully observed."""
    points_w = np.asarray(points_w, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if points_w.ndim != 2 or points_w.shape[1] != 3:
        raise ValueError("points_w must have shape (N, 3)")
    if mask.ndim != 2:
        raise ValueError("mask must have shape (H, W)")

    points_w = points_w[np.all(np.isfinite(points_w), axis=1)]
    if len(points_w) < min_points:
        return _invalid_estimate()

    (cx, cy), (width, height), _ = cv2.minAreaRect(
        points_w[:, :2].astype(np.float32)
    )

    size_ok = (
        abs(width - ZONE_SIDE) <= 0.008
        and abs(height - ZONE_SIDE) <= 0.008
    )
    image_ok = not _touches_image_margin(mask, IMAGE_MARGIN_PX)

    return ZoneEstimate(
        xy=np.array([cx, cy], dtype=float),
        size=(float(width), float(height)),
        complete=bool(size_ok and image_ok),
    )