import cv2
import numpy as np

CUBE_HSV = {
    "h_ranges": ((0, 7), (173, 179)),
    "s_min": 120,
    "v_min": 60,
}
ZONE_HSV = {
    "h_ranges": ((141, 165),),
    "s_min": 120,
    "v_min": 60,
}

def color_mask(
    rgb: np.ndarray,
    h_ranges: tuple[tuple[int, int], ...],
    s_min: int,
    v_min: int,
) -> np.ndarray:
    """Return a boolean color mask from an RGB uint8 image."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must use uint8 values")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    hue_mask = np.zeros(rgb.shape[:2], dtype=bool)
    for h_low, h_high in h_ranges:
        hue_mask |= (
            (hsv[:, :, 0] >= h_low)
            & (hsv[:, :, 0] <= h_high)
        )

    return (
        hue_mask
        & (hsv[:, :, 1] > s_min)
        & (hsv[:, :, 2] > v_min)
    )


def largest_component(
    mask: np.ndarray,
    min_pixels: int = 100,
) -> np.ndarray | None:
    """Return the largest cleaned component, or None if it is too small."""
    if mask.ndim != 2:
        raise ValueError("mask must have shape (H, W)")

    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if n_labels <= 1:
        return None

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    label = 1 + int(np.argmax(component_areas))

    if stats[label, cv2.CC_STAT_AREA] < min_pixels:
        return None

    return labels == label