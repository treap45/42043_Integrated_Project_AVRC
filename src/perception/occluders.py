from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from src.perception.projection import backproject, camera_to_world
from src.perception.segmentation import CUBE_HSV, ZONE_HSV, color_mask


TABLE_Z = 0.75
HEIGHT_MIN = TABLE_Z + 0.02
HEIGHT_MAX = TABLE_Z + 0.25  # the gripper fingertips are at about TABLE_Z + 0.27 in the observe view

CUBE_DILATE_SIZE = 7       # 3 px expansion around the cube mask
IMAGE_MARGIN_PX = 3
MIN_COMPONENT_PIXELS = 30
MIN_TOP_POINTS = 15
TOP_BAND_M = 0.005
MAX_OCCLUDER_SIDE_M = 0.085  # 0.08 m physical limit plus small measurement tolerance
FUSE_DIST_M = 0.05  # centre distance within which two views' estimates are taken as the same occluder
# Occluders are at least 0.04 m wide. Narrower strips inside the image are the cube's darker side faces,
# which fail the red HSV test just outside the dilated cube mask.
MIN_OCCLUDER_SIDE_M = 0.02
# Points this close to the cube centre (horizontally) belong to the cube: its half-diagonal is 0.035 m, and an
# occluder edge is never nearer than about 0.045 m. Catches side faces too dark for the HSV test.
CUBE_EXCLUDE_RADIUS_M = 0.04


@dataclass(frozen=True)
class OccluderEstimate:
    """Top-face estimate; half_xy is ordered along yaw and its perpendicular axis."""

    xy: np.ndarray
    yaw: float
    half_xy: np.ndarray
    top_z: float
    n_points: int
    truncated: bool
    merged: bool = False  # larger than one occluder: likely two touching ones, kept as one box
    # For a truncated estimate: unit world-frame xy directions (k, 2) in which the occluder may extend out
    # of view, one per image border it touches.
    cut_dirs: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))


def _touches_image_margin(mask: np.ndarray, margin: int = IMAGE_MARGIN_PX) -> bool:
    """Return True if foreground pixels touch the image-edge margin."""
    height, width = mask.shape
    edge = np.zeros((height, width), dtype=bool)

    edge[:margin, :] = True
    edge[-margin:, :] = True
    edge[:, :margin] = True
    edge[:, -margin:] = True

    return bool(np.any(mask & edge))


def _cut_directions(mask: np.ndarray, cam_rot: np.ndarray, margin: int = IMAGE_MARGIN_PX) -> np.ndarray:
    """World xy directions out of the image for each border the mask touches.

    Image right is the OpenCV camera +x axis, image down the OpenCV +y axis; cam_rot is in the MuJoCo camera
    convention (OpenCV y = -MuJoCo y), as in camera_to_world().
    """
    right, down = cam_rot[:, 0], -cam_rot[:, 1]
    borders = (
        (mask[:, -margin:].any(), right),
        (mask[:, :margin].any(), -right),
        (mask[-margin:, :].any(), down),
        (mask[:margin, :].any(), -down),
    )
    dirs = [d[:2] / np.linalg.norm(d[:2]) for touches, d in borders if touches]
    return np.array(dirs) if dirs else np.empty((0, 2))


def _wrap_axis_yaw(angle: float) -> float:
    """Wrap an undirected rectangle axis to [-pi/2, pi/2)."""
    return float((angle + np.pi / 2.0) % np.pi - np.pi / 2.0)


def _fit_top_face(
    points_w: np.ndarray,
    component_mask: np.ndarray,
) -> OccluderEstimate | None:
    """Fit one connected image component using only its top-face points."""
    if len(points_w) < MIN_TOP_POINTS:
        return None

    z_ref = float(np.percentile(points_w[:, 2], 98.0))
    top = points_w[points_w[:, 2] >= z_ref - TOP_BAND_M]

    if len(top) < MIN_TOP_POINTS:
        return None

    (cx, cy), (width, height), angle_deg = cv2.minAreaRect(
        top[:, :2].astype(np.float32)
    )

    if width <= 0.0 or height <= 0.0:
        return None

    # Align yaw with the longer rectangle axis so half_xy[0] is along yaw.
    if width >= height:
        long_side, short_side = float(width), float(height)
        yaw = np.deg2rad(angle_deg)
    else:
        long_side, short_side = float(height), float(width)
        yaw = np.deg2rad(angle_deg + 90.0)

    yaw = _wrap_axis_yaw(float(yaw))

    # A thin strip cut by the image border is the visible part of a real occluder: keep it.
    truncated = _touches_image_margin(component_mask)
    if short_side < MIN_OCCLUDER_SIDE_M and not truncated:
        return None

    # A component this large is likely two touching occluders. Keep it as one box: dropping it
    # would hide both from the planner.
    merged = long_side > MAX_OCCLUDER_SIDE_M

    return OccluderEstimate(
        xy=np.array([cx, cy], dtype=float),
        yaw=yaw,
        half_xy=np.array([long_side / 2.0, short_side / 2.0], dtype=float),
        top_z=float(np.median(top[:, 2])),
        n_points=int(len(top)),
        truncated=truncated,
        merged=merged,
    )


def estimate_occluders(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
) -> list[OccluderEstimate]:
    """Estimate visible table-top occluders from one RGB-D image.

    cam_rot must use the MuJoCo camera convention expected by camera_to_world().
    """
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
        raise ValueError("depth must have shape (H, W) and match rgb")

    valid_depth = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid_depth):
        return []

    # backproject preserves row-major order for the valid pixels in valid_depth.
    v, u = np.nonzero(valid_depth)
    points_cv = backproject(valid_depth, depth, K)

    if len(points_cv) != len(v):
        raise RuntimeError("Back-projected points no longer match valid pixel indices")

    points_w = camera_to_world(points_cv, cam_pos, cam_rot)
    in_height_range = (
        (points_w[:, 2] > HEIGHT_MIN)
        & (points_w[:, 2] < HEIGHT_MAX)
        & np.all(np.isfinite(points_w), axis=1)
    )

    height_mask = np.zeros(depth.shape, dtype=bool)
    height_mask[v[in_height_range], u[in_height_range]] = True

    cube_mask = color_mask(rgb, **CUBE_HSV)
    zone_mask = color_mask(rgb, **ZONE_HSV)

    # Grow the cube mask so its less-saturated border is not classified as an occluder.
    cube_kernel = np.ones((CUBE_DILATE_SIZE, CUBE_DILATE_SIZE), dtype=np.uint8)
    cube_mask = cv2.dilate(
        cube_mask.astype(np.uint8),
        cube_kernel,
        iterations=1,
    ).astype(bool)

    candidate_mask = height_mask & ~cube_mask & ~zone_mask

    cube_points = (cube_mask & valid_depth)[v, u] & in_height_range
    if np.count_nonzero(cube_points) >= MIN_TOP_POINTS:
        cube_xy = np.median(points_w[cube_points, :2], axis=0)
        near_cube = np.hypot(*(points_w[:, :2] - cube_xy).T) < CUBE_EXCLUDE_RADIUS_M
        candidate_mask[v[near_cube], u[near_cube]] = False

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8),
        connectivity=8,
    )

    estimates = []

    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_PIXELS:
            continue

        component_mask = labels == label

        # The mask was formed from valid-depth pixels, so these points correspond
        # to this component and are already within the requested height range.
        component_cv = backproject(component_mask, depth, K)
        if len(component_cv) < MIN_TOP_POINTS:
            continue

        component_w = camera_to_world(component_cv, cam_pos, cam_rot)
        estimate = _fit_top_face(component_w, component_mask)

        if estimate is not None:
            if estimate.truncated:
                estimate = replace(estimate, cut_dirs=_cut_directions(component_mask, cam_rot))
            estimates.append(estimate)

    estimates.sort(key=lambda item: (item.xy[0], item.xy[1]))
    return estimates


def fuse_occluders(per_view: list[list[OccluderEstimate]]) -> list[OccluderEstimate]:
    """Combine the estimates of several views of a static scene.

    An occluder seen whole in some view is represented by that estimate (the one with most top-face points),
    and cut-off estimates of it from other views are dropped. Cut-off estimates that no whole estimate explains
    are all kept, since each bounds the occluder from a different side.
    """
    candidates = sorted((e for view in per_view for e in view), key=lambda e: (e.truncated, -e.n_points))
    fused = []
    for e in candidates:
        if any(not k.truncated and np.hypot(*(k.xy - e.xy)) < FUSE_DIST_M for k in fused):
            continue
        fused.append(e)
    fused.sort(key=lambda item: (item.xy[0], item.xy[1]))
    return fused

