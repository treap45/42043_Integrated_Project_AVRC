"""Occluder clearance of a top-down grasp, from the estimated occluder boxes.

The descent is vertical, so each gripper part sweeps a vertical prism over its footprint, from the hover pose
down to the grasp pose. A part hits an occluder if its footprint overlaps the occluder's (grown by the margin)
and its lowest point at the grasp pose is below the occluder top plus the margin.

Part boxes are design constants of the gripper and camera mount in the TCP frame (x along the finger opening,
z towards the cube), measured once from the model with the gripper open, in the same way as the TCP offset.
They are not read from the simulator at run time.
"""

import cv2
import numpy as np

from src.perception.occluders import TABLE_Z, OccluderEstimate
from src.planning.grasp import GRASP_DEPTH

# name: slabs of (x range, y range, z range) in the TCP frame (m); z < 0 is above the TCP. Each part is split
# into height slabs from its mesh vertices (rounded out by 2 mm): the lowest 4 cm of a finger is the narrow
# fingertip, and the lower half of the gripper base is a narrow neck.
GRIPPER_PARTS = {
    "left_finger": [((-0.076, -0.005), (-0.021, 0.021), (-0.105, -0.040)),
                    ((-0.064, -0.041), (-0.015, 0.015), (-0.040, 0.006))],
    "right_finger": [((0.005, 0.076), (-0.021, 0.021), (-0.105, -0.040)),
                     ((0.041, 0.064), (-0.015, 0.015), (-0.040, 0.006))],
    "gripper_base": [((-0.043, 0.043), (-0.040, 0.040), (-0.158, -0.100)),
                     ((-0.044, 0.044), (-0.014, 0.014), (-0.100, -0.053))],
    "camera": [((-0.076, 0.014), (-0.102, -0.073), (-0.145, -0.116))],
}
OCCLUDER_MAX_SIDE = 0.08  # m, largest occluder side (sampling range)
OCCLUDER_SPAN = OCCLUDER_MAX_SIDE * np.sqrt(2.0)  # m, largest extent of an occluder along any direction
OCCLUDER_MAX_HEIGHT = 0.15  # m above the table (sampling range)
_MJ_FROM_CV = np.diag([1.0, -1.0, -1.0])
_MM = 1e3  # rotatedRectangleIntersection works in mm, clear of its absolute tolerances


def _rect(center, sides, yaw) -> tuple:
    """OpenCV rotated rectangle in mm and degrees."""
    return ((float(center[0] * _MM), float(center[1] * _MM)),
            (float(sides[0] * _MM), float(sides[1] * _MM)), float(np.rad2deg(yaw)))


def conservative_box(occ: OccluderEstimate) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre (2,), sides (2,) and yaw of a box that should contain the whole occluder.

    A box that is fully in view (or merged, which already covers both occluders) is used as it is. For one cut
    by the image border, the fitted yaw is unreliable (a cut corner looks like a rotated box), so the bound is
    built in the frame of the cut direction d instead, from the visible box corners and OCCLUDER_SPAN, the
    largest extent of any occluder along any direction: along d, the hidden part lies beyond the border, so the
    box starts at the visible minimum and spans at most OCCLUDER_SPAN; across d, the box contains the visible
    part and spans at most OCCLUDER_SPAN, either way. With two or more cut borders (an image corner) the
    across-d bound is used along both axes.
    """
    if len(occ.cut_dirs) == 0:
        return occ.xy.astype(float).copy(), 2.0 * occ.half_xy.astype(float), occ.yaw
    c, s = np.cos(occ.yaw), np.sin(occ.yaw)
    corners = np.array([occ.xy + np.array([c * dx * occ.half_xy[0] - s * dy * occ.half_xy[1],
                                           s * dx * occ.half_xy[0] + c * dy * occ.half_xy[1]])
                        for dx in (-1, 1) for dy in (-1, 1)])
    d = occ.cut_dirs[0]
    axes = np.array([d, [-d[1], d[0]]])
    proj = corners @ axes.T  # (4, 2): along d, across d
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    lo_b, hi_b = hi - OCCLUDER_SPAN, lo + OCCLUDER_SPAN  # contains the visible part, spans at most the max
    if len(occ.cut_dirs) == 1:
        lo_b[0], hi_b[0] = lo[0], lo[0] + OCCLUDER_SPAN
    center = 0.5 * (lo_b + hi_b) @ axes
    return center, hi_b - lo_b, float(np.arctan2(d[1], d[0]))


def _part_rect(tcp_xy: np.ndarray, psi: float, part: tuple) -> tuple:
    (x0, x1), (y0, y1), _ = part
    x_axis = np.array([np.cos(psi), np.sin(psi)])
    y_axis = np.array([np.sin(psi), -np.cos(psi)])  # TCP y in the world, as in top_down_rotation()
    center = tcp_xy + 0.5 * (x0 + x1) * x_axis + 0.5 * (y0 + y1) * y_axis
    return _rect(center, (x1 - x0, y1 - y0), psi)


def _part_corners(tcp_xy: np.ndarray, psi: float, part: tuple) -> np.ndarray:
    (x0, x1), (y0, y1), _ = part
    x_axis = np.array([np.cos(psi), np.sin(psi)])
    y_axis = np.array([np.sin(psi), -np.cos(psi)])
    return np.array([tcp_xy + x * x_axis + y * y_axis for x in (x0, x1) for y in (y0, y1)])


def in_view(points_w: np.ndarray, view: tuple) -> bool:
    """Whether every world point projects inside the image of view = (K, cam_pos, cam_rot, (width, height)),
    cam_rot in the MuJoCo camera convention."""
    K, cam_pos, cam_rot, (width, height) = view
    p_cv = (points_w - cam_pos) @ (cam_rot @ _MJ_FROM_CV)
    if np.any(p_cv[:, 2] <= 0.0):
        return False
    uv = (p_cv / p_cv[:, 2:3]) @ K.T
    return bool(np.all((uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)))


def grasp_clearance(cube_pos: np.ndarray, psi: float, occluders: list[OccluderEstimate],
                    margin: float, views: list[tuple] | None = None) -> tuple[bool, str | None]:
    """Return (clear, first part that hits an occluder) for the vertical descent to the grasp at yaw psi.

    views: the frames the occluders came from, each (K, cam_pos, cam_rot, (width, height)) as in in_view().
    With them, space no camera saw counts as occupied: a part whose footprint column, from the table up to the
    tallest possible occluder (or the part's own bottom, if lower), is not wholly inside one of the views is not
    clear (reported as "<part> out of view"), since an occluder there would be missed.
    """
    cube_pos = np.asarray(cube_pos, dtype=float)
    tcp_z = cube_pos[2] + GRASP_DEPTH
    boxes = [(conservative_box(o), o.top_z) for o in occluders]
    for name, slabs in GRIPPER_PARTS.items():
        hit = _slabs_hit(cube_pos[:2], psi, tcp_z, slabs, boxes, margin, views)
        if hit:
            return False, name if hit is True else f"{name} {hit}"
    return True, None


def part_columns(tcp_xy: np.ndarray, psi: float, tcp_z: float, margin: float) -> list[np.ndarray]:
    """The footprint columns (8 corner points each) that must be seen for the grasp at yaw psi to be judged:
    one per slab whose bottom at the grasp pose is lower than the tallest possible occluder."""
    columns = []
    for slabs in GRIPPER_PARTS.values():
        for part in slabs:
            column = _column(tcp_xy, psi, tcp_z, part, margin)
            if column is not None:
                columns.append(column)
    return columns


def _column(tcp_xy, psi, tcp_z, part, margin) -> np.ndarray | None:
    bottom = tcp_z - part[2][1]
    if bottom >= TABLE_Z + OCCLUDER_MAX_HEIGHT + margin:
        return None
    corners = _part_corners(tcp_xy, psi, part)
    top = min(bottom, TABLE_Z + OCCLUDER_MAX_HEIGHT + margin)
    return np.vstack([np.column_stack([corners, np.full(4, z)]) for z in (TABLE_Z, top)])


def _slabs_hit(tcp_xy, psi, tcp_z, slabs, boxes, margin, views) -> bool | str:
    """True if a slab overlaps an occluder, "out of view" if one stands over unseen table, else False."""
    for part in slabs:
        bottom = tcp_z - part[2][1]  # world height of the slab's lowest point at the grasp pose
        part_rect = _part_rect(tcp_xy, psi, part)
        column = _column(tcp_xy, psi, tcp_z, part, margin) if views is not None else None
        if column is not None and not any(in_view(column, v) for v in views):
            return "out of view"
        for (center, sides, yaw), top in boxes:
            if top + margin <= bottom:
                continue
            occ_rect = _rect(center, sides + 2.0 * margin, yaw)
            hit, _ = cv2.rotatedRectangleIntersection(part_rect, occ_rect)
            if hit != cv2.INTERSECT_NONE:
                return True
    return False


def lift_height(occluders: list[OccluderEstimate], margin: float, default: float) -> float:
    """World height for the bottom of the held cube during transport: default, or above every occluder."""
    tops = [o.top_z for o in occluders]
    return max([default] + [t + margin for t in tops])
