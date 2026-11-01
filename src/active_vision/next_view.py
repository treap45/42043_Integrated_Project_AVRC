"""Next view for the occluder map: look where the grasp decision still lacks information.

The grasp-yaw check (src/planning/clearance.py) treats unseen table as occupied, so a yaw whose gripper columns
are not wholly in some view, or that may run into the hidden part of a cut-off occluder, cannot be cleared. This
module picks the next camera pose by how much of that missing information it would add:

  required columns: for each of the four grasp yaws, the footprint columns of the gripper slabs that reach
      below the tallest possible occluder; and for each cut-off occluder, the column of its conservative box;
  gain of a view: the number of required columns that no earlier view shows whole and this one would.

Candidate poses keep the observe pose's orientation and height offset, turned about the base axis and moved
radially so that the camera's optical axis meets the table at a target point (as the ZONE_VIEW pose does), for
targets between the cube and the centroid of the unseen columns and a few extra heights. The camera pose of each
candidate comes from its IK solution through the camera model, so the gain is evaluated with the pose the
pipeline will actually use.
"""

import numpy as np

from src.perception.occluders import TABLE_Z, OccluderEstimate
from src.planning.clearance import OCCLUDER_MAX_HEIGHT, conservative_box, in_view, part_columns
from src.planning.grasp import GRASP_DEPTH, yaw_candidates
from src.planning.ik import solve_ik

TARGET_FRACTIONS = (0.0, 0.5, 1.0)  # target = cube + fraction * (unseen centroid - cube)
EXTRA_HEIGHTS = (0.0, 0.06, 0.12)  # m above the observe pose's height: higher sees more, with noisier depth
# m, targets nearer the base axis are pushed out to this radius: with the observe orientation the arm cannot
# reach a look-at point much closer (the same limit as ZONE_VIEW_MIN_RADIUS in src/task/pick_place.py)
MIN_TARGET_RADIUS = 0.30


def required_columns(cube_pos: np.ndarray, cube_yaw: float, occluders: list[OccluderEstimate],
                     margin: float) -> list[np.ndarray]:
    """Columns (8 corner points each, world frame) the grasp decision needs to have seen."""
    cube_pos = np.asarray(cube_pos, dtype=float)
    tcp_z = cube_pos[2] + GRASP_DEPTH
    columns = []
    for psi in yaw_candidates(cube_yaw):
        columns.extend(part_columns(cube_pos[:2], psi, tcp_z, margin))
    for occ in occluders:
        if not occ.truncated:
            continue
        center, sides, yaw = conservative_box(occ)
        c, s = np.cos(yaw), np.sin(yaw)
        corners = [center + np.array([c * dx * sides[0] - s * dy * sides[1], s * dx * sides[0] + c * dy * sides[1]]) / 2
                   for dx in (-1, 1) for dy in (-1, 1)]
        top = min(occ.top_z, TABLE_Z + OCCLUDER_MAX_HEIGHT)
        columns.append(np.array([[*xy, z] for z in (TABLE_Z, top) for xy in corners]))
    return columns


def unseen(columns: list[np.ndarray], views: list[tuple]) -> list[np.ndarray]:
    return [col for col in columns if not any(in_view(col, v) for v in views)]


def _look_at(cam_pos: np.ndarray, cam_rot: np.ndarray) -> np.ndarray:
    """Table point on the optical axis (MuJoCo camera convention: the camera looks along -z)."""
    axis = -cam_rot[:, 2]
    t = (TABLE_Z - cam_pos[2]) / axis[2]
    return (cam_pos + t * axis)[:2]


def _candidate_pose(p_obs, R_obs, look0, target, dz):
    """TCP pose: the observe pose turned about the base axis and moved radially to look at target."""
    phi = np.arctan2(target[1], target[0]) - np.arctan2(look0[1], look0[0])
    e_r = np.array([*look0 / np.linalg.norm(look0), 0.0])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]])
    shift = (np.linalg.norm(target) - np.linalg.norm(look0)) * e_r + np.array([0.0, 0.0, dz])
    return Rz @ (p_obs + shift), Rz @ R_obs


def select_next_view(robot, camera, q_observe: np.ndarray, q_now: np.ndarray, cube_pos: np.ndarray,
                     cube_yaw: float, occluders: list[OccluderEstimate], views: list[tuple], margin: float,
                     image_size: tuple[int, int]) -> dict | None:
    """Best next view, or None if no reachable candidate shows any required column not seen yet.

    Returns {"q", "view", "gain", "unseen", "target", "dz"}; view is (K, cam_pos, cam_rot, image_size) of the
    planned pose, for the record (the pipeline uses the pose it reaches).
    """
    cube_pos = np.asarray(cube_pos, dtype=float)
    missing = unseen(required_columns(cube_pos, cube_yaw, occluders, margin), views)
    if not missing:
        return None
    centroid = np.mean([col[:, :2].mean(axis=0) for col in missing], axis=0)

    p_obs, R_obs = robot.site_pose(q_observe)
    look0 = _look_at(*camera.pose(robot, q_observe))
    best = None
    for frac in TARGET_FRACTIONS:
        target = cube_pos[:2] + frac * (centroid - cube_pos[:2])
        target = target * max(np.linalg.norm(target), MIN_TARGET_RADIUS) / np.linalg.norm(target)
        for dz in EXTRA_HEIGHTS:
            p, R = _candidate_pose(p_obs, R_obs, look0, target, dz)
            q, ok = solve_ik(robot, p, R, q_now)
            if not ok:
                continue
            view = (camera.K, *camera.pose(robot, q), image_size)
            gain = sum(in_view(col, view) for col in missing)
            motion = float(np.sum(np.abs(q - q_now)))
            if gain > 0 and (best is None or (gain, -motion) > (best["gain"], -best["motion"])):
                best = {"q": q, "view": view, "gain": gain, "unseen": len(missing), "target": target,
                        "dz": dz, "motion": motion}
    return best
