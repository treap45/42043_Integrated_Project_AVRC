import numpy as np

TABLE_Z = 0.75
CUBE_HALF_HEIGHT = 0.025
HOVER = 0.10
GRASP_DEPTH = -0.005
RELEASE_GAP = 0.003

def top_down_rotation(psi: float) -> np.ndarray:
    """Return a world-frame rotation matrix for a top-down grasp."""
    x_axis = np.array([np.cos(psi), np.sin(psi), 0.0])
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def yaw_candidates(psi: float) -> list[float]:
    """Return four grasp orientations equivalent for a square cube."""
    return [psi + k * np.pi / 2.0 for k in range(4)]


def grasp_poses(cube_pos: np.ndarray, psi: float) -> dict:
    """Return pregrasp, grasp, and lift TCP poses."""
    cube_pos = np.asarray(cube_pos, dtype=float)
    rotation = top_down_rotation(psi)

    grasp_pos = cube_pos + np.array([0.0, 0.0, GRASP_DEPTH])
    pregrasp_pos = grasp_pos + np.array([0.0, 0.0, HOVER])
    lift_pos = grasp_pos + np.array([0.0, 0.0, HOVER])

    return {
        "pregrasp": (pregrasp_pos, rotation),
        "grasp": (grasp_pos, rotation),
        "lift": (lift_pos, rotation),
    }


def place_poses(
    zone_xy: np.ndarray,
    psi: float,
    tcp_above_cube: float = GRASP_DEPTH,
) -> dict:
    """Return above-zone, place, and retreat TCP poses."""
    zone_xy = np.asarray(zone_xy, dtype=float)
    rotation = top_down_rotation(psi)

    placed_cube_center_z = TABLE_Z + CUBE_HALF_HEIGHT + RELEASE_GAP
    place_pos = np.array([
        zone_xy[0],
        zone_xy[1],
        placed_cube_center_z + tcp_above_cube,
    ])
    above_pos = place_pos + np.array([0.0, 0.0, HOVER])

    return {
        "above": (above_pos, rotation),
        "place": (place_pos, rotation),
        "retreat": (above_pos.copy(), rotation),
    }