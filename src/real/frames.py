"""Homogeneous transforms and the relation between the RTDE base frame and the simulation world.

RTDE reports the TCP as [x, y, z, rx, ry, rz] (metres, rotation vector in radians) in the UR "base" frame. That frame
is the simulation's robot base rotated 180 degrees about z: at the lab's Home pose RTDE read the flange at
(-0.001, -0.224, 0.694) m, and FK of the controller-side model at the same q puts it at (0.000, +0.223, 1.4435 - 0.75).
The pendant TCP was the flange (no RG2 offset set) on 2026-10-23.
"""

import cv2
import numpy as np

# Robot base in the simulation world: the base body of assets/controller_model.xml sits 0.75 m up, on the table.
SIM_BASE_HEIGHT = 0.75
# RTDE base frame expressed in the simulation's base frame: 180 degrees about z.
R_SIMBASE_RTDE = np.diag([-1.0, -1.0, 1.0])


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rtde_pose_to_T(pose) -> np.ndarray:
    """[x, y, z, rx, ry, rz] from getActualTCPPose to a 4x4 transform T_base_tcp."""
    pose = np.asarray(pose, dtype=float).reshape(6)
    R, _ = cv2.Rodrigues(pose[3:])
    return make_T(R, pose[:3])


def rotation_angle_deg(R: np.ndarray) -> float:
    """Angle of a rotation matrix, in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def sim_world_from_rtde() -> np.ndarray:
    """T_world_rtdebase: maps RTDE base coordinates into the simulation world frame."""
    return make_T(R_SIMBASE_RTDE, [0.0, 0.0, SIM_BASE_HEIGHT])
