"""Position-based visual servoing (PBVS) for the approach to the cube.

The vision loop (about 10 Hz) re-estimates the cube and updates the desired TCP pose; the PBVS law (every
control period) turns the TCP pose error into a joint velocity reference for the joint controller.
"""

import numpy as np

from src.planning.ik import orientation_error

GAIN = 1.5  # 1/s; time constant 0.67 s, well below the joint loop bandwidth (about 10 Hz)
V_MAX = 0.03  # m/s, TCP linear speed limit
W_MAX = 0.5  # rad/s, TCP angular speed limit
DAMPING = 0.02  # damped least-squares lambda, same as solve_ik
QD_MAX = 1.0  # rad/s, joint velocity limit
TRANSIT_V_MAX = 0.15  # m/s, TRANSIT limit: above the 0.10 m/s feed-forward peak, leaving room for feedback
TRANSIT_W_MAX = 0.8  # rad/s, TRANSIT limit: above the 0.5 rad/s feed-forward peak


def pose_error(
    p: np.ndarray,
    R: np.ndarray,
    p_des: np.ndarray,
    R_des: np.ndarray,
) -> np.ndarray:
    """Return the world-frame position and orientation error."""
    p = np.asarray(p, dtype=float)
    p_des = np.asarray(p_des, dtype=float)
    R = np.asarray(R, dtype=float)
    R_des = np.asarray(R_des, dtype=float)

    if p.shape != (3,) or p_des.shape != (3,):
        raise ValueError("p and p_des must have shape (3,)")
    if R.shape != (3, 3) or R_des.shape != (3, 3):
        raise ValueError("R and R_des must have shape (3, 3)")

    return np.concatenate((
        p_des - p,
        orientation_error(R, R_des),
    ))


def _limit_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale a vector down when its Euclidean norm exceeds max_norm."""
    norm = np.linalg.norm(vector)
    if norm > max_norm:
        return vector * (max_norm / norm)
    return vector


def pbvs_twist(
    error: np.ndarray,
    gain: float = GAIN,
    v_max: float = V_MAX,
    w_max: float = W_MAX,
    feedforward: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a six-dimensional pose error into a bounded Cartesian twist."""
    error = np.asarray(error, dtype=float)
    if error.shape != (6,):
        raise ValueError("error must have shape (6,)")

    twist = gain * error

    if feedforward is not None:
        feedforward = np.asarray(feedforward, dtype=float)
        if feedforward.shape != (6,):
            raise ValueError("feedforward must have shape (6,)")
        twist += feedforward

    linear = _limit_norm(twist[:3], v_max)
    angular = _limit_norm(twist[3:], w_max)
    return np.concatenate((linear, angular))


def joint_velocity(
    J: np.ndarray,
    twist: np.ndarray,
    damping: float = DAMPING,
    qd_max: float = QD_MAX,
) -> np.ndarray:
    """Map a Cartesian twist to bounded joint velocities using damped least squares."""
    J = np.asarray(J, dtype=float)
    twist = np.asarray(twist, dtype=float)

    if J.shape != (6, 6) or twist.shape != (6,):
        raise ValueError("J and twist must have shapes (6, 6) and (6,)")
    if damping <= 0 or qd_max <= 0:
        raise ValueError("damping and qd_max must be positive")

    task_matrix = J @ J.T + damping**2 * np.eye(6)
    qd = J.T @ np.linalg.solve(task_matrix, twist)

    largest_rate = np.max(np.abs(qd))
    if largest_rate > qd_max:
        qd *= qd_max / largest_rate

    return qd
