import numpy as np

from src.model.ur3e import JOINT_NAMES


def orientation_error(R: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Return the world-frame orientation error as a three-vector."""
    error = np.zeros(3)
    for i in range(3):
        error += np.cross(R[:, i], R_des[:, i])
    return 0.5 * error


def wrap_to_seed(q: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
    """Wrap revolute angles to the equivalent solution nearest q_seed."""
    q = np.asarray(q, dtype=float)
    q_seed = np.asarray(q_seed, dtype=float)
    return q_seed + (q - q_seed + np.pi) % (2.0 * np.pi) - np.pi


def _joint_limit_data(robot):
    """Return lower limits, upper limits, and the limited-joint mask."""
    joint_ids = [robot.model.joint(name).id for name in JOINT_NAMES]
    limits = robot.model.jnt_range[joint_ids]
    limited = robot.model.jnt_limited[joint_ids].astype(bool)
    return limits[:, 0], limits[:, 1], limited


def _nearest_within_limits(q, q_seed, lower, upper, limited) -> np.ndarray:
    """Return the solution wrapped towards q_seed, or q itself if wrapping leaves the joint limits."""
    q_wrapped = wrap_to_seed(q, q_seed)
    within_limits = (
        np.all(q_wrapped[limited] >= lower[limited])
        and np.all(q_wrapped[limited] <= upper[limited])
    )
    return q_wrapped if within_limits else q


def solve_ik(
    robot,
    p_des: np.ndarray,
    R_des: np.ndarray,
    q_seed: np.ndarray,
    lam: float = 0.02,
    max_iter: int = 100,
    tol_p: float = 1e-4,
    tol_r: float = 1e-3,
    max_step: float = 0.2,
) -> tuple[np.ndarray, bool]:
    """Solve TCP inverse kinematics with damped least squares."""
    p_des = np.asarray(p_des, dtype=float)
    R_des = np.asarray(R_des, dtype=float)
    q_seed = np.asarray(q_seed, dtype=float)

    if p_des.shape != (3,) or R_des.shape != (3, 3) or q_seed.shape != (6,):
        raise ValueError("Expected p_des (3,), R_des (3, 3), and q_seed (6,)")

    lower, upper, limited = _joint_limit_data(robot)
    margin = 1e-5
    q = q_seed.copy()

    for _ in range(max_iter):
        p, R = robot.site_pose(q, "tcp")
        e_p = p_des - p
        e_r = orientation_error(R, R_des)

        if np.linalg.norm(e_p) < tol_p and np.linalg.norm(e_r) < tol_r:
            return _nearest_within_limits(q, q_seed, lower, upper, limited), True

        error = np.concatenate((e_p, e_r))
        J = robot.site_jacobian(q, "tcp")

        # Solve in task space rather than explicitly computing a matrix inverse.
        task_matrix = J @ J.T + (lam**2) * np.eye(6)
        dq = J.T @ np.linalg.solve(task_matrix, error)

        largest_step = np.max(np.abs(dq))
        if largest_step > max_step:
            dq *= max_step / largest_step

        q += dq
        q[limited] = np.clip(
            q[limited],
            lower[limited] + margin,
            upper[limited] - margin,
        )

    p, R = robot.site_pose(q, "tcp")
    ok = (
        np.linalg.norm(p_des - p) < tol_p
        and np.linalg.norm(orientation_error(R, R_des)) < tol_r
    )
    if ok:
        q = _nearest_within_limits(q, q_seed, lower, upper, limited)
    return q, bool(ok)