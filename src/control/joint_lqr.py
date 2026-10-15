"""Discrete LQR for each joint's controllable [q, qd] subsystem.

The lumped disturbance d in the joint model is not controllable, so it is left out of the LQR design and
cancelled by feed-forward of its estimate instead (see arm_controller.py). Weights follow Bryson's rule:

    Q = diag(1 / q_max^2, 1 / qd_max^2),    R = 1 / u_max^2

where q_max and qd_max are the acceptable position and velocity errors and u_max is the torque available
for feedback.
"""

import numpy as np
from scipy.linalg import solve_discrete_are

from src.model.joint_ss import JointSSModel
from src.model.ur3e import N_JOINTS


def _as_joint_positive(value, name: str) -> np.ndarray:
    """Convert a positive scalar or one-value-per-joint sequence to an array of length N_JOINTS."""
    values = np.asarray(value, dtype=float)

    if values.ndim == 0:
        values = np.full(N_JOINTS, float(values))
    elif values.shape != (N_JOINTS,):
        raise ValueError(f"{name} must be a scalar or have shape ({N_JOINTS},)")

    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{name} must contain positive finite values")

    return values


class JointLQR:
    """Discrete LQR for the controllable [q, qd] subsystem of one joint."""

    def __init__(self, model: JointSSModel, q_max: float, qd_max: float, u_max: float):
        q_max = float(q_max)
        qd_max = float(qd_max)
        u_max = float(u_max)

        if min(q_max, qd_max, u_max) <= 0:
            raise ValueError("All Bryson limits must be positive")

        Ad, Bd = model.discretize()

        self.dt = model.dt
        self.A2 = Ad[:2, :2]
        self.B2 = Bd[:2, :]
        self.Q = np.diag([1.0 / q_max**2, 1.0 / qd_max**2])
        self.R = np.array([[1.0 / u_max**2]])

        self.P = solve_discrete_are(self.A2, self.B2, self.Q, self.R)
        self.K = np.linalg.solve(
            self.R + self.B2.T @ self.P @ self.B2,
            self.B2.T @ self.P @ self.A2,
        )

    def feedback(self, e_q: float, e_qd: float) -> float:
        """Return the scalar feedback torque for position and velocity errors."""
        error = np.array([e_q, e_qd], dtype=float)
        return float(-(self.K @ error)[0])

    def closed_loop_poles(self) -> np.ndarray:
        """Return the eigenvalues of the discrete closed-loop system."""
        return np.linalg.eigvals(self.A2 - self.B2 @ self.K)

    def continuous_pole_metrics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return continuous poles, natural frequencies, and damping ratios."""
        poles = self.closed_loop_poles().astype(complex)
        s = np.log(poles) / self.dt
        omega_n = np.abs(s)
        damping_ratio = -s.real / np.maximum(omega_n, np.finfo(float).tiny)
        return s, omega_n, damping_ratio


class ArmLQR:
    """Six independent joint LQR controllers."""

    def __init__(self, models: list[JointSSModel], q_max, qd_max, u_max):
        """q_max, qd_max and u_max may be scalars or one value per joint."""
        if len(models) != N_JOINTS:
            raise ValueError(f"Expected one model for each of the {N_JOINTS} joints")

        q_max = _as_joint_positive(q_max, "q_max")
        qd_max = _as_joint_positive(qd_max, "qd_max")
        u_max = _as_joint_positive(u_max, "u_max")

        self.controllers = [
            JointLQR(models[i], q_max[i], qd_max[i], u_max[i])
            for i in range(N_JOINTS)
        ]

    def feedback(self, e_q: np.ndarray, e_qd: np.ndarray) -> np.ndarray:
        """Return six feedback torques from six position and velocity errors."""
        e_q = np.asarray(e_q, dtype=float)
        e_qd = np.asarray(e_qd, dtype=float)

        if e_q.shape != (N_JOINTS,) or e_qd.shape != (N_JOINTS,):
            raise ValueError(f"e_q and e_qd must both have shape ({N_JOINTS},)")

        return np.array([
            controller.feedback(e_q[i], e_qd[i])
            for i, controller in enumerate(self.controllers)
        ])
