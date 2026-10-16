"""PD feedback for each joint, the baseline to compare against the LQR (joint_lqr.py).

Each joint is critically damped at the same bandwidth, using its inertia J from the joint model:

    u = -Kp * e_q - Kd * e_qd,    Kp = J * omega^2,    Kd = 2 * J * omega,    omega = 2 * pi * bandwidth_hz

so the closed-loop poles of J * qdd = u sit at s = -omega (double). There is no integral term: the joint
observer's disturbance estimate d_hat is fed forward in arm_controller.py and removes the steady-state error
an integrator would otherwise be needed for.
"""

import numpy as np

from src.model.joint_ss import JointSSModel
from src.model.ur3e import N_JOINTS


class ArmPD:
    """Six independent critically damped PD controllers, with the same interface as ArmLQR."""

    def __init__(self, models: list[JointSSModel], bandwidth_hz: float):
        if len(models) != N_JOINTS:
            raise ValueError(f"Expected one model for each of the {N_JOINTS} joints")
        if bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive")

        inertia = np.array([model.inertia for model in models])
        omega = 2.0 * np.pi * float(bandwidth_hz)
        self.Kp = inertia * omega**2
        self.Kd = 2.0 * inertia * omega

    def feedback(self, e_q: np.ndarray, e_qd: np.ndarray) -> np.ndarray:
        """Return six feedback torques from six position and velocity errors."""
        e_q = np.asarray(e_q, dtype=float)
        e_qd = np.asarray(e_qd, dtype=float)

        if e_q.shape != (N_JOINTS,) or e_qd.shape != (N_JOINTS,):
            raise ValueError(f"e_q and e_qd must both have shape ({N_JOINTS},)")

        return -self.Kp * e_q - self.Kd * e_qd
