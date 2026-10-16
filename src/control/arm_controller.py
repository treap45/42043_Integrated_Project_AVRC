"""Joint-space arm controller: joint Kalman filters + gravity compensation + PD or LQR feedback.

Each control period:

    tau = clip( g_hat(q_hat) + u_fb(q_hat - q_ref, qd_hat - qd_ref) - d_hat ,  +/- TORQUE_LIMIT )

g_hat comes from the payload-free controller-side model, u_fb is LQR (joint_lqr.py, default) or PD
(pid.py) feedback on the estimated state, and -d_hat cancels the estimated lumped disturbance (joint model:
J * qdd = u + d). The observer is then propagated with the torque that was actually applied,
u = clip(tau) - g_hat.
"""

import numpy as np

from src.control.joint_lqr import ArmLQR
from src.control.pid import ArmPD
from src.estimation.joint_kf import ArmObserver
from src.model.joint_ss import build_joint_models
from src.model.ur3e import N_JOINTS, UR3eModel

# UR3e joint torque limits (N m) as specified for the robot, set independently of the simulator.
TORQUE_LIMIT = np.array([56.0, 56.0, 28.0, 12.0, 12.0, 12.0])


class ArmController:
    """Gravity-compensated six-joint PD or LQR controller with an arm observer."""

    def __init__(
        self,
        robot: UR3eModel,
        dt: float,
        y0: np.ndarray,
        mode: str = "lqr",
        sigma_d=(30.0, 30.0, 20.0, 10.0, 8.0, 8.0),
        pd_bandwidth_hz: float = 5.0,
        q_max=0.01,
        qd_max=1.0,
        u_max=(20.0, 20.0, 10.0, 4.0, 4.0, 4.0),
        use_disturbance_ff: bool = True,
    ):
        """mode "lqr" (default): Bryson-weighted LQR with q_max, qd_max, u_max (scalars or one per joint).
        mode "pd": critically damped PD at pd_bandwidth_hz per joint, the baseline for comparison."""
        y0 = np.asarray(y0, dtype=float)
        if y0.shape != (N_JOINTS,):
            raise ValueError(f"y0 must have shape ({N_JOINTS},), got {y0.shape}")

        if mode not in ("pd", "lqr"):
            raise ValueError("mode must be 'pd' or 'lqr'")

        self.robot = robot
        self.dt = float(dt)
        self.models = build_joint_models(robot, y0, dt)
        self.observer = ArmObserver(self.models, y0, sigma_d=sigma_d)

        self.mode = mode
        self.use_disturbance_ff = bool(use_disturbance_ff)
        self._first = True

        if mode == "pd":
            self.gains = ArmPD(self.models, pd_bandwidth_hz)
        else:
            self.gains = ArmLQR(
                self.models,
                q_max=q_max,
                qd_max=qd_max,
                u_max=u_max,
            )

    @property
    def xhat(self) -> np.ndarray:
        """Return the six estimated [q, qd, d] states."""
        return self.observer.xhat

    def step(
        self,
        y: np.ndarray,
        q_ref: np.ndarray,
        qd_ref: np.ndarray,
    ) -> np.ndarray:
        """Process one encoder sample and return the torque for the next interval."""
        y = np.asarray(y, dtype=float)
        q_ref = np.asarray(q_ref, dtype=float)
        qd_ref = np.asarray(qd_ref, dtype=float)

        if any(v.shape != (N_JOINTS,) for v in (y, q_ref, qd_ref)):
            raise ValueError(f"y, q_ref, and qd_ref must all have shape ({N_JOINTS},)")

        # y0 already initialised the observer, so only later samples are updates.
        if self._first:
            self._first = False
        else:
            self.observer.update(y)

        x = self.observer.xhat
        qhat = x[:, 0]
        qdhat = x[:, 1]
        dhat = x[:, 2]

        # Reuse this gravity estimate in both the torque and observer input.
        gravity = self.robot.gravity_torque(qhat)

        e_q = qhat - q_ref
        e_qd = qdhat - qd_ref

        u_feedback = self.gains.feedback(e_q, e_qd)

        u_disturbance_ff = -dhat if self.use_disturbance_ff else 0.0
        torque_command = gravity + u_feedback + u_disturbance_ff
        torque_applied = np.clip(
            torque_command,
            -TORQUE_LIMIT,
            TORQUE_LIMIT,
        )

        # Predict with the clipped torque after subtracting gravity compensation.
        self.observer.predict(torque_applied - gravity)

        return torque_applied.copy()
