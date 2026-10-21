"""Joint-space Kalman filters: one 3-state filter per arm joint, x = [q, qd, d].

Each filter uses the per-joint model in src/model/joint_ss.py. Its input is the feedback torque that
actually reached the joint, u = clip(tau) - g_hat(q_hat), i.e. the applied torque after actuator
saturation minus the gravity compensation computed by the controller. Its measurement is the encoder
angle.
"""

import numpy as np
from scipy.linalg import expm

from src.model.joint_ss import JointSSModel
from src.model.ur3e import N_JOINTS

# Encoder specification as the estimator assumes it (datasheet values). It is set here independently
# of the simulator; the simulated encoder in src/sim/sensors.py is configured to the same values.
ENCODER_NOISE_STD = 5e-4  # rad
ENCODER_RESOLUTION = 2 * np.pi / 2**14  # rad per count


def discrete_process_noise(A: np.ndarray, Qc: np.ndarray, dt: float) -> np.ndarray:
    """Van Loan: Qd = integral_0^dt exp(A s) Qc exp(A^T s) ds for continuous white noise Qc on xdot."""
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = Qc
    M[n:, n:] = A.T
    E = expm(M * dt)
    Phi = E[n:, n:].T
    return Phi @ E[:n, n:]


class JointKalmanFilter:
    """Single-joint KF with state x = [q (rad), qd (rad/s), d (N m)] and measurement y = q (rad)."""

    def __init__(
        self,
        model: JointSSModel,
        y0: float,
        encoder_noise_std: float = ENCODER_NOISE_STD,
        encoder_resolution: float = ENCODER_RESOLUTION,
        qd0_std: float = 0.1,
        d0_std: float = 5.0,
        sigma_qd: float = 0.05,
        sigma_d: float = 30.0,
    ):
        """sigma_qd (rad/s per sqrt(s)) and sigma_d (N m per sqrt(s)) are continuous white-noise intensities
        on qd and d: unmodelled acceleration and the rate at which the lumped disturbance drifts."""
        self.sigma_qd, self.sigma_d = float(sigma_qd), float(sigma_d)
        self.set_model(model)

        # Quantisation error treated as uniform noise with variance resolution^2 / 12, independent of
        # the Gaussian encoder noise.
        self.R = encoder_noise_std**2 + encoder_resolution**2 / 12.0


        # Initialise from the first encoder reading.
        # Do not call update() again with this same y0 sample.
        self.xhat = np.array([y0, 0.0, 0.0], dtype=float)
        self.P = np.diag([
            self.R,
            qd0_std**2,
            d0_std**2,
        ])

        # Last innovation and its predicted variance, for consistency checks (NIS) without ground truth.
        self.innovation = 0.0
        self.S = self.R

    def set_model(self, model: JointSSModel) -> None:
        """Use a new joint model (for example a new inertia J) from the next sample on; the estimate and its
        covariance are kept."""
        self.Ad, self.Bd = model.discretize()
        self.C = model.C
        Qc = np.diag([0.0, self.sigma_qd**2, self.sigma_d**2])
        self.Q = discrete_process_noise(model.A, Qc, model.dt)

    def predict(self, u: float) -> None:
        """Propagate one sample with u = applied torque (after saturation) minus gravity compensation."""
        self.xhat = self.Ad @ self.xhat + self.Bd[:, 0] * float(u)
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)  # Reduce floating-point asymmetry.

    def update(self, y: float) -> None:
        """Update the state estimate using one encoder position measurement."""
        innovation = float(y) - (self.C @ self.xhat)[0]
        S = float((self.C @ self.P @ self.C.T)[0, 0] + self.R)
        K = (self.P @ self.C.T)[:, 0] / S

        self.xhat = self.xhat + K * innovation
        self.innovation, self.S = innovation, S

        # Use the Joseph form for improved numerical stability.
        I = np.eye(3)
        KC = np.outer(K, self.C[0])
        self.P = (
            (I - KC) @ self.P @ (I - KC).T
            + np.outer(K, K) * self.R
        )
        self.P = 0.5 * (self.P + self.P.T)

    @property
    def nis(self) -> float:
        """Normalised innovation squared of the last update; averages to 1 for a consistent filter."""
        return self.innovation**2 / self.S


class ArmObserver:
    """Wrapper around six independent single-joint Kalman filters."""

    def __init__(
        self,
        models: list[JointSSModel],
        y0: np.ndarray,
        sigma_qd: float | np.ndarray = 0.05,
        sigma_d: float | np.ndarray = 30.0,
        **kf_options,
    ):
        """sigma_qd and sigma_d may be scalars or one value per joint; other options apply to all joints."""
        y0 = np.asarray(y0, dtype=float)
        if y0.shape != (N_JOINTS,):
            raise ValueError(f"y0 must have shape ({N_JOINTS},), got {y0.shape}")
        if len(models) != N_JOINTS:
            raise ValueError(f"Expected one model for each of the {N_JOINTS} joints")
        sigma_qd = np.broadcast_to(np.asarray(sigma_qd, dtype=float), (N_JOINTS,))
        sigma_d = np.broadcast_to(np.asarray(sigma_d, dtype=float), (N_JOINTS,))

        self.filters = [
            JointKalmanFilter(model, y0[i], sigma_qd=sigma_qd[i], sigma_d=sigma_d[i], **kf_options)
            for i, model in enumerate(models)
        ]

    def set_models(self, models: list[JointSSModel]) -> None:
        """Replace the six joint models, keeping the estimates and covariances."""
        if len(models) != N_JOINTS:
            raise ValueError(f"Expected one model for each of the {N_JOINTS} joints")
        for filt, model in zip(self.filters, models):
            filt.set_model(model)

    def predict(self, u: np.ndarray) -> None:
        """Predict using u[i] = applied torque[i] (after saturation) - gravity compensation[i]."""
        u = np.asarray(u, dtype=float)
        if u.shape != (N_JOINTS,):
            raise ValueError(f"u must have shape ({N_JOINTS},), got {u.shape}")

        for i, filt in enumerate(self.filters):
            filt.predict(u[i])

    def update(self, y: np.ndarray) -> None:
        """Update all six filters from the encoder position readings."""
        y = np.asarray(y, dtype=float)
        if y.shape != (N_JOINTS,):
            raise ValueError(f"y must have shape ({N_JOINTS},), got {y.shape}")

        for i, filt in enumerate(self.filters):
            filt.update(y[i])

    @property
    def xhat(self) -> np.ndarray:
        """Return an array of shape (6, 3), with columns q, qd, and d."""
        return np.stack([filt.xhat for filt in self.filters])

    @property
    def covariance(self) -> np.ndarray:
        """Return an array of shape (6, 3, 3) containing the state covariances."""
        return np.stack([filt.P for filt in self.filters])

    @property
    def nis(self) -> np.ndarray:
        """Normalised innovation squared of the last update, one per joint."""
        return np.array([filt.nis for filt in self.filters])
