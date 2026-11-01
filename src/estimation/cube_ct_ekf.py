"""Coordinated-turn EKF for the cube: state x = [x, y, psi, vx, vy, omega] (world frame).

The process model is nonlinear: the velocity vector turns at the rate omega, which is itself estimated, so the path is
a circular arc (a straight line at omega = 0). The measurement is the same [u, v, Z, psi] as in CubeEKF. The prediction
therefore linearises f(x) with its Jacobian F at every step, and the update linearises the camera projection.
"""

import numpy as np

from src.estimation.cube_ekf import GATE, CubeEKF, H_cube, _wrap90, h_cube
from src.estimation.joint_kf import discrete_process_noise

OMEGA_TAU = 5.0  # s, omega decays to zero with this time constant, so it stays bounded while unobservable (cube at rest)
SMALL_ANGLE = 1e-4  # rad, below this |omega dt| the arc coefficients use their Taylor series

_A = np.zeros((6, 6))
_A[0, 3] = 1.0
_A[1, 4] = 1.0


def arc_coefficients(omega: float, dt: float) -> tuple[float, float, float, float]:
    """s = sin(w dt) / w, c = (1 - cos(w dt)) / w and their derivatives with respect to w; both limits exist at w = 0."""
    a = omega * dt
    if abs(a) < SMALL_ANGLE:
        return (dt - omega**2 * dt**3 / 6.0, omega * dt**2 / 2.0, -omega * dt**3 / 3.0, dt**2 / 2.0)
    s = np.sin(a) / omega
    c = (1.0 - np.cos(a)) / omega
    return s, c, (dt * np.cos(a) - s) / omega, (dt * np.sin(a) - c) / omega


def f_ct(x: np.ndarray, dt: float, decay: float) -> np.ndarray:
    """One prediction step of the nonlinear process model."""
    px, py, psi, vx, vy, omega = x
    s, c, _, _ = arc_coefficients(omega, dt)
    a = omega * dt
    return np.array([
        px + s * vx - c * vy,
        py + c * vx + s * vy,
        psi,
        np.cos(a) * vx - np.sin(a) * vy,
        np.sin(a) * vx + np.cos(a) * vy,
        decay * omega,
    ])


def F_ct(x: np.ndarray, dt: float, decay: float) -> np.ndarray:
    """Jacobian of f_ct with respect to the state."""
    _, _, _, vx, vy, omega = x
    s, c, ds, dc = arc_coefficients(omega, dt)
    a = omega * dt
    F = np.eye(6)
    F[0, 3], F[0, 4], F[0, 5] = s, -c, ds * vx - dc * vy
    F[1, 3], F[1, 4], F[1, 5] = c, s, dc * vx + ds * vy
    F[3, 3], F[3, 4], F[3, 5] = np.cos(a), -np.sin(a), dt * (-np.sin(a) * vx - np.cos(a) * vy)
    F[4, 3], F[4, 4], F[4, 5] = np.sin(a), np.cos(a), dt * (np.cos(a) * vx - np.sin(a) * vy)
    F[5, 5] = decay
    return F


class CubeCTEKF(CubeEKF):
    """CubeEKF with a turn rate in the state; position, yaw, velocity and the stop test work as in CubeEKF."""

    def __init__(
        self,
        x0: np.ndarray,
        dt: float,
        P0: np.ndarray | None = None,
        sigma_a: float = 0.01,
        sigma_psi: float = 0.01,
        sigma_omega: float = 0.3,
        R: np.ndarray | None = None,
    ):
        x0 = np.asarray(x0, dtype=float)
        super().__init__(x0[:5], dt, sigma_a=sigma_a, sigma_psi=sigma_psi, R=R)
        self.x = np.append(self.x, x0[5] if x0.size == 6 else 0.0)

        p0 = np.zeros((6, 6))
        p0[:5, :5] = self.P0
        p0[5, 5] = 0.5**2  # (rad/s)^2: the turn rate is unknown at the start
        self.P0 = p0 if P0 is None else np.asarray(P0, dtype=float).copy()
        if self.P0.shape != (6, 6):
            raise ValueError("P0 must have shape (6, 6)")
        self.P = self.P0.copy()

        self.decay = float(np.exp(-self.dt / OMEGA_TAU))
        self.Qd = discrete_process_noise(_A, np.diag([0.0, 0.0, sigma_psi**2, sigma_a**2, sigma_a**2, sigma_omega**2]),
                                         self.dt)

    @property
    def omega(self) -> float:
        return float(self.x[5])

    def predict(self) -> None:
        F = F_ct(self.x, self.dt, self.decay)  # evaluated at the state before the step
        self.x = f_ct(self.x, self.dt, self.decay)
        self.P = F @ self.P @ F.T + self.Qd
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z: np.ndarray, K_cam: np.ndarray, cam_pos: np.ndarray, R_cv: np.ndarray) -> str:
        z = np.asarray(z, dtype=float)
        if z.shape != (4,) or not np.all(np.isfinite(z)) or z[2] <= 0.0:
            return "rejected"

        try:
            z_pred = h_cube(self.x[:5], K_cam, cam_pos, R_cv)
            H = np.hstack((H_cube(self.x[:5], K_cam, cam_pos, R_cv), np.zeros((4, 1))))
        except ValueError:
            return "rejected"

        innovation = z - z_pred
        innovation[3] = _wrap90(innovation[3])

        S = H @ self.P @ H.T + self.R
        nis = float(innovation @ np.linalg.solve(S, innovation))

        if not np.isfinite(nis) or nis > GATE:
            self._gate_rejects += 1
            if self._gate_rejects >= 3:
                self._reinitialize(z, K_cam, cam_pos, R_cv)
                return "reinit"
            return "rejected"

        self._gate_rejects = 0
        K = np.linalg.solve(S, H @ self.P).T

        self.x = self.x + K @ innovation
        self.x[2] = _wrap90(self.x[2])

        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._nis = nis
        return "accepted"

    def _reinitialize(self, z, K_cam, cam_pos, R_cv):
        super()._reinitialize(z, K_cam, cam_pos, R_cv)  # sets the 5 states of CubeEKF and P = P0
        self.x = np.append(self.x, 0.0)
