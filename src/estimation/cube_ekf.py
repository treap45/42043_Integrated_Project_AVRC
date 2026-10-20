"""Extended Kalman filter for the cube on the table: state x = [x, y, psi, vx, vy] (world frame).

Constant-velocity process model; the measurement z = [u, v, Z, psi] is the pixel position and depth of the top-face
centre plus the yaw, so the measurement model goes through the camera projection. Design: docs/cube_ekf_design.md.
"""

import numpy as np

from src.estimation.joint_kf import discrete_process_noise

_A_CV = np.zeros((5, 5))
_A_CV[0, 3] = 1.0
_A_CV[1, 4] = 1.0
Z_TOP = 0.80  # m, top-face centre height of a cube resting on the table (TABLE_Z + 2 * CUBE_HALF)
GATE = 13.28  # chi-square, 4 degrees of freedom, 99 %



def _wrap90(angle: float) -> float:
    """Wrap square yaw to [-pi/4, pi/4)."""
    return float((angle + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)



def h_cube(
    x: np.ndarray,
    K_cam: np.ndarray,
    cam_pos: np.ndarray,
    R_cv: np.ndarray,
) -> np.ndarray:
    """Predicted measurement [u, v, Z, psi]. R_cv maps camera vectors to world."""
    x = np.asarray(x, dtype=float)
    K_cam = np.asarray(K_cam, dtype=float)
    cam_pos = np.asarray(cam_pos, dtype=float)
    R_cv = np.asarray(R_cv, dtype=float)

    p_w = np.array([x[0], x[1], Z_TOP])
    p_c = R_cv.T @ (p_w - cam_pos)
    X, Y, Z = p_c

    if Z <= 1e-8:
        raise ValueError("Cube is behind the camera")

    fx, fy = K_cam[0, 0], K_cam[1, 1]
    cx, cy = K_cam[0, 2], K_cam[1, 2]

    return np.array([
        fx * X / Z + cx,
        fy * Y / Z + cy,
        Z,
        x[2],
    ])

    


def H_cube(
    x: np.ndarray,
    K_cam: np.ndarray,
    cam_pos: np.ndarray,
    R_cv: np.ndarray,
) -> np.ndarray:
    """Jacobian of h_cube with respect to [x, y, psi, vx, vy]."""
    x = np.asarray(x, dtype=float)
    K_cam = np.asarray(K_cam, dtype=float)
    cam_pos = np.asarray(cam_pos, dtype=float)
    R_cv = np.asarray(R_cv, dtype=float)

    p_w = np.array([x[0], x[1], Z_TOP])
    X, Y, Z = R_cv.T @ (p_w - cam_pos)

    if Z <= 1e-8:
        raise ValueError("Cube is behind the camera")

    fx, fy = K_cam[0, 0], K_cam[1, 1]

    J_pi = np.array([
        [fx / Z, 0.0, -fx * X / Z**2],
        [0.0, fy / Z, -fy * Y / Z**2],
        [0.0, 0.0, 1.0],
    ])

    # The camera-frame point changes with world x and y through R_cv.T.
    dpc_dxy = R_cv.T[:, :2]

    H = np.zeros((4, 5))
    H[:3, :2] = J_pi @ dpc_dxy
    H[3, 2] = 1.0
    return H


class CubeEKF:
    """Cube EKF: predict every control period, update whenever a camera measurement is available."""

    def __init__(
        self,
        x0: np.ndarray,
        dt: float,
        P0: np.ndarray | None = None,
        sigma_a: float = 0.01,
        sigma_psi: float = 0.01,
        R: np.ndarray | None = None,
    ):
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")

        self.x = np.asarray(x0, dtype=float).copy()
        if self.x.shape != (5,):
            raise ValueError("x0 must have shape (5,)")
        self.x[2] = _wrap90(self.x[2])

        self.P0 = (
            np.diag([0.002**2, 0.002**2, 0.02**2, 0.03**2, 0.03**2])
            if P0 is None else np.asarray(P0, dtype=float).copy()
        )
        self.R = (
            # Measured on a static cube: about 0.3 px, < 0.5 mm and 0.006 rad (docs/worklog.md, 2026-10-20).
            np.diag([0.5**2, 0.5**2, 0.001**2, 0.01**2])
            if R is None else np.asarray(R, dtype=float).copy()
        )

        if self.P0.shape != (5, 5) or self.R.shape != (4, 4):
            raise ValueError("P0 and R must have shapes (5, 5) and (4, 4)")

        Qc = np.diag([
            0.0,
            0.0,
            sigma_psi**2,
            sigma_a**2,
            sigma_a**2,
        ])

        self.F = np.eye(5)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.Qd = discrete_process_noise(_A_CV, Qc, self.dt)

        self.P = self.P0.copy()
        self._gate_rejects = 0
        self._nis = float("nan")

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Qd
        self.P = 0.5 * (self.P + self.P.T)

    def update(
        self,
        z: np.ndarray,
        K_cam: np.ndarray,
        cam_pos: np.ndarray,
        R_cv: np.ndarray,
    ) -> str:
        z = np.asarray(z, dtype=float)
        # An unusable frame is not evidence that the cube jumped, so it leaves the rejection count alone.
        if z.shape != (4,) or not np.all(np.isfinite(z)) or z[2] <= 0.0:
            return "rejected"

        try:
            z_pred = h_cube(self.x, K_cam, cam_pos, R_cv)
            H = H_cube(self.x, K_cam, cam_pos, R_cv)
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

        # Joseph form preserves covariance symmetry and positive semidefiniteness better.
        I = np.eye(5)
        KH = K @ H
        self.P = (
            (I - KH) @ self.P @ (I - KH).T
            + K @ self.R @ K.T
        )
        self.P = 0.5 * (self.P + self.P.T)
        self._nis = nis

        return "accepted"

    def _reinitialize(self, z, K_cam, cam_pos, R_cv):
        u, v, Z, psi = z
        fx, fy = K_cam[0, 0], K_cam[1, 1]
        cx, cy = K_cam[0, 2], K_cam[1, 2]

        p_c = np.array([
            (u - cx) * Z / fx,
            (v - cy) * Z / fy,
            Z,
        ])
        p_w = R_cv @ p_c + cam_pos

        self.x = np.array([
            p_w[0],
            p_w[1],
            _wrap90(psi),
            0.0,
            0.0,
        ])
        self.P = self.P0.copy()
        self._gate_rejects = 0

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def yaw(self) -> float:
        return float(self.x[2])

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:5].copy()

    @property
    def nis(self) -> float:
        return self._nis

    def velocity_test(self) -> float:
        v = self.x[3:5]
        P_v = self.P[3:5, 3:5]
        return float(v @ np.linalg.solve(P_v, v))
