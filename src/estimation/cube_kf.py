import numpy as np

from src.estimation.cube_ekf import CubeEKF, _wrap90


GATE_KF = 11.34  # chi-square, 3 degrees of freedom, 99 %
R_WORLD_DEFAULT = np.diag([
    0.0003**2,
    0.0003**2,
    0.01**2,
])


class CubeWorldKF(CubeEKF):
    """Linear KF whose measurements are cube x, y and yaw in the world frame."""

    def __init__(
        self,
        x0: np.ndarray,
        dt: float,
        P0: np.ndarray | None = None,
        sigma_a: float = 0.01,
        sigma_psi: float = 0.01,
        R: np.ndarray | None = None,
    ):
        # CubeEKF validates its measurement covariance as 4x4.
        super().__init__(
            x0=x0,
            dt=dt,
            P0=P0,
            sigma_a=sigma_a,
            sigma_psi=sigma_psi,
            R=np.eye(4),
        )

        self.R = (
            R_WORLD_DEFAULT.copy()
            if R is None
            else np.asarray(R, dtype=float).copy()
        )
        if self.R.shape != (3, 3):
            raise ValueError("R must have shape (3, 3)")

        self.H = np.zeros((3, 5))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

    def update(
        self,
        z: np.ndarray,
        K_cam: np.ndarray,
        cam_pos: np.ndarray,
        R_cv: np.ndarray,
    ) -> str:
        """Back-project [u, v, Z, yaw], then apply a linear KF update."""
        z = np.asarray(z, dtype=float)
        K_cam = np.asarray(K_cam, dtype=float)
        cam_pos = np.asarray(cam_pos, dtype=float)
        R_cv = np.asarray(R_cv, dtype=float)

        # Invalid measurements do not count toward the consecutive gate rejects.
        if (
            z.shape != (4,)
            or not np.all(np.isfinite(z))
            or z[2] <= 0.0
            or K_cam.shape != (3, 3)
            or cam_pos.shape != (3,)
            or R_cv.shape != (3, 3)
        ):
            return "rejected"

        u, v, Z, psi = z
        fx, fy = K_cam[0, 0], K_cam[1, 1]
        cx, cy = K_cam[0, 2], K_cam[1, 2]

        if fx <= 0.0 or fy <= 0.0:
            return "rejected"

        p_c = np.array([
            (u - cx) * Z / fx,
            (v - cy) * Z / fy,
            Z,
        ])
        p_w = R_cv @ p_c + cam_pos
        z_world = np.array([p_w[0], p_w[1], _wrap90(psi)])

        innovation = z_world - self.H @ self.x
        innovation[2] = _wrap90(innovation[2])

        S = self.H @ self.P @ self.H.T + self.R

        try:
            nis = float(innovation @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            return "rejected"

        if not np.isfinite(nis) or nis > GATE_KF:
            self._gate_rejects += 1
            if self._gate_rejects >= 3:
                # Parent method expects the original image-space [u, v, Z, psi].
                self._reinitialize(z, K_cam, cam_pos, R_cv)
                return "reinit"
            return "rejected"

        self._gate_rejects = 0
        K_gain = np.linalg.solve(S, self.H @ self.P).T

        self.x = self.x + K_gain @ innovation
        self.x[2] = _wrap90(self.x[2])

        # Joseph form.
        I = np.eye(5)
        KH = K_gain @ self.H
        self.P = (
            (I - KH) @ self.P @ (I - KH).T
            + K_gain @ self.R @ K_gain.T
        )
        self.P = 0.5 * (self.P + self.P.T)
        self._nis = nis

        return "accepted"