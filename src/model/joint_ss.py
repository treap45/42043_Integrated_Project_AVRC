
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.linalg import expm

from .ur3e import N_JOINTS, UR3eModel


@dataclass(frozen=True)
class JointSSModel:
    """State x = [q (rad), qd (rad/s), d (N m)], input u (N m), output y = q (rad)."""

    inertia: float  # J_i, kg m^2
    dt: float  # sample period, s

    def __post_init__(self):
        if not np.isfinite(self.inertia) or self.inertia <= 0:
            raise ValueError("inertia must be a positive finite value")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be a positive finite value")

    @property
    def A(self) -> np.ndarray:
        J = self.inertia
        return np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0 / J],
            [0.0, 0.0, 0.0],
        ])

    @property
    def B(self) -> np.ndarray:
        return np.array([
            [0.0],
            [1.0 / self.inertia],
            [0.0],
        ])

    @property
    def C(self) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0]])

    @cached_property
    def _zoh(self) -> tuple[np.ndarray, np.ndarray]:
        # Zero-order hold: the top-right block of exp([[A, B], [0, 0]] * dt) is Bd.
        augmented = np.zeros((4, 4))
        augmented[:3, :3] = self.A
        augmented[:3, 3:4] = self.B
        result = expm(augmented * self.dt)
        return result[:3, :3], result[:3, 3:4]

    @property
    def Ad(self) -> np.ndarray:
        return self._zoh[0].copy()

    @property
    def Bd(self) -> np.ndarray:
        return self._zoh[1].copy()

    def discretize(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact zero-order-hold discretisation (Ad, Bd), computed once and cached."""
        return self.Ad, self.Bd

    def discretize_closed_form(self) -> tuple[np.ndarray, np.ndarray]:
        """Hand-derived (Ad, Bd); used to cross-check discretize()."""
        J, dt = self.inertia, self.dt
        Ad = np.array([
            [1.0, dt, dt**2 / (2.0 * J)],
            [0.0, 1.0, dt / J],
            [0.0, 0.0, 1.0],
        ])
        Bd = np.array([
            [dt**2 / (2.0 * J)],
            [dt / J],
            [0.0],
        ])
        return Ad, Bd


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """[B, AB, ..., A^(n-1) B]."""
    n = A.shape[0]
    return np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])


def observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """[C; CA; ...; C A^(n-1)]."""
    n = A.shape[0]
    return np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])


def build_joint_models(robot: UR3eModel, q_op: np.ndarray, dt: float) -> list[JointSSModel]:
    """One model per arm joint, with J_i taken from the diagonal of M(q_op)."""
    M = robot.mass_matrix(q_op)

    if M.shape != (N_JOINTS, N_JOINTS):
        raise ValueError(f"Expected a {N_JOINTS}x{N_JOINTS} arm mass matrix, got {M.shape}")

    return [JointSSModel(float(J_i), dt) for J_i in np.diag(M)]
