import numpy as np
from scipy.interpolate import CubicSpline

from src.planning.ik import orientation_error, solve_ik


class QuinticJoint:
    """Quintic joint-space trajectory with zero endpoint velocities."""

    def __init__(
        self,
        q0: np.ndarray,
        q1: np.ndarray,
        v_max: float = 0.5,
        t_min: float = 1.0,
    ):
        self.q0 = np.asarray(q0, dtype=float)
        self.q1 = np.asarray(q1, dtype=float)

        if self.q0.shape != (6,) or self.q1.shape != (6,):
            raise ValueError("q0 and q1 must both have shape (6,)")
        if v_max <= 0 or t_min <= 0:
            raise ValueError("v_max and t_min must be positive")

        self.delta = self.q1 - self.q0
        self.T = max(
            t_min,
            1.875 * np.max(np.abs(self.delta)) / v_max,
        )

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return q_ref and qd_ref at time t."""
        if t >= self.T:
            return self.q1.copy(), np.zeros(6)

        tau = np.clip(float(t) / self.T, 0.0, 1.0)
        s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        ds = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / self.T

        return self.q0 + self.delta * s, self.delta * ds


class CartesianLine:
    """Cartesian straight-line path with precomputed IK and joint interpolation."""

    def __init__(
        self,
        robot,
        q_start: np.ndarray,
        p0: np.ndarray,
        p1: np.ndarray,
        R: np.ndarray,
        T: float = 2.0,
        n: int = 40,
    ):
        self.T = float(T)
        if self.T <= 0 or n < 2:
            raise ValueError("T must be positive and n must be at least 2")

        q_start = np.asarray(q_start, dtype=float)
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        R = np.asarray(R, dtype=float)

        p_start, R_start = robot.site_pose(q_start, "tcp")
        if (
            np.linalg.norm(p_start - p0) > 5e-3
            or np.linalg.norm(orientation_error(R_start, R)) > 5e-2
        ):
            raise ValueError("q_start must match the start Cartesian pose")

        times = np.linspace(0.0, self.T, n + 1)
        q_points = [q_start.copy()]
        q_seed = q_start.copy()

        for i in range(1, n + 1):
            alpha = i / n
            s = 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5
            p = p0 + (p1 - p0) * s

            q_solution, ok = solve_ik(robot, p, R, q_seed)
            if not ok:
                raise ValueError(f"IK failed at Cartesian waypoint {i}/{n}")

            q_points.append(q_solution)
            q_seed = q_solution

        self._spline = CubicSpline(
            times,
            np.asarray(q_points),
            axis=0,
            bc_type=((1, np.zeros(6)), (1, np.zeros(6))),
        )

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return q_ref and qd_ref at time t."""
        if t <= 0.0:
            return self._spline(0.0), np.zeros(6)
        if t >= self.T:
            return self._spline(self.T), np.zeros(6)

        return self._spline(t), self._spline(t, 1)