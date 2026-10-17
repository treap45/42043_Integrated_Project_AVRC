"""Check GoalRelativeQuintic with a fixed goal: it starts at p0, ends exactly on the goal pose, and its
feed-forward integrates to the total displacement and rotation (so it agrees with the reference itself).

    python scripts/check_transit_ref.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.planning.trajectory import GoalRelativeQuintic  # noqa: E402


def check_transit_reference(p0, R0, g0, R_goal):
    traj = GoalRelativeQuintic(p0, R0, g0, R_goal)
    times = np.linspace(0.0, traj.T, 1001)

    p_start, _, _, _ = traj(0.0, g0)
    p_end, R_end, _, _ = traj(traj.T, g0)

    ff = np.stack([traj(t, g0)[2] for t in times])
    ff_integral = np.trapezoid(ff, times, axis=0)

    np.testing.assert_allclose(p_start, p0, atol=1e-9)
    np.testing.assert_allclose(p_end, g0, atol=1e-9)
    np.testing.assert_allclose(R_end, R_goal, atol=1e-8)
    np.testing.assert_allclose(
        ff_integral[:3], -(np.asarray(p0) - np.asarray(g0)), atol=1e-5
    )
    np.testing.assert_allclose(
        ff_integral[3:],
        R0 @ Rotation.from_matrix(R0.T @ R_goal).as_rotvec(),
        atol=1e-5,
    )
    return traj.T


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for i in range(20):
        R0 = Rotation.random(random_state=rng).as_matrix()
        R_goal = Rotation.random(random_state=rng).as_matrix()
        T = check_transit_reference(rng.normal(size=3) * 0.3, R0, rng.normal(size=3) * 0.3, R_goal)
    print("20 random start/goal pairs: start, end, and feed-forward integral all match")
