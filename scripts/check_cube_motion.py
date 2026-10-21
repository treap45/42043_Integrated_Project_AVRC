"""Check the moving-cube task (src/sim CubeMotionConfig): a straight line at constant speed, then a stop.

The arm is held at the observe pose by the joint controller (without it the arm would sag onto the table).
Ground truth is read only to check the cube's motion.

Run from the repo root with the venv active:
    python scripts/check_cube_motion.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.control.arm_controller import ArmController  # noqa: E402
from src.model.ur3e import UR3eModel  # noqa: E402
from src.sim import CubeMotionConfig, TaskSim  # noqa: E402
from src.sim.randomize import CUBE_HALF, TABLE_Z, _segment_distance, sample_trial  # noqa: E402

N_SEEDS = 20
AFTER_STOP = 1.0  # s of physics after the stop, to see that the cube stays where it was left
PATH_TOL = 1e-3  # m, cube xy to the prescribed position
HEIGHT_TOL = 1e-3  # m, cube centre to its resting height
YAW_TOL = np.deg2rad(0.5)
DRIFT_TOL = 1e-3  # m, movement after the stop


def yaw_of(quat: np.ndarray) -> float:
    return 2.0 * np.arctan2(quat[3], quat[0])


def run(sim: TaskSim, robot: UR3eModel, seed: int) -> dict:
    sim.reset(seed)
    m = sim.cube_motion
    ctrl = ArmController(robot, sim.dt, sim.read_joint_positions())
    q_hold = ctrl.xhat[:, 0].copy()
    yaw0 = yaw_of(sim.ground_truth()["cube_quat"])
    path_err = height_err = yaw_err = speed_err = 0.0
    xy_at_stop = None
    while sim.time < m.stop_time + AFTER_STOP:
        sim.step(ctrl.step(sim.read_joint_positions(), q_hold, np.zeros(6)))
        gt, t = sim.ground_truth(), sim.time
        xy_ref, v_ref = m.at(t)
        path_err = max(path_err, float(np.linalg.norm(gt["cube_pos"][:2] - xy_ref)))
        height_err = max(height_err, abs(gt["cube_pos"][2] - (TABLE_Z + CUBE_HALF)))
        yaw_err = max(yaw_err, abs(yaw_of(gt["cube_quat"]) - yaw0))
        if m.start_time < t < m.stop_time:
            speed_err = max(speed_err, float(np.linalg.norm(gt["cube_vel"][:2] - v_ref)))
        if xy_at_stop is None and t >= m.stop_time:
            xy_at_stop = gt["cube_pos"][:2].copy()
    drift = float(np.linalg.norm(sim.ground_truth()["cube_pos"][:2] - xy_at_stop))
    return {"motion": m, "path": path_err, "height": height_err, "yaw": yaw_err, "speed": speed_err,
            "drift": drift, "end": sim.ground_truth()["cube_pos"][:2].copy()}


def main():
    sim = TaskSim(cube_motion=CubeMotionConfig())
    robot = UR3eModel()
    ok = True

    print("== 1. Fixed place zone, path clear of it, cube start kept unless it was in the way ==")
    cfg = CubeMotionConfig()
    zone = np.array(cfg.zone_xy)
    same = True
    for s in range(N_SEEDS):
        trial, static = sim.reset(s), sample_trial(s)
        m = sim.cube_motion
        start_kept = np.array_equal(trial.cube_xy, static.cube_xy)
        start_moved_ok = np.linalg.norm(static.cube_xy - zone) < cfg.zone_clearance
        same &= (np.array_equal(trial.zone_xy, zone) and (start_kept or start_moved_ok)
                 and _segment_distance(zone, m.start_xy, m.end_xy) >= cfg.zone_clearance)
    ok &= same
    print(f"  seeds 0-{N_SEEDS - 1}: {same}")

    print(f"\n== 2. Motion, seeds 0-{N_SEEDS - 1} (arm held at observe) ==")
    for seed in range(N_SEEDS):
        r = run(sim, robot, seed)
        m = r["motion"]
        seed_ok = (r["path"] < PATH_TOL and r["height"] < HEIGHT_TOL and r["yaw"] < YAW_TOL
                   and r["drift"] < DRIFT_TOL and np.hypot(*m.end_xy) <= CubeMotionConfig().max_radius)
        ok &= seed_ok
        print(f"  seed {seed:2d}: speed {np.linalg.norm(m.velocity)*100:4.1f} cm/s  "
              f"distance {np.linalg.norm(m.end_xy - m.start_xy)*100:4.1f} cm  stop at {m.stop_time:4.1f} s  "
              f"path err {r['path']*1e3:4.2f} mm  speed err {r['speed']*1e3:4.2f} mm/s  "
              f"height err {r['height']*1e3:4.2f} mm  yaw drift {np.rad2deg(r['yaw']):4.2f} deg  "
              f"drift after stop {r['drift']*1e3:4.2f} mm  {'PASS' if seed_ok else 'FAIL'}")

    print("\n== 3. Reproducible ==")
    a, b = run(sim, robot, 3)["end"], run(sim, robot, 3)["end"]
    same = np.array_equal(a, b)
    ok &= same
    print(f"  seed 3 run twice, identical final cube position: {same}")

    sim.close()
    print(f"\n  cube motion  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
