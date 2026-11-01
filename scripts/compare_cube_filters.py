"""Compare the cube filters on the same measurements, for straight and curved cube motion.

Filters: world-frame KF (constant velocity), CubeEKF (constant velocity, pixel/depth measurement), the same EKF with a
larger sigma_a, which is the plain way to let a constant-velocity filter follow a turn, and CubeCTEKF (turn rate in the
state). Each seed runs the moving-cube task once, up to 5 s after the end of TRACK, with CubeEKF in the loop, and
records every vision frame: measurement [u, v, Z, psi], camera pose and the true cube state. The filters then replay
that record offline, so they see exactly the same data and neither one influences the arm's motion.

Reported per filter: position and velocity RMS and NEES (5 states shared by all filters, ideal 5) while the cube moves,
from 0.5 s after its start to its stop; NIS divided by the measurement size (ideal 1); rejected frames and
re-initialisations; the time the cube is declared stopped, after the true stop, using the velocity conditions of
PickPlace._track; and the position error after a GAP_S gap with no measurements while the cube moves (a second replay
with those frames removed), the case of a cube hidden from the camera.

This is an evaluation script: it reads simulator ground truth freely. Nothing here goes into the pipeline.

Run from the repo root with the venv active:
    python scripts/compare_cube_filters.py                     # line, arc and S-curve, seeds 0-19
    python scripts/compare_cube_filters.py --motions arc --seeds 5 --r-scales 1 0.1
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.control.arm_controller import ArmController  # noqa: E402
from src.estimation.cube_ct_ekf import CubeCTEKF  # noqa: E402
from src.estimation.cube_ekf import CubeEKF, _wrap90  # noqa: E402
from src.estimation.cube_kf import CubeWorldKF  # noqa: E402
from src.model.ur3e import UR3eModel  # noqa: E402
from src.perception.camera_model import CameraModel  # noqa: E402
from src.sim import CameraNoiseConfig, CubeMotionConfig, TaskSim  # noqa: E402
from src.task.pick_place import STOP_CHI2, STOP_COUNT, STOP_SPEED, PickPlace  # noqa: E402

EXTRA_RECORD_S = 5.0  # s recorded after TRACK ends
FRAME_PERIOD = 0.1  # s, the vision period of TRACK
GAP_S = 1.0  # s without measurements
GAP_END_BEFORE_STOP = 0.5  # s between the end of the gap and the true stop
SPEED = (0.03, 0.04)  # m/s
DURATION = (6.0, 8.0)  # s
END_X, END_Y = (0.22, 0.42), (-0.25, 0.25)  # m


def record(sim, robot, camera, seed) -> dict:
    """Run one seed to the end of TRACK and return the initial state and the per-frame record."""
    sim.reset(seed)
    ctrl = ArmController(robot=robot, dt=sim.dt, y0=sim.read_joint_positions(), mode="lqr", schedule_period=50)
    pp = PickPlace(sim, robot, ctrl, camera=camera, approach="track")
    rec = {"frames": [], "x0": None, "true_stop": sim.cube_motion.stop_time, "start": sim.cube_motion.start_time,
           "dt": sim.dt}
    measure, track = pp._measure_cube, pp._track

    def logged_measure():
        m = measure()
        gt = sim.ground_truth()
        rec["frames"].append({
            "t": sim.time, "z": None if m is None else m[1], "cam_pos": None if m is None else m[2],
            "R_cv": None if m is None else m[3], "xy": gt["cube_pos"][:2].copy(), "v": gt["cube_vel"][:2].copy(),
            "yaw": 2.0 * np.arctan2(gt["cube_quat"][3], gt["cube_quat"][0]),
        })
        return m

    def logged_track(cube, psi):
        rec["x0"] = np.array([cube.pos[0], cube.pos[1], cube.yaw, 0.0, 0.0])
        track(cube, psi)
        # Keep recording while holding: the record must not end when the in-loop EKF declares the stop, or a slower
        # filter would look as if it never declares it.
        q_hold = ctrl.xhat[:, 0].copy()
        for _ in range(int(EXTRA_RECORD_S / FRAME_PERIOD)):
            pp._hold("EXTRA_RECORD", FRAME_PERIOD, q_hold=q_hold)
            logged_measure()
        return None  # the rest of the task is not needed: end the run here

    pp._measure_cube, pp._track = logged_measure, logged_track
    pp.run()
    return rec


def replay(rec, make_filter, camera, gap: tuple | None = None) -> dict:
    """Run a filter over a record: predict up to each frame's time, then update, as in PickPlace._track.

    gap = (t0, t1): frames inside it are not given to the filter, and the result holds the position error at t1.
    """
    f = make_filter(rec["x0"], rec["dt"])
    moving_from, moving_to = rec["start"] + 0.5, rec["true_stop"]
    err_p, err_v, nees, nis = [], [], [], []
    rejected, reinit, still, stop_t, previous, gap_err = 0, 0, 0, None, None, None
    for frame in rec["frames"]:
        steps = 1 if previous is None else round((frame["t"] - previous) / rec["dt"])
        previous = frame["t"]
        for _ in range(steps):
            f.predict()
        hidden = gap is not None and gap[0] <= frame["t"] <= gap[1]
        if hidden:
            gap_err = float(np.linalg.norm(f.position - frame["xy"]))
        if frame["z"] is not None and not hidden:
            status = f.update(frame["z"], camera.K, frame["cam_pos"], frame["R_cv"])
            rejected += status == "rejected"
            reinit += status == "reinit"
            if status == "accepted":
                nis.append(f.nis / f.R.shape[0])
        e = np.array([*(f.position - frame["xy"]), _wrap90(f.yaw - frame["yaw"]), *(f.velocity - frame["v"])])
        if moving_from <= frame["t"] <= moving_to:
            err_p.append(e[:2])
            err_v.append(e[3:])
            nees.append(float(e @ np.linalg.solve(f.P[:5, :5], e)))
        moving = np.linalg.norm(f.velocity) >= STOP_SPEED or f.velocity_test() >= STOP_CHI2
        still = 0 if moving else still + 1
        if stop_t is None and still >= STOP_COUNT and frame["t"] > rec["true_stop"] - 1.0:
            stop_t = frame["t"]
    return {"p": np.array(err_p), "v": np.array(err_v), "nees": nees, "nis": nis, "rejected": rejected,
            "reinit": reinit, "delay": None if stop_t is None else stop_t - rec["true_stop"], "gap_err": gap_err}


def summary(name, results, gaps) -> None:
    rms = lambda a: float(np.sqrt(np.mean(np.sum(a**2, axis=1))))  # noqa: E731
    delays = [r["delay"] for r in results if r["delay"] is not None]
    g = np.array([r["gap_err"] for r in gaps]) * 1e3
    print(f"  {name:20s} pos {rms(np.vstack([r['p'] for r in results])) * 1e3:5.2f} mm  "
          f"vel {rms(np.vstack([r['v'] for r in results])) * 1e3:5.2f} mm/s  "
          f"NEES {np.median(np.concatenate([r['nees'] for r in results])):7.2f}  "
          f"NIS/m {np.mean(np.concatenate([r['nis'] for r in results])):5.2f}  "
          f"rej {sum(r['rejected'] for r in results):3d} reinit {sum(r['reinit'] for r in results):3d}  "
          f"stop {np.median(delays) if delays else float('nan'):5.2f} s ({len(delays)}/{len(results)})  "
          f"gap {np.median(g):5.2f} mm (max {g.max():5.2f})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--motions", nargs="+", default=["line", "arc", "s_curve"], choices=["line", "arc", "s_curve"])
    p.add_argument("--r-scales", type=float, nargs="+", default=[1.0], help="factors applied to each filter's R")
    args = p.parse_args()

    robot = UR3eModel()
    camera = CameraModel.from_calibration()
    filters = [("KF", CubeWorldKF, {}), ("EKF", CubeEKF, {}), ("EKF sigma_a 0.03", CubeEKF, {"sigma_a": 0.03}),
               ("CT-EKF", CubeCTEKF, {})]
    print(f"cube speed {SPEED} m/s, duration {DURATION} s; NEES is the median while the cube moves")
    for motion in args.motions:
        cfg = CubeMotionConfig(kind=motion, speed=SPEED, duration=DURATION, end_x=END_X, end_y=END_Y, max_tries=20000,
                               tries_per_start=200)
        sim = TaskSim(camera_noise=CameraNoiseConfig(), cube_motion=cfg)
        records = [record(sim, robot, camera, s) for s in range(args.seeds)]
        sim.close()
        print(f"{motion}: {args.seeds} seeds, {sum(len(r['frames']) for r in records)} frames")
        for scale in args.r_scales:
            for name, cls, kwargs in filters:
                def make_filter(x0, dt, cls=cls, kwargs=kwargs, scale=scale):
                    f = cls(x0, dt, **kwargs)
                    f.R = f.R * scale
                    return f
                results = [replay(r, make_filter, camera) for r in records]
                gaps = [replay(r, make_filter, camera, (r["true_stop"] - GAP_END_BEFORE_STOP - GAP_S,
                                                        r["true_stop"] - GAP_END_BEFORE_STOP)) for r in records]
                summary(f"{name}, R x {scale:g}" if len(args.r_scales) > 1 else name, results, gaps)


if __name__ == "__main__":
    main()
