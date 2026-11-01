"""Full pick-and-place trials (closed loop) with a chosen cube motion and TRACK estimator.

Reports placed and grasped counts, grasp offset and place error, how often the cube is declared stopped before it has
really stopped (the arm then starts to descend while the cube will still move), the delay of the other stops, hover
error during TRACK, and rejected frames and re-initialisations. Ground truth is read only by main.py's TrialMonitor and
for the true stop time.

Run from the repo root with the venv active:
    python scripts/run_cube_motion_trials.py --motion variable --filter cv --seeds 20
    python scripts/run_cube_motion_trials.py --motion long_pause --pause 1.5 --speed 0.01 0.02
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as pipeline  # noqa: E402
from src.model.ur3e import UR3eModel  # noqa: E402
from src.perception.camera_model import CameraModel  # noqa: E402
from src.sim import CameraNoiseConfig, CubeMotionConfig, TaskSim  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", default="line", choices=["line", "arc", "s_curve", "variable", "long_pause"])
    p.add_argument("--filter", default="cv", choices=["cv", "ct"])
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--speed", type=float, nargs=2, default=[0.03, 0.04], metavar=("LO", "HI"), help="m/s")
    p.add_argument("--duration", type=float, nargs=2, default=[6.0, 8.0], metavar=("LO", "HI"), help="s")
    p.add_argument("--pause", type=float, default=2.0, help="s, the pause in long_pause")
    p.add_argument("--turn-radius", type=float, nargs=2, default=[0.10, 0.16], metavar=("LO", "HI"), help="m")
    args = p.parse_args()

    cfg = CubeMotionConfig(kind=args.motion, speed=tuple(args.speed), duration=tuple(args.duration), end_x=(0.22, 0.42),
                           end_y=(-0.25, 0.25), turn_radius=tuple(args.turn_radius), pause_s=args.pause,
                           max_tries=20000, tries_per_start=200)
    sim = TaskSim(camera_noise=CameraNoiseConfig(), cube_motion=cfg)
    robot = UR3eModel()
    camera = CameraModel.from_calibration()

    rows = []
    for seed in range(args.seeds):
        r = pipeline.run_trial(sim, robot, camera, seed, "L0", "lqr", 60.0, cube_filter=args.filter)
        rows.append(r)
        t = r["track"] or {}
        print(f"seed {seed:3d}  {'OK  ' if r['placed'] else 'FAIL'} grasped={r['grasped']!s:5s} "
              f"stop_delay {t.get('stop_delay')}  place {r['place_err'] * 1e3:6.1f} mm  {r['reason'] or ''}", flush=True)
    sim.close()

    delays = np.array([r["track"]["stop_delay"] for r in rows if r["track"] and r["track"]["stop_delay"] is not None])
    placed = [r for r in rows if r["placed"]]
    offsets = [r["grasp_offset"] * 1e3 for r in rows if r["grasp_offset"] is not None]
    fails = {}
    for r in rows:
        if not r["placed"]:
            key = r["reason"] or "success criteria not met"
            fails[key] = fails.get(key, 0) + 1
    late = delays[delays >= 0]
    print(f"\n{args.motion} {args.filter} speed {args.speed} pause {args.pause}: placed {len(placed)}/{len(rows)}, "
          f"grasped {sum(r['grasped'] for r in rows)}/{len(rows)}")
    print(f"  stop declared early {np.sum(delays < 0)}, late delay median "
          f"{np.median(late) if late.size else float('nan'):.2f} s, no stop {len(rows) - delays.size}")
    if offsets:
        print(f"  grasp offset median {np.median(offsets):.1f} mm, max {max(offsets):.1f} mm")
    if placed:
        err = np.array([r["place_err"] for r in placed]) * 1e3
        print(f"  place error median {np.median(err):.1f} mm, max {err.max():.1f} mm")
    print(f"  rejected {sum((r['track'] or {}).get('rejected', 0) for r in rows)}, "
          f"reinit {sum((r['track'] or {}).get('reinit', 0) for r in rows)}, failures {fails}")


if __name__ == "__main__":
    main()
