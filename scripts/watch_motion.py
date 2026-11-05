"""Watch one harder-motion trial live, with the same settings as scripts/run_cube_motion_trials.py at its defaults.

Run from the repo root with the venv active (q or Esc in the window ends it):
    python scripts/watch_motion.py --motion arc
    python scripts/watch_motion.py --motion s_curve
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as pipeline  # noqa: E402
from src.model.ur3e import UR3eModel  # noqa: E402
from src.perception.camera_model import CameraModel  # noqa: E402
from src.sim import CameraNoiseConfig, CubeMotionConfig, TaskSim  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--motion", required=True, choices=["arc", "s_curve"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--record", type=Path, help="also write an mp4")
args = p.parse_args()

cfg = CubeMotionConfig(kind=args.motion, speed=(0.03, 0.04), duration=(6.0, 8.0), end_x=(0.22, 0.42),
                       end_y=(-0.25, 0.25), turn_radius=(0.10, 0.16), pause_s=2.0,
                       max_tries=20000, tries_per_start=200)
sim = TaskSim(camera_noise=CameraNoiseConfig(), cube_motion=cfg)
display = pipeline.Display(sim, True, args.record)
try:
    r = pipeline.run_trial(sim, UR3eModel(), CameraModel.from_calibration(), args.seed, "L0", "lqr", 60.0,
                           display=display, cube_filter="cv")
    t = r["track"] or {}
    print(f"seed {args.seed:3d}  {'OK  ' if r['placed'] else 'FAIL'} grasped={r['grasped']!s:5s} "
          f"stop_delay {t.get('stop_delay')}  place {r['place_err'] * 1e3:6.1f} mm  {r['reason'] or ''}", flush=True)
finally:
    display.close()
    sim.close()
