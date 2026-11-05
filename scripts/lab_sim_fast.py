"""scripts/run_real.py --sim with the cube moving at about hand-push speed (5-6 cm/s); the code under test is unchanged.

    python scripts/lab_sim_fast.py --sim --mode moving --hover-mm 150 --v-max 0.10 --no-window --seed 0

Some seeds have no valid path at this speed and stop with "could not place a cube motion"; the project page used seeds
0, 1, 4, 5, 6, 8, 9, 10, 11, 13.
"""

import functools
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.real.io as real_io  # noqa: E402
import src.sim as sim  # noqa: E402

fast = functools.partial(sim.CubeMotionConfig, speed=(0.05, 0.06), duration=(4.0, 6.0), end_x=(0.22, 0.42), end_y=(-0.25, 0.25))
real_io.CubeMotionConfig = fast
sim.CubeMotionConfig = fast

runpy.run_path(str(ROOT / "scripts/run_real.py"), run_name="__main__")
