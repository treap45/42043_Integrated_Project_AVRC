# Run from the root of branch `advanced`, with main.py arguments. Replays one L1 seed and reports when any geom touches an occluder, and the grasp yaw chosen.
import sys, runpy
from pathlib import Path
import numpy as np, mujoco
root = Path.cwd(); sys.path.insert(0, str(root))
from src.sim.env import TaskSim
from src.task.pick_place import PickPlace
step0 = TaskSim.step; seen = {}
def step(self, tau, n_steps=1):
    step0(self, tau, n_steps)
    m, d = self.model, self.data
    occ = {m.geom(f"occluder_{i}").id for i in range(3)}
    for c in d.contact[:d.ncon]:
        a, b = int(c.geom1), int(c.geom2)
        if (a in occ) == (b in occ): continue
        o, g = (a, b) if a in occ else (b, a)
        body = m.body(m.geom_bodyid[g]).name
        if body in ("world", "table") : continue
        k = (m.geom(o).name, body)
        if k not in seen:
            seen[k] = float(d.time); print(f"contact t={d.time:.2f}s {k[0]} <-> {k[1]}", flush=True)
TaskSim.step = step
sel0 = PickPlace._select_grasp
def sel(self, *a, **k):
    r = sel0(self, *a, **k)
    if r is not None: print(f"grasp yaw psi={np.degrees(r[0]):.1f} deg cube yaw={np.degrees(a[1]):.1f} deg", flush=True)
    return r
PickPlace._select_grasp = sel
sys.argv = ["main.py"] + sys.argv[1:]
runpy.run_path("main.py", run_name="__main__")
