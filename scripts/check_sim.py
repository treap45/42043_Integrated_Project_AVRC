"""Sim interface check: randomisation over many seeds, reproducibility, and encoder noise statistics.

Test script: reads ground truth and holds the arm with MuJoCo's own bias forces, which the pipeline must
not do.

Run from the repo root with the venv active:
    python scripts/check_sim.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim import N_ARM, TaskSim  # noqa: E402
from src.sim.randomize import CUBE_HALF, TABLE_Z  # noqa: E402

N_SEEDS = 200
SETTLE_S = 0.5


def red_pixels(rgb: np.ndarray) -> int:
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    return int(np.sum((r > 120) & (g < 60) & (b < 60)))


def check_levels(sim: TaskSim) -> bool:
    print(f"== 1. Randomisation ({N_SEEDS} seeds per level, {SETTLE_S} s settle) ==")
    ok = True
    for level in ["L0", "L1"]:
        n_occ, pen, moved, visible = [], 0, 0, []
        for seed in range(N_SEEDS):
            sim.reset(seed, level)
            n_occ.append(len(sim.trial.occluders))
            d = sim.data
            pen += any(c.dist < -1e-3 for c in d.contact[: d.ncon])
            rgb, _ = sim.read_camera()
            visible.append(red_pixels(rgb))
            p0 = sim.ground_truth()["cube_pos"]
            for _ in range(int(SETTLE_S / sim.dt)):
                sim.step(d.qfrc_bias[:N_ARM].copy())
            p1 = sim.ground_truth()["cube_pos"]
            moved += np.linalg.norm(p1 - p0) > 1e-3 or abs(p1[2] - TABLE_Z - CUBE_HALF) > 2e-3
        visible = np.array(visible)
        lvl_ok = pen == 0 and moved == 0 and (level != "L0" or visible.min() > 500)
        ok &= lvl_ok
        print(f"  {level}: occluders {min(n_occ)}-{max(n_occ)}, initial penetrations {pen}, cube moved {moved}, "
              f"red pixels min/median {visible.min()}/{int(np.median(visible))}, "
              f"fully hidden {np.sum(visible == 0)}  {'PASS' if lvl_ok else 'FAIL'}")
    return ok


def check_reproducible(sim: TaskSim) -> bool:
    print("\n== 2. Reproducibility ==")
    a = sim.reset(17, "L1")
    qa = [sim.read_joint_positions() for _ in range(5)]
    b = sim.reset(17, "L1")
    qb = [sim.read_joint_positions() for _ in range(5)]
    same = (np.allclose(a.cube_xy, b.cube_xy) and a.cube_yaw == b.cube_yaw and len(a.occluders) == len(b.occluders)
            and np.allclose(qa, qb))
    print(f"  same seed gives same scene and same encoder noise: {'PASS' if same else 'FAIL'}")
    return same


def check_encoders(sim: TaskSim) -> bool:
    print("\n== 3. Encoder model ==")
    sim.reset(0, "L0")
    q_true = sim.ground_truth()["q"]
    err = np.array([sim.read_joint_positions() - q_true for _ in range(5000)])
    res = sim.encoder_cfg.resolution
    on_grid = np.allclose(np.round((err + q_true) / res) * res, err + q_true)
    std = err.std(axis=0)
    expected = np.sqrt(sim.encoder_cfg.noise_std**2 + res**2 / 12)
    ok = on_grid and np.all(np.abs(std / expected - 1) < 0.1) and np.all(np.abs(err.mean(axis=0)) < 3 * res)
    print(f"  noise_std {sim.encoder_cfg.noise_std:.2e} rad, resolution {res:.2e} rad")
    print(f"  measured error std {np.round(std, 6)} rad (expected ~{expected:.2e}), readings on encoder grid: "
          f"{on_grid}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    sim = TaskSim()
    results = {"levels": check_levels(sim), "reproducible": check_reproducible(sim), "encoders": check_encoders(sim)}
    sim.close()
    print("\n== Summary ==")
    for k, v in results.items():
        print(f"  {k:12s} {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
