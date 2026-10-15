"""Wrist camera noise check: the noise-free path is unchanged, and the noisy frames follow CameraNoiseConfig.

Test script: compares noisy frames with the exact render (ground truth), which the pipeline must not do.

Run from the repo root with the venv active:
    python scripts/check_camera_noise.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim import CameraNoiseConfig, TaskSim  # noqa: E402

N_SEEDS = 10
CFG = CameraNoiseConfig()


def edge_pixels(z: np.ndarray) -> np.ndarray:
    jump = np.zeros(z.shape, dtype=bool)
    dv = np.abs(np.diff(z, axis=0)) > CFG.edge_jump
    du = np.abs(np.diff(z, axis=1)) > CFG.edge_jump
    jump[1:] |= dv
    jump[:-1] |= dv
    jump[:, 1:] |= du
    jump[:, :-1] |= du
    return jump


def check_noise_off() -> bool:
    print("== 1. Noise off ==")
    sim = TaskSim()
    sim.reset(0)
    rgb, depth = sim.read_camera()
    rgb_exact, depth_exact = sim.camera.render(sim.data)
    ok = np.array_equal(rgb, rgb_exact) and np.array_equal(depth, depth_exact)
    print(f"  read_camera() equals the exact render: {'PASS' if ok else 'FAIL'}")
    return ok


def check_statistics(sim: TaskSim) -> bool:
    print(f"\n== 2. Noise statistics ({N_SEEDS} seeds, observe pose) ==")
    residuals, rgb_err, invalid, dropped_share = [], [], [], []
    for seed in range(N_SEEDS):
        sim.reset(seed)
        rgb_exact, z_exact = sim.camera.render(sim.data)
        rgb, z = sim.read_camera()
        z_exact = z_exact.astype(float)
        cos_theta = np.maximum(sim.camera_noise._cos_incidence(z_exact), CFG.cos_min)
        sigma = CFG.k_axial * z_exact**2 / cos_theta
        valid = z > 0
        edges = edge_pixels(z_exact)
        interior = valid & ~edges
        residuals.append((z - z_exact)[interior] / sigma[interior])
        rgb_err.append((rgb.astype(float) - rgb_exact).ravel())
        invalid.append(np.mean(~valid & ~edges))
        dropped_share.append(np.sum(~valid & edges) / max(np.sum(edges), 1))
    r = np.concatenate(residuals)
    # 1 mm quantisation adds about (1 mm)^2 / 12 to a ~2 mm sigma: expected std of r is about 1.02.
    depth_ok = abs(r.mean()) < 0.02 and abs(r.std() - 1.0) < 0.05
    rgb_std = np.concatenate(rgb_err).std()
    rgb_ok = abs(rgb_std / CFG.rgb_std - 1) < 0.05
    edge_ok = abs(np.mean(dropped_share) - CFG.edge_dropout) < 0.05 and max(invalid) < 1e-3
    print(f"  depth error / sigma_z(d, theta): mean {r.mean():+.3f}, std {r.std():.3f} (expected ~1.02)  "
          f"{'PASS' if depth_ok else 'FAIL'}")
    print(f"  RGB error std {rgb_std:.2f} (configured {CFG.rgb_std})  {'PASS' if rgb_ok else 'FAIL'}")
    print(f"  edge pixels dropped {np.mean(dropped_share):.2f} (configured {CFG.edge_dropout}), "
          f"invalid away from edges {max(invalid):.1e}  {'PASS' if edge_ok else 'FAIL'}")
    return depth_ok and rgb_ok and edge_ok


def check_reproducible(sim: TaskSim) -> bool:
    print("\n== 3. Reproducibility ==")
    sim.reset(5)
    a = sim.read_camera()
    sim.reset(5)
    for _ in range(3):
        sim.read_camera_for_display()
    b = sim.read_camera()
    ok = np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    print(f"  same seed gives the same noisy frame, display reads do not change it: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ok = check_noise_off()
    sim = TaskSim(camera_noise=CFG)
    ok &= check_statistics(sim)
    ok &= check_reproducible(sim)
    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
