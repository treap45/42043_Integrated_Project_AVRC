"""Measure the D435i depth noise on the table, to set the noise level the fixed-camera pipeline is tested with.

Point the camera at the table as it will be used, with nothing in a central patch, and run. The script averages
nothing: it takes N frames, and in a patch of the image reports the per-pixel standard deviation over time and the
residual of a plane fitted to one frame, then the k_axial of the simulation noise model (sigma = k Z^2) they imply.
scripts/run_real.py --sim --depth-k <k> then tests the pipeline at that level.

    python scripts/check_real_depth.py
    python scripts/check_real_depth.py --patch 300 200 80     # centre u, v and half size in pixels
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.real.camera import D435i  # noqa: E402
from src.real.io import SIM_DEPTH_K, average_depth  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--patch", type=int, nargs=3, default=None, metavar=("U", "V", "HALF"))
    args = p.parse_args()

    with D435i() as cam:
        frames = [cam.capture().depth_m for _ in range(args.frames)]
        K = cam.K
    D = np.stack(frames)
    h, w = D.shape[1:]
    u0, v0, half = args.patch if args.patch else (w // 2, h // 2, 60)
    P = D[:, v0 - half:v0 + half, u0 - half:u0 + half]
    valid = np.all(P > 0, axis=0)
    if valid.sum() < 100:
        sys.exit("too few valid depth pixels in the patch; move it with --patch")
    z = float(np.median(P[:, valid]))
    temporal = float(np.median(np.std(P[:, valid], axis=0)))

    # Plane fit to one frame (the table may be tilted in the camera frame).
    vv, uu = np.nonzero(valid)
    zz = P[0][valid]
    x = (uu + u0 - half - K[0, 2]) * zz / K[0, 0]
    y = (vv + v0 - half - K[1, 2]) * zz / K[1, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, zz, rcond=None)
    spatial = float(np.std(zz - A @ coef))
    avg = average_depth(frames[:4])[v0 - half:v0 + half, u0 - half:u0 + half][valid]
    spatial4 = float(np.std(avg - A @ coef))

    k_t, k_s = temporal / z**2, spatial / z**2
    print(f"patch ({u0}, {v0}) +/- {half} px: {valid.sum()} valid pixels, depth {z:.3f} m, "
          f"{100 * np.mean(D[:, v0 - half:v0 + half, u0 - half:u0 + half] > 0):.0f} % valid readings")
    print(f"  temporal std per pixel  {1000 * temporal:5.2f} mm  -> k = {k_t:.4f} 1/m")
    print(f"  plane residual, 1 frame {1000 * spatial:5.2f} mm  -> k = {k_s:.4f} 1/m")
    print(f"  plane residual, 4-frame average {1000 * spatial4:5.2f} mm")
    print(f"  pipeline tested with k = {SIM_DEPTH_K} by default: "
          + ("the lab camera is no worse" if max(k_t, k_s) <= SIM_DEPTH_K else
             f"the lab camera is noisier; rerun  python scripts/run_real.py --sim --depth-k {max(k_t, k_s):.4f}"))


if __name__ == "__main__":
    main()
