"""Check the occluder estimate (src/perception/occluders.py) against the true occluder boxes.

For every seed the arm stays at the observe pose; one wrist frame goes through estimate_occluders with the
pipeline's camera model (calibrated by default) and the encoder reading. Each true occluder is matched to the
nearest estimate within MATCH_DIST, and the script reports centre, top-height, side-length and yaw errors,
misses (split into "in view" and "out of view", from projecting the true top-face corners with the true camera)
and estimates that match no occluder.

This is an evaluation script: it reads simulator ground truth freely. Nothing here goes into the pipeline.

Run from the repo root with the venv active:
    python scripts/check_occluders.py                 # L1, seeds 0-99, noisy camera
    python scripts/check_occluders.py --no-camera-noise --seeds 50 -v
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.ur3e import UR3eModel  # noqa: E402
from src.perception.camera_model import CameraModel  # noqa: E402
from src.perception.occluders import estimate_occluders  # noqa: E402
from src.planning.clearance import conservative_box  # noqa: E402
from src.sim import CameraNoiseConfig, TaskSim  # noqa: E402
from src.sim.randomize import TABLE_Z  # noqa: E402

MATCH_DIST = 0.05  # m, true centre to estimated centre
COVER_TOL = 0.006  # m, a true corner this far outside the conservative box still counts as covered
W, H = 640, 480


def wrap90(a: float) -> float:
    """Yaw difference of two rectangles, which repeat every 90 degrees, in [-45, 45) degrees."""
    return float((np.rad2deg(a) + 45.0) % 90.0 - 45.0)


def corners_in_view(o, gt) -> int:
    """How many of the true top-face corners project inside the image (true camera)."""
    c, s = np.cos(o.yaw), np.sin(o.yaw)
    pts = [np.array([o.x + c * dx * o.half_x - s * dy * o.half_y, o.y + s * dx * o.half_x + c * dy * o.half_y,
                     TABLE_Z + o.height]) for dx in (-1, 1) for dy in (-1, 1)]
    R_cv = gt["cam_mat"] @ np.diag([1.0, -1.0, -1.0])
    n = 0
    for p in pts:
        pc = R_cv.T @ (p - gt["cam_pos"])
        if pc[2] <= 0:
            continue
        u, v = (gt["K"] @ (pc / pc[2]))[:2]
        n += 0 <= u < W and 0 <= v < H
    return n


def covers(occ_est, o, tol: float = COVER_TOL) -> tuple[bool, float]:
    """Whether the conservative box of the estimate contains every true corner (within tol), and the worst
    corner's distance outside it (m, <= 0 inside)."""
    center, sides, yaw = conservative_box(occ_est)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    c, s = np.cos(o.yaw), np.sin(o.yaw)
    worst = -np.inf
    for dx in (-1, 1):
        for dy in (-1, 1):
            p = np.array([o.x + c * dx * o.half_x - s * dy * o.half_y, o.y + s * dx * o.half_x + c * dy * o.half_y])
            local = R.T @ (p - center)
            worst = max(worst, float(np.max(np.abs(local) - sides / 2)))
    return worst <= tol, worst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--no-camera-noise", action="store_true")
    p.add_argument("--nominal-camera", action="store_true", help="nominal K and mount instead of calibration")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    sim = TaskSim(camera_noise=None if args.no_camera_noise else CameraNoiseConfig())
    robot = UR3eModel()
    camera = CameraModel.nominal(robot) if args.nominal_camera else CameraModel.from_calibration()

    err_xy, err_z, err_side, err_yaw, rows = [], [], [], [], []
    part_xy, part_side = [], []
    cover = {True: [], False: []}  # truncated or not -> worst corner outside the conservative box
    miss_in, miss_out, partial_out, extra, n_true, n_trunc = 0, 0, 0, 0, 0, 0
    for seed in range(args.seeds):
        sim.reset(seed, "L1")
        gt = sim.ground_truth()
        rgb, depth = sim.read_camera()
        cam_pos, cam_rot = camera.pose(robot, sim.read_joint_positions())
        est = estimate_occluders(rgb, depth, camera.K, cam_pos, cam_rot)
        used = set()
        for o in sim.trial.occluders:
            n_true += 1
            view = corners_in_view(o, gt)
            d = [np.hypot(e.xy[0] - o.x, e.xy[1] - o.y) for e in est]
            j = int(np.argmin(d)) if d else -1
            if j < 0 or d[j] > MATCH_DIST or j in used:
                if view == 4:
                    miss_in += 1
                else:
                    miss_out += 1
                rows.append(f"seed {seed:3d}  MISS  true ({o.x:.3f}, {o.y:.3f}) h {o.height:.3f}  "
                            f"corners in view {view}/4")
                continue
            used.add(j)
            e = est[j]
            n_trunc += e.truncated
            partial_out += view < 4
            cover[bool(e.truncated)].append(covers(e, o)[1])
            true_sides = np.sort([2 * o.half_x, 2 * o.half_y])[::-1]
            side = 2 * e.half_xy - true_sides
            yaw = wrap90(e.yaw - o.yaw)
            if view == 4:  # errors only for fully visible occluders; partial ones are reported apart
                err_xy.append(d[j])
                err_z.append(e.top_z - (TABLE_Z + o.height))
                err_side.extend(side)
                err_yaw.append(yaw)
            else:
                part_xy.append(d[j])
                part_side.extend(side)
            rows.append(f"seed {seed:3d}  xy {d[j] * 1e3:5.1f} mm  top {(e.top_z - TABLE_Z - o.height) * 1e3:+5.1f} mm  "
                        f"sides {side[0] * 1e3:+5.1f}/{side[1] * 1e3:+5.1f} mm  yaw {yaw:+5.1f} deg  "
                        f"pts {e.n_points:4d}  view {view}/4{'  TRUNC' if e.truncated else ''}")
        extra += len(est) - len(used)
        if len(est) > len(used):
            rows.append(f"seed {seed:3d}  EXTRA {len(est) - len(used)} estimate(s) with no occluder")
    if args.verbose:
        print("\n".join(rows))

    def stat(name, a, scale, unit):
        a = np.asarray(a) * scale
        print(f"  {name:12s} mean {a.mean():+6.2f}  std {a.std():5.2f}  |max| {np.abs(a).max():5.2f} {unit}")

    noise = "exact" if args.no_camera_noise else "noisy"
    print(f"\nL1, {args.seeds} seeds, {noise} camera, {camera.source} camera model: {n_true} true occluders")
    print(f"  missed: {miss_in} fully in view, {miss_out} partly or fully out of view; "
          f"estimates with no occluder: {extra}; matched but partly out of view: {partial_out} "
          f"(truncated flag set on {n_trunc} matches)")
    if err_xy:
        print(f"  errors over the {len(err_xy)} matched occluders fully in view:")
        stat("centre xy", err_xy, 1e3, "mm")
        stat("top z", err_z, 1e3, "mm")
        stat("side length", err_side, 1e3, "mm")
        stat("yaw", err_yaw, 1.0, "deg")
    for trunc, w in cover.items():
        if w:
            w = np.array(w) * 1e3
            print(f"  conservative box, {'truncated' if trunc else 'not truncated'} ({len(w)}): covers the true box "
                  f"(corners <= {COVER_TOL * 1e3:.0f} mm outside) in {np.sum(w <= COVER_TOL * 1e3)}; "
                  f"worst corner outside {w.max():+.1f} mm")
    if part_xy:
        print(f"  errors over the {len(part_xy)} matched occluders partly out of view:")
        stat("centre xy", part_xy, 1e3, "mm")
        stat("side length", part_side, 1e3, "mm")


if __name__ == "__main__":
    main()
