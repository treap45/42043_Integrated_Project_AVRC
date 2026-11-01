"""Check the occluder-aware grasp-yaw choice (src/planning/clearance.py) against true collisions.

For every seed the arm is at the observe pose; one wrist frame gives the cube estimate and the occluder estimate,
as in the pipeline (noisy camera, calibrated camera model, encoder reading). For each of the four grasp yaws the
script solves IK for the pregrasp and grasp poses of the estimated cube, then places the arm there (kinematics
only) and measures the true occluder-to-robot clearance with mj_geomDistance, as in check_hover_views.py. It
compares two choices:
  - now:   least joint motion among the IK-feasible yaws (PickPlace._select_grasp);
  - aware: least joint motion among the yaws that grasp_clearance() calls clear, for each margin given.
A yaw that grasp_clearance() calls clear but whose true clearance is below zero is a false clear.

This is an evaluation script: it reads simulator ground truth freely. Nothing here goes into the pipeline.

Run from the repo root with the venv active:
    python scripts/check_clearance.py                       # L1, seeds 0-99, margins 5 and 10 mm
    python scripts/check_clearance.py --margins 0.0 0.005 0.01 0.015 --seeds 50
"""

import argparse
import collections
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_hover_views import N_ARM, clearance, robot_geoms, set_arm  # noqa: E402
from src.model.ur3e import UR3eModel  # noqa: E402
from src.perception.camera_model import CameraModel  # noqa: E402
from src.perception.cube_pose import estimate_cube  # noqa: E402
from src.active_vision.next_view import select_next_view  # noqa: E402
from src.perception.occluders import estimate_occluders, fuse_occluders  # noqa: E402
from src.perception.projection import backproject, camera_to_world  # noqa: E402
from src.perception.segmentation import CUBE_HSV, color_mask, largest_component  # noqa: E402
from src.planning.clearance import grasp_clearance  # noqa: E402
from src.planning.grasp import grasp_poses, yaw_candidates  # noqa: E402
from src.planning.ik import solve_ik  # noqa: E402
from src.sim import CameraNoiseConfig, TaskSim  # noqa: E402
from src.sim.randomize import N_OCCLUDERS  # noqa: E402

SAFE_TOL = 0.0  # m, true clearance below this is a collision
BLOCK_TOL = 0.01  # m, a yaw blocked although its true clearance exceeds margin + this is a false block


def run_seed(sim, robot, camera, seed, margins, ids, use_view=True, max_views=0):
    sim.reset(seed, "L1")
    robot_ids, occ_ids = ids
    occ_ids = occ_ids[:len(sim.trial.occluders)]
    rgb, depth = sim.read_camera()
    q_hat = sim.read_joint_positions()
    cam_pos, cam_rot = camera.pose(robot, q_hat)
    mask = largest_component(color_mask(rgb, **CUBE_HSV), min_pixels=100)
    cube = estimate_cube(camera_to_world(backproject(mask, depth, camera.K), cam_pos, cam_rot))
    occluders = estimate_occluders(rgb, depth, camera.K, cam_pos, cam_rot)
    q_observe = sim.data.qpos[:N_ARM].copy()

    cands = []
    for psi in yaw_candidates(cube.yaw):
        poses = grasp_poses(cube.pos, psi)
        q_pre, ok_pre = solve_ik(robot, poses["pregrasp"][0], poses["pregrasp"][1], q_hat)
        q_grasp, ok_grasp = solve_ik(robot, poses["grasp"][0], poses["grasp"][1], q_pre)
        if not (ok_pre and ok_grasp):
            continue
        set_arm(sim, q_pre)
        c_pre, _ = clearance(sim.model, sim.data, robot_ids, occ_ids)
        set_arm(sim, q_grasp)
        c_grasp, hit = clearance(sim.model, sim.data, robot_ids, occ_ids)
        cands.append({"psi": psi, "motion": float(np.sum(np.abs(q_pre - q_hat))),
                      "true": min(c_pre, c_grasp), "hit": hit, "pred": {}})

    size = (rgb.shape[1], rgb.shape[0])
    extra_views = {}
    for m in margins:
        views = [(camera.K, cam_pos, cam_rot, size)] if use_view else None
        per_view, fused = [occluders], occluders
        for _ in range(max_views if use_view else 0):
            if any(grasp_clearance(cube.pos, c["psi"], fused, m, views)[0] for c in cands):
                break
            nv = select_next_view(robot, camera, q_observe, q_hat, cube.pos, cube.yaw, fused, views, m, size)
            if nv is None:
                break
            set_arm(sim, nv["q"])
            rgb2, depth2 = sim.read_camera()
            pos2, rot2 = camera.pose(robot, sim.read_joint_positions())
            per_view.append(estimate_occluders(rgb2, depth2, camera.K, pos2, rot2))
            fused = fuse_occluders(per_view)
            views.append((camera.K, pos2, rot2, size))
        extra_views[m] = len(per_view) - 1
        for c in cands:
            c["pred"][m] = grasp_clearance(cube.pos, c["psi"], fused, m, views)
    set_arm(sim, q_observe)
    return cands, occluders, extra_views


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--margins", type=float, nargs="+", default=[0.005, 0.01])
    p.add_argument("--no-view-check", action="store_true", help="do not treat unseen space as occupied")
    p.add_argument("--max-views", type=int, default=0,
                   help="active perception: up to this many extra views (src/active_vision/next_view.py) while no "
                        "yaw is clear")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    sim = TaskSim(camera_noise=CameraNoiseConfig())
    robot = UR3eModel()
    camera = CameraModel.from_calibration()
    model = sim.model
    ids = (robot_geoms(model), [model.geom(f"occluder_{i}").id for i in range(N_OCCLUDERS)])

    now = {"hit": 0, "none": 0}
    aware = {m: {"hit": 0, "none": 0, "false_clear": 0, "false_block": 0, "true_safe_exists": 0, "views": [],
              "why": collections.Counter()} for m in args.margins}
    truly_safe_seeds = 0
    for seed in range(args.seeds):
        cands, occ, extra = run_seed(sim, robot, camera, seed, args.margins, ids, not args.no_view_check, args.max_views)
        if not cands:
            now["none"] += 1
            for m in args.margins:
                aware[m]["none"] += 1
            continue
        safe_exists = any(c["true"] >= SAFE_TOL for c in cands)
        truly_safe_seeds += safe_exists
        base = min(cands, key=lambda c: c["motion"])
        now["hit"] += base["true"] < SAFE_TOL
        line = [f"seed {seed:3d}  occ est {len(occ)} ({sum(o.truncated for o in occ)} cut)  "
                f"now {base['true'] * 1e3:+6.1f} mm"]
        for m in args.margins:
            a = aware[m]
            a["views"].append(extra[m])
            a["false_clear"] += sum(c["pred"][m][0] and c["true"] < SAFE_TOL for c in cands)
            a["false_block"] += sum(not c["pred"][m][0] and c["true"] >= m + BLOCK_TOL for c in cands)
            a["why"].update(c["pred"][m][1] for c in cands if not c["pred"][m][0])
            clear = [c for c in cands if c["pred"][m][0]]
            if not clear:
                a["none"] += 1
                a["true_safe_exists"] += safe_exists
                line.append(f"m{m * 1e3:.0f}: no clear yaw{' (a safe one exists)' if safe_exists else ''}")
                continue
            pick = min(clear, key=lambda c: c["motion"])
            a["hit"] += pick["true"] < SAFE_TOL
            line.append(f"m{m * 1e3:.0f}: {pick['true'] * 1e3:+6.1f} mm")
        if args.verbose:
            print("   ".join(line))

    n = args.seeds
    print(f"\nL1, {n} seeds, noisy camera, estimated cube and occluders, up to {args.max_views} extra views; "
          "collision = true clearance < 0")
    print(f"  seeds with at least one truly collision-free yaw: {truly_safe_seeds}")
    print(f"  now (least motion):  collides {now['hit']}, no IK {now['none']}")
    for m in args.margins:
        a = aware[m]
        print(f"  aware, margin {m * 1e3:4.1f} mm: collides {a['hit']}, no clear yaw {a['none']} "
              f"(of which a safe yaw existed {a['true_safe_exists']}); per yaw: false clear {a['false_clear']}, "
              f"false block (true clearance > margin + {BLOCK_TOL * 1e3:.0f} mm) {a['false_block']}")
        print(f"      blocked yaws by reason: {dict(a['why'].most_common())}")
        if args.max_views:
            v = np.array(a["views"])
            print(f"      extra views per seed: mean {v.mean():.2f}, " +
                  ", ".join(f"{k}: {np.sum(v == k)}" for k in range(args.max_views + 1)))


if __name__ == "__main__":
    main()
