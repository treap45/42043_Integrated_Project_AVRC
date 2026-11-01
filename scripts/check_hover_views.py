"""Check the candidate hover views on the occluded levels: collision and cube visibility per grasp yaw.

For a top-down grasp the wrist camera is tied to the gripper, so the grasp yaw (four equivalent ones for a square
cube) and the hover height decide where the camera is while the arm is above the cube. For every seed and every
candidate (yaw, hover height) this script puts the arm at the pregrasp pose (IK on the true cube pose, kinematics
only, no physics) and measures:
  - clearance between the occluders and the arm, gripper and camera, at the hover pose and at the grasp pose;
  - the visible fraction of the cube in the wrist image: cube pixels with the occluders in place over cube pixels
    with them parked (segmentation render), so the gripper's own cover counts in both.
It then compares the candidate the pipeline picks now (least joint motion from the observe pose, hover 0.10 m,
as in PickPlace._select_grasp) with the best candidate, to show how much room a view-selection step has.

This is an evaluation script: it reads simulator ground truth freely. Nothing here goes into the pipeline.

Run from the repo root with the venv active:
    python scripts/check_hover_views.py                  # L1, seeds 0-49
    python scripts/check_hover_views.py --seeds 30 --heights 0.10 0.15 0.20
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.ur3e import UR3eModel  # noqa: E402
from src.planning.grasp import GRASP_DEPTH, top_down_rotation, yaw_candidates  # noqa: E402
from src.planning.ik import solve_ik  # noqa: E402
from src.sim import TaskSim  # noqa: E402
from src.sim.randomize import N_OCCLUDERS, PARK_POS  # noqa: E402

N_ARM = 6
CLEAR_TOL = 0.01  # m, minimum occluder clearance counted as safe
VIS_OK = 0.9  # visible fraction counted as a full view
DIST_MAX = 0.10  # m, clearances above this are reported as this


def robot_geoms(model: mujoco.MjModel) -> list[int]:
    """Collision geoms of the arm, gripper and camera (every body under the robot base)."""
    root = model.body("base").id
    return [g for g in range(model.ngeom)
            if model.body_rootid[model.geom_bodyid[g]] == root and model.geom_contype[g] | model.geom_conaffinity[g]]


def clearance(model, data, robot_ids, occ_ids) -> tuple[float, str]:
    """Smallest occluder-to-robot distance (negative = penetration), capped at DIST_MAX, and the robot body."""
    best, body = DIST_MAX, ""
    fromto = np.zeros(6)
    for o in occ_ids:
        for g in robot_ids:
            d = mujoco.mj_geomDistance(model, data, o, g, DIST_MAX, fromto)
            if d < best:
                best, body = d, model.body(model.geom_bodyid[g]).name
    return best, body


def cube_pixels(renderer, data, cube_geom: int) -> int:
    renderer.update_scene(data, "wrist_cam")
    seg = renderer.render()
    return int(np.count_nonzero((seg[..., 0] == cube_geom) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)))


def visible_fraction(sim, renderer, cube_geom, occ_mocap) -> tuple[float, int]:
    data = sim.data
    n_with = cube_pixels(renderer, data, cube_geom)
    saved = data.mocap_pos[occ_mocap].copy()
    data.mocap_pos[occ_mocap] = PARK_POS
    mujoco.mj_forward(sim.model, data)
    n_free = cube_pixels(renderer, data, cube_geom)
    data.mocap_pos[occ_mocap] = saved
    mujoco.mj_forward(sim.model, data)
    return (n_with / n_free if n_free else 0.0), n_free


def set_arm(sim, q) -> None:
    sim.data.qpos[:N_ARM] = q
    mujoco.mj_forward(sim.model, sim.data)


def run_seed(sim, robot, renderer, seed, level, heights, ids) -> dict:
    sim.reset(seed, level)
    robot_ids, occ_ids, occ_mocap, cube_geom = ids
    n_occ = len(sim.trial.occluders)
    q_observe = sim.data.qpos[:N_ARM].copy()
    gt = sim.ground_truth()
    cube_pos = np.asarray(gt["cube_pos"], dtype=float)
    cube_yaw = float(sim.trial.cube_yaw)
    occ_ids, occ_mocap = occ_ids[:n_occ], occ_mocap[:n_occ]

    candidates = []
    for k, psi in enumerate(yaw_candidates(cube_yaw)):
        R = top_down_rotation(psi)
        p_grasp = cube_pos + np.array([0.0, 0.0, GRASP_DEPTH])
        q_grasp, ok_grasp = solve_ik(robot, p_grasp, R, q_observe)
        for h in heights:
            q_pre, ok = solve_ik(robot, p_grasp + np.array([0.0, 0.0, h]), R, q_observe)
            c = {"k": k, "psi": psi, "h": h, "ik": bool(ok and ok_grasp),
                 "motion": float(np.sum(np.abs(q_pre - q_observe)))}
            if c["ik"]:
                set_arm(sim, q_pre)
                c["clear_hover"], c["hit_hover"] = clearance(sim.model, sim.data, robot_ids, occ_ids)
                c["vis"], c["cube_px"] = visible_fraction(sim, renderer, cube_geom, occ_mocap)
                set_arm(sim, q_grasp)
                c["clear_grasp"], c["hit_grasp"] = clearance(sim.model, sim.data, robot_ids, occ_ids)
                c["safe"] = min(c["clear_hover"], c["clear_grasp"]) >= CLEAR_TOL
            candidates.append(c)
    set_arm(sim, q_observe)

    feasible = [c for c in candidates if c["ik"]]
    # The pipeline today: least joint motion among the IK-feasible yaws at the default hover height.
    base_pool = [c for c in feasible if np.isclose(c["h"], heights[0])]
    baseline = min(base_pool, key=lambda c: c["motion"]) if base_pool else None
    safe = [c for c in feasible if c["safe"]]
    best = max(safe, key=lambda c: (c["vis"], -c["h"], -c["motion"])) if safe else None
    return {"seed": seed, "n_occ": n_occ, "candidates": candidates, "baseline": baseline, "best": best}


def fmt(c) -> str:
    if c is None:
        return "none"
    return (f"k{c['k']} h{c['h']:.2f} vis {c['vis']:.2f} clr {c['clear_hover'] * 1e3:5.1f}/"
            f"{c['clear_grasp'] * 1e3:5.1f} mm{' ' + c['hit_hover'] if c['clear_hover'] < CLEAR_TOL else ''}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", default="L1", choices=["L0", "L1"])
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--heights", type=float, nargs="+", default=[0.10, 0.15],
                   help="hover heights above the grasp pose (m); the first is the pipeline's HOVER")
    p.add_argument("-v", "--verbose", action="store_true", help="print every candidate")
    args = p.parse_args()

    sim = TaskSim()
    robot = UR3eModel()
    model = sim.model
    occ_ids = [model.geom(f"occluder_{i}").id for i in range(N_OCCLUDERS)]
    occ_mocap = [model.body(f"occluder_{i}").mocapid[0] for i in range(N_OCCLUDERS)]
    ids = (robot_geoms(model), occ_ids, occ_mocap, model.geom("cube").id)
    w, h = model.cam_resolution[model.camera("wrist_cam").id]
    renderer = mujoco.Renderer(model, height=h, width=w)
    renderer.enable_segmentation_rendering()

    rows = []
    for seed in range(args.seeds):
        r = run_seed(sim, robot, renderer, seed, args.level, args.heights, ids)
        rows.append(r)
        print(f"seed {seed:3d}  occ {r['n_occ']}  now: {fmt(r['baseline'])}   best: {fmt(r['best'])}")
        if args.verbose:
            for c in r["candidates"]:
                print(f"      {fmt(c) if c['ik'] else 'k%d h%.2f IK failed' % (c['k'], c['h'])}"
                      f"  motion {c['motion']:.2f} rad")
    renderer.close()

    base = [r["baseline"] for r in rows if r["baseline"] is not None]
    best = [r["best"] for r in rows]
    n = len(rows)
    print(f"\n{args.level}, {n} seeds, hover heights {args.heights} m (clearance >= {CLEAR_TOL * 1e3:.0f} mm = safe)")
    print(f"  pipeline choice: no IK {n - len(base)}, "
          f"collides at hover {sum(c['clear_hover'] < CLEAR_TOL for c in base)}, "
          f"at grasp {sum(c['clear_grasp'] < CLEAR_TOL for c in base)}, "
          f"unsafe in all {sum(not c['safe'] for c in base)} "
          f"(penetrating {sum(min(c['clear_hover'], c['clear_grasp']) < 0 for c in base)}); "
          f"visible < {VIS_OK} in {sum(c['vis'] < VIS_OK for c in base)}, "
          f"median visible {np.median([c['vis'] for c in base]):.2f}, min {min(c['vis'] for c in base):.2f}")
    got = [c for c in best if c is not None]
    print(f"  best safe candidate: none in {n - len(got)}; visible < {VIS_OK} in "
          f"{sum(c['vis'] < VIS_OK for c in got)}, median visible {np.median([c['vis'] for c in got]):.2f}; "
          f"higher hover chosen in {sum(not np.isclose(c['h'], args.heights[0]) for c in got)}")
    better = sum(1 for r in rows if r["baseline"] is not None and r["best"] is not None
                 and (not r["baseline"]["safe"] or r["best"]["vis"] > r["baseline"]["vis"] + 0.05))
    print(f"  seeds where a view choice would help (pipeline unsafe, or > 5 pt more visible): {better}/{n}")


if __name__ == "__main__":
    main()
