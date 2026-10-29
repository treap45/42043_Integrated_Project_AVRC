"""Solve the fixed camera's pose T_base_cam from a capture_handeye.py session, check it, and save it.

Steps:
    1. Marker pose per view from the averaged corners (SOLVEPNP_IPPE_SQUARE); gripper pose from the RTDE TCP pose.
    2. Closed-form hand-eye with the robot poses inverted (src/real/handeye.py), all five OpenCV methods.
    3. Joint refinement of T_base_cam and T_ee_marker on the corner reprojection error, from the Park solution.
    4. Checks: per-view residuals, outlier views, bootstrap spread of T_base_cam, the depth image against the
       predicted marker distance, and FK of the controller-side model against the RTDE pose (frame convention).

Writes configs/real_camera_extrinsics.json (only if the result passes, or with --force).

    python scripts/calibrate_eye_to_hand.py results/handeye/20261028_101500
    python scripts/calibrate_eye_to_hand.py results/handeye/20261028_101500 --drop 3 7
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.ur3e import UR3eModel  # noqa: E402
from src.real.aruco import marker_object_points  # noqa: E402
from src.real.frames import (invert_T, make_T, rotation_angle_deg, rtde_pose_to_T,  # noqa: E402
                             sim_world_from_rtde)
from src.real.handeye import METHODS, evaluate, refine, rotation_spread_deg, solve_T_base_cam  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "configs/real_camera_extrinsics.json"
MIN_VIEWS = 10
MIN_SPREAD_DEG = 30.0
PASS_REPROJ_PX = 2.0      # mean corner reprojection after refinement
PASS_BOOT_MM = 3.0        # bootstrap spread (1 sigma) of the camera position
OUTLIER_FACTOR = 3.0      # a view is flagged when its reprojection error is this many times the median
BOOTSTRAP = 50


def fk_check(views) -> str:
    """Compare the RTDE pose with FK of the controller-side model, both in the simulation world frame."""
    robot = UR3eModel()
    T_w_b = sim_world_from_rtde()
    d_pos, R_off = [], []
    for v in views:
        p, R = robot.site_pose(np.asarray(v["q"]), "attachment_site")
        T_rtde = T_w_b @ rtde_pose_to_T(v["tcp_pose"])
        d_pos.append(1000 * np.linalg.norm(T_rtde[:3, 3] - p))
        R_off.append(R.T @ T_rtde[:3, :3])
    spread = max(rotation_angle_deg(R_off[0].T @ R) for R in R_off)
    return (f"RTDE TCP vs FK flange (attachment_site): position {np.mean(d_pos):.1f} mm mean, "
            f"{np.max(d_pos):.1f} mm max; constant rotation offset {rotation_angle_deg(R_off[0]):.1f} deg, "
            f"varies by {spread:.2f} deg across views (near 0 means the frames match up to that offset)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path)
    p.add_argument("--drop", type=int, nargs="*", default=[], help="view indices to leave out")
    p.add_argument("--force", action="store_true", help="save even if the checks fail")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    session = json.loads((args.session / "session.json").read_text())
    views = [v for v in json.loads((args.session / "views.json").read_text()) if v["index"] not in args.drop]
    views = [v for v in views if "tcp_pose" in v]
    K, dist = np.asarray(session["K"]), np.asarray(session["dist"])
    side = session["marker_side_m"]
    obj = marker_object_points(side)
    print(f"session {args.session.name}: {len(views)} views with robot poses, marker {side * 1000:.1f} mm"
          + (f", dropped {args.drop}" if args.drop else ""))
    if len(views) < 3:
        sys.exit("need at least 3 views")

    T_base_ee = [rtde_pose_to_T(v["tcp_pose"]) for v in views]
    corners = [np.asarray(v["corners_px"]) for v in views]
    T_cam_marker = []
    for c in corners:
        ok, rvec, tvec = cv2.solvePnP(obj, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        T_cam_marker.append(make_T(cv2.Rodrigues(rvec)[0], tvec))
    spread = rotation_spread_deg(T_base_ee)
    print(f"gripper rotation spread {spread:.0f} deg; tilt to camera {min(v['tilt_deg'] for v in views):.0f}-"
          f"{max(v['tilt_deg'] for v in views):.0f} deg; distance "
          f"{min(np.linalg.norm(T[:3, 3]) for T in T_cam_marker):.2f}-"
          f"{max(np.linalg.norm(T[:3, 3]) for T in T_cam_marker):.2f} m")

    print("\nclosed form (camera position in the RTDE base frame):")
    for m in METHODS:
        T = solve_T_base_cam(T_base_ee, T_cam_marker, m)
        r = evaluate(T, T_base_ee, T_cam_marker, corners, obj, K, dist)
        print(f"  {m:10s} cam at {np.round(T[:3, 3], 4)}  reprojection {r.reproj_px.mean():6.2f} px mean")

    T0 = solve_T_base_cam(T_base_ee, T_cam_marker, "Park")
    T_em0 = evaluate(T0, T_base_ee, T_cam_marker, corners, obj, K, dist).T_ee_marker
    T_bc, T_em = refine(T0, T_em0, T_base_ee, corners, obj, K, dist)
    res = evaluate(T_bc, T_base_ee, T_cam_marker, corners, obj, K, dist)
    # evaluate() refits T_ee_marker by averaging; report residuals with the refined one instead.
    T_cb = invert_T(T_bc)
    reproj, depth_err = [], []
    for v, Tb, c in zip(views, T_base_ee, corners):
        T_cm = T_cb @ Tb @ T_em
        proj, _ = cv2.projectPoints(obj, cv2.Rodrigues(T_cm[:3, :3])[0], T_cm[:3, 3], K, dist)
        reproj.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - c) ** 2, axis=1)))))
        depth_err.append(1000 * (v["marker_depth_m"] - T_cm[2, 3]) if np.isfinite(v["marker_depth_m"]) else np.nan)
    reproj, depth_err = np.array(reproj), np.array(depth_err)

    print(f"\nrefined: camera at {np.round(T_bc[:3, 3], 4)} m (RTDE base), "
          f"{rotation_angle_deg(T_bc[:3, :3] @ T0[:3, :3].T):.2f} deg / "
          f"{1000 * np.linalg.norm(T_bc[:3, 3] - T0[:3, 3]):.1f} mm from Park")
    print(f"  marker on the flange at {np.round(1000 * T_em[:3, 3], 1)} mm")
    print(f"  reprojection {reproj.mean():.2f} px mean, {reproj.max():.2f} px max")
    print(f"  depth image minus predicted marker z: {np.nanmean(depth_err):+.1f} mm mean, "
          f"{np.nanstd(depth_err):.1f} mm std")
    print("  per view (index: reprojection px, depth error mm):")
    med = np.median(reproj)
    flagged = []
    for v, r, d in zip(views, reproj, depth_err):
        mark = ""
        if r > OUTLIER_FACTOR * med and r > 1.0:
            mark = "  <- outlier"
            flagged.append(v["index"])
        print(f"    {v['index']:3d}: {r:5.2f}  {d:+6.1f}{mark}")

    rng = np.random.default_rng(0)
    boot = []
    for _ in range(BOOTSTRAP):
        idx = rng.choice(len(views), len(views), replace=True)
        if len(set(idx)) < 3:
            continue
        Tb_b, c_b = [T_base_ee[i] for i in idx], [corners[i] for i in idx]
        T_b, _ = refine(T_bc, T_em, Tb_b, c_b, obj, K, dist)
        boot.append(T_b)
    boot_mm = 1000 * np.std([T[:3, 3] for T in boot], axis=0)
    boot_deg = np.std([rotation_angle_deg(T[:3, :3] @ T_bc[:3, :3].T) for T in boot])
    print(f"  bootstrap ({len(boot)}): camera position std {np.round(boot_mm, 2)} mm, rotation std {boot_deg:.3f} deg")
    if all("q" in v for v in views):
        print("  " + fk_check(views))

    checks = {
        f"at least {MIN_VIEWS} views": len(views) >= MIN_VIEWS,
        f"rotation spread >= {MIN_SPREAD_DEG:.0f} deg": spread >= MIN_SPREAD_DEG,
        f"mean reprojection < {PASS_REPROJ_PX} px": reproj.mean() < PASS_REPROJ_PX,
        f"bootstrap position std < {PASS_BOOT_MM} mm": np.linalg.norm(boot_mm) < PASS_BOOT_MM,
    }
    print("\nchecks:")
    for name, ok in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if flagged:
        print(f"  outlier views {flagged}: rerun with --drop {' '.join(map(str, flagged))} if they look wrong")
    passed = all(checks.values())

    if passed or args.force:
        T_wc = sim_world_from_rtde() @ T_bc
        args.out.write_text(json.dumps({
            "date": datetime.now().isoformat(timespec="seconds"),
            "session": str(args.session),
            "views_used": [v["index"] for v in views],
            "camera_serial": session["camera_serial"],
            "frame_note": "T_base_cam: OpenCV camera frame in the RTDE base frame. T_world_cam: the same in the "
                          "simulation world (RTDE base = sim base rotated 180 deg about z, 0.75 m up).",
            "K": session["K"], "dist": session["dist"],
            "T_base_cam": T_bc.tolist(),
            "T_world_cam": T_wc.tolist(),
            "T_ee_marker": T_em.tolist(),
            "reprojection_px_mean": float(reproj.mean()),
            "bootstrap_position_std_mm": boot_mm.tolist(),
            "bootstrap_rotation_std_deg": float(boot_deg),
            "depth_error_mm_mean": float(np.nanmean(depth_err)),
            "checks_passed": passed,
        }, indent=2))
        print(f"\nwrote {args.out}" + ("" if passed else " (checks failed, saved with --force)"))
    else:
        print("\nnot saved: fix the failing checks (more or more varied views) or pass --force")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
