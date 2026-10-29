"""Check the eye-to-hand solver (src/real/handeye.py) on synthetic views with known answers.

A camera is placed about 0.8 m from the robot looking at the workspace, a marker is fixed to the flange with an
offset, and random flange poses are drawn so the marker stays in view and tilted toward the camera. Corners are
projected with the lab D435i intrinsics plus pixel noise, the marker pose is recovered with SOLVEPNP_IPPE_SQUARE as
on the real data, and T_base_cam is solved and compared with the truth. No hardware is used.

    python scripts/check_eye_to_hand.py
    python scripts/check_eye_to_hand.py --views 25 --noise-px 0.5 --side-mm 80
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.real.aruco import marker_object_points  # noqa: E402
from src.real.frames import invert_T, make_T, rotation_angle_deg  # noqa: E402
from src.real.handeye import METHODS, evaluate, refine, rotation_spread_deg, solve_T_base_cam  # noqa: E402

INTRINSICS = Path(__file__).resolve().parents[1] / "configs/real_camera_intrinsics.json"
PASS_POS_MM = 3.0
PASS_ROT_DEG = 0.5


def rot(axis: str, deg: float) -> np.ndarray:
    R, _ = cv2.Rodrigues({"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}[axis]
                         * np.radians(deg))
    return R


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """R_base_cam of an OpenCV camera (z forward, y down) at eye looking at target, roughly upright."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def synthetic_views(rng, T_base_cam, T_ee_marker, K, side, n, noise_px, robot_noise_mm):
    obj = marker_object_points(side)
    cam_pos = T_base_cam[:3, 3]
    T_base_ee, corners = [], []
    for _ in range(200000):
        if len(T_base_ee) == n:
            break
        p = rng.uniform([-0.15, -0.45, 0.10], [0.15, -0.20, 0.40])
        # Marker normal toward the camera, then tilted 20-45 degrees about a random axis and spun about the normal.
        R_face = look_at(p, cam_pos)  # marker z (out of the paper) points at the camera
        tilt = rng.uniform(20.0, 45.0) * rng.choice([-1, 1])
        R_base_marker = R_face @ rot(rng.choice(["x", "y"]), tilt) @ rot("z", rng.uniform(-60, 60))
        T_bm = make_T(R_base_marker, p)
        T_be = T_bm @ invert_T(T_ee_marker)
        T_cm = invert_T(T_base_cam) @ T_bm
        if T_cm[2, 3] < 0.2 or (T_cm[:3, :3] @ [0, 0, 1.0])[2] > -0.3:  # in front, facing the camera
            continue
        rvec, _ = cv2.Rodrigues(T_cm[:3, :3])
        px, _ = cv2.projectPoints(obj, rvec, T_cm[:3, 3], K, np.zeros(5))
        px = px.reshape(4, 2)
        if np.any(px < 0) or np.any(px[:, 0] > 640) or np.any(px[:, 1] > 480):
            continue
        noisy_T_be = T_be.copy()
        noisy_T_be[:3, 3] += rng.normal(0.0, robot_noise_mm / 1000.0, 3)
        T_base_ee.append(noisy_T_be)
        corners.append(px + rng.normal(0.0, noise_px, px.shape))
    if len(T_base_ee) < n:
        raise RuntimeError(f"only {len(T_base_ee)} of {n} views found")
    return T_base_ee, corners


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--views", type=int, default=20)
    p.add_argument("--noise-px", type=float, default=0.3)
    p.add_argument("--robot-noise-mm", type=float, default=0.05, help="UR3e repeatability is about 0.03 mm")
    p.add_argument("--side-mm", type=float, default=80.0)
    p.add_argument("--trials", type=int, default=20)
    args = p.parse_args()

    K = np.asarray(json.loads(INTRINSICS.read_text())["K"])
    side = args.side_mm / 1000.0
    obj = marker_object_points(side)
    errs = {m: [] for m in [*METHODS, "Park + refine"]}
    for trial in range(args.trials):
        rng = np.random.default_rng(trial)
        eye = np.array([rng.uniform(-0.3, 0.3), -0.95, rng.uniform(0.45, 0.7)])
        T_base_cam = make_T(look_at(eye, np.array([0.0, -0.3, 0.15])), eye)
        T_ee_marker = make_T(rot("x", rng.uniform(-20, 20)) @ rot("y", rng.uniform(-20, 20)),
                             rng.uniform([-0.03, -0.03, 0.08], [0.03, 0.03, 0.15]))
        T_base_ee, corners = synthetic_views(rng, T_base_cam, T_ee_marker, K, side, args.views, args.noise_px,
                                             args.robot_noise_mm)
        T_cam_marker = []
        for c in corners:
            ok, rvec, tvec = cv2.solvePnP(obj, c, K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
            T_cam_marker.append(make_T(cv2.Rodrigues(rvec)[0], tvec))
        for m in METHODS:
            T_est = solve_T_base_cam(T_base_ee, T_cam_marker, m)
            errs[m].append((1000 * np.linalg.norm(T_est[:3, 3] - T_base_cam[:3, 3]),
                            rotation_angle_deg(T_est[:3, :3].T @ T_base_cam[:3, :3])))
        T0 = solve_T_base_cam(T_base_ee, T_cam_marker)
        T_ref, T_em_ref = refine(T0, evaluate(T0, T_base_ee, T_cam_marker, corners, obj, K, np.zeros(5)).T_ee_marker,
                                 T_base_ee, corners, obj, K, np.zeros(5))
        errs["Park + refine"].append((1000 * np.linalg.norm(T_ref[:3, 3] - T_base_cam[:3, 3]),
                                      rotation_angle_deg(T_ref[:3, :3].T @ T_base_cam[:3, :3])))
        if trial == 0:
            res = evaluate(solve_T_base_cam(T_base_ee, T_cam_marker), T_base_ee, T_cam_marker, corners, obj, K,
                           np.zeros(5))
            print(f"trial 0: {args.views} views, rotation spread {rotation_spread_deg(T_base_ee):.0f} deg, "
                  f"camera distance {np.linalg.norm(eye - [0, -0.3, 0.15]):.2f} m")
            print(f"  Park residuals: position {res.pos_err_mm.mean():.2f} mm mean, rotation "
                  f"{res.rot_err_deg.mean():.2f} deg mean, reprojection {res.reproj_px.mean():.2f} px mean")
            print(f"  T_ee_marker error {1000 * np.linalg.norm(res.T_ee_marker[:3, 3] - T_ee_marker[:3, 3]):.2f} mm")

    print(f"\n{args.trials} trials, {args.views} views, {args.side_mm:.0f} mm marker, {args.noise_px} px noise: "
          f"T_base_cam error, median / max")
    ok = True
    for m, e in errs.items():
        e = np.array(e)
        print(f"  {m:10s} position {np.median(e[:, 0]):6.2f} / {e[:, 0].max():6.2f} mm   "
              f"rotation {np.median(e[:, 1]):5.3f} / {e[:, 1].max():5.3f} deg")
        if m == "Park + refine":
            ok = e[:, 0].max() < PASS_POS_MM and e[:, 1].max() < PASS_ROT_DEG
    print(f"\n{'PASS' if ok else 'FAIL'} (Park + refine: max position < {PASS_POS_MM} mm, max rotation < {PASS_ROT_DEG} deg)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
