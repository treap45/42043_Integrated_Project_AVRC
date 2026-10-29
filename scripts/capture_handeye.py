"""Collect eye-to-hand calibration views: the fixed D435i sees an ArUco marker on the gripper, RTDE gives the pose.

Read-only on the robot: only RTDE receive is opened, so nothing here can move the arm. Jog the arm on the pendant
(Local mode is fine), let it stop, and press Enter in the image window. Each saved view averages the marker corners
over several frames and stores the image, the depth, the joint angles and the RTDE TCP pose.

The overlay shows, for the current frame:
    tilt   angle between the marker normal and the line of sight; below about 20 deg the pose can flip
    new    rotation to the closest view already saved; aim for views that differ by 15 deg or more
    dist   marker distance from the camera, side_px its size in the image

Keys: Enter or space save, q or Esc finish.

    python scripts/capture_handeye.py --side-mm 80.3
    python scripts/capture_handeye.py --side-mm 80.3 --no-robot     # camera and detection only
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.real.aruco import DICTIONARY, MARKER_ID, detect, make_detector, marker_depth, marker_pose  # noqa: E402
from src.real.camera import D435i  # noqa: E402
from src.real.frames import rotation_angle_deg  # noqa: E402
from src.real.robot import ROBOT_IP, URReader  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parents[1] / "results/handeye"
AVERAGE_FRAMES = 10
STILL_QD = 1e-3  # rad/s; above this on any joint the arm is still moving
MIN_TILT_DEG = 20.0
MIN_NEW_DEG = 15.0
WINDOW = "eye-to-hand capture"


def tilt_deg(obs) -> float:
    """Angle between the marker normal and the ray from the camera to the marker centre."""
    normal = obs.R_cam_marker[:, 2]
    ray = obs.t_cam_marker / np.linalg.norm(obs.t_cam_marker)
    return float(np.degrees(np.arccos(np.clip(abs(normal @ ray), 0.0, 1.0))))


def draw(bgr, obs, K, dist, side, lines, colour):
    if obs is not None:
        cv2.polylines(bgr, [obs.corners.astype(np.int32)], True, (0, 255, 0), 2)
        rvec, _ = cv2.Rodrigues(obs.R_cam_marker)
        cv2.drawFrameAxes(bgr, K, dist, rvec, obs.t_cam_marker, side * 0.75, 2)
    for i, (text, c) in enumerate(zip(lines, colour)):
        cv2.putText(bgr, text, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(bgr, text, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side-mm", type=float, required=True, help="measured black-square edge of the printed marker")
    p.add_argument("--ip", default=ROBOT_IP)
    p.add_argument("--no-robot", action="store_true", help="no RTDE connection; saves views without poses")
    p.add_argument("--out", type=Path, default=None, help="session folder (default results/handeye/<time>)")
    args = p.parse_args()

    side = args.side_mm / 1000.0
    out = args.out or OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    detector = make_detector()
    robot = None if args.no_robot else URReader(args.ip)
    views, saved_R = [], []
    try:
        with D435i() as cam:
            session = {"date": datetime.now().isoformat(timespec="seconds"), "camera_serial": cam.serial,
                       "K": cam.K.tolist(), "dist": cam.dist.tolist(), "depth_scale_m": cam.depth_scale,
                       "aruco_dictionary": "DICT_4X4_50", "aruco_dictionary_id": int(DICTIONARY),
                       "marker_id": MARKER_ID, "marker_side_m": side, "robot_ip": None if robot is None else args.ip,
                       "pose_source": "RTDE getActualTCPPose (pendant TCP, RTDE base frame)"}
            (out / "session.json").write_text(json.dumps(session, indent=2))
            print(f"saving to {out}\nEnter/space: save view   q/Esc: finish")
            cv2.namedWindow(WINDOW)
            while True:
                frame = cam.capture()
                corners = detect(detector, frame.rgb)
                obs = None if corners is None else marker_pose(corners, side, cam.K, cam.dist)
                qd = None if robot is None else robot.qd()
                moving = qd is not None and np.max(np.abs(qd)) > STILL_QD
                lines, colours = [f"saved {len(views)}"], [(255, 255, 255)]
                if obs is None:
                    lines.append("marker not detected")
                    colours.append((0, 0, 255))
                else:
                    tilt = tilt_deg(obs)
                    new = min((rotation_angle_deg(R.T @ obs.R_cam_marker) for R in saved_R), default=180.0)
                    lines += [f"dist {np.linalg.norm(obs.t_cam_marker):.3f} m  side {obs.side_px:.0f} px  "
                              f"reproj {obs.reproj_px:.2f} px",
                              f"tilt {tilt:.0f} deg" + ("  (too frontal)" if tilt < MIN_TILT_DEG else ""),
                              f"new {new:.0f} deg" + ("  (close to a saved view)" if new < MIN_NEW_DEG else "")]
                    colours += [(255, 255, 255), (0, 165, 255) if tilt < MIN_TILT_DEG else (0, 255, 0),
                                (0, 165, 255) if new < MIN_NEW_DEG else (0, 255, 0)]
                if moving:
                    lines.append("robot moving")
                    colours.append((0, 0, 255))
                bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                draw(bgr, obs, cam.K, cam.dist, side, lines, colours)
                cv2.imshow(WINDOW, bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key not in (13, 10, ord(" ")):
                    continue
                if obs is None or moving:
                    print("  not saved: " + ("marker not detected" if obs is None else "robot moving"))
                    continue

                # Average the corners over a short burst; the arm is still, so this only removes pixel noise.
                stack = []
                for _ in range(AVERAGE_FRAMES):
                    f = cam.capture()
                    c = detect(detector, f.rgb)
                    if c is not None:
                        stack.append(c)
                if len(stack) < AVERAGE_FRAMES // 2:
                    print("  not saved: marker lost during averaging")
                    continue
                mean_corners = np.mean(stack, axis=0)
                obs = marker_pose(mean_corners, side, cam.K, cam.dist)
                i = len(views)
                cv2.imwrite(str(out / f"{i:03d}.png"), cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR))
                np.save(out / f"{i:03d}_depth.npy", f.depth_m)
                view = {"index": i, "time": time.time(), "corners_px": mean_corners.tolist(),
                        "corner_std_px": float(np.std(stack, axis=0).mean()), "frames_averaged": len(stack),
                        "marker_depth_m": marker_depth(f.depth_m, mean_corners),
                        "pnp_t_cam_marker": obs.t_cam_marker.tolist(), "pnp_reproj_px": obs.reproj_px,
                        "tilt_deg": tilt_deg(obs)}
                if robot is not None:
                    view["q"] = robot.q().tolist()
                    view["tcp_pose"] = robot.tcp_pose().tolist()
                views.append(view)
                saved_R.append(obs.R_cam_marker)
                (out / "views.json").write_text(json.dumps(views, indent=1))
                print(f"  view {i}: dist {np.linalg.norm(obs.t_cam_marker):.3f} m, tilt {view['tilt_deg']:.0f} deg, "
                      f"corner std {view['corner_std_px']:.2f} px, depth {view['marker_depth_m']:.3f} m "
                      f"vs PnP z {obs.t_cam_marker[2]:.3f} m")
    finally:
        cv2.destroyAllWindows()
        if robot is not None:
            robot.close()
    print(f"{len(views)} views in {out}")
    if len(views) and not args.no_robot:
        print(f"next: python scripts/calibrate_eye_to_hand.py {out}")


if __name__ == "__main__":
    main()
