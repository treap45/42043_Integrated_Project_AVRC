"""Automatic eye-to-hand calibration: the arm collects the views by itself, then the camera pose is solved and saved.

THIS MOVES THE ROBOT, slowly (0.2 rad/s by default). Run it standing at the robot with a hand near the e-stop.

Before running:
    1. Fix the camera; it must not move afterwards. Fix the marker to the RG2 housing (not the fingers).
    2. Pendant: brakes released, status green. Jog the arm so that the camera sees the marker near the middle of the
       image, with free space around the gripper, then select Remote.
    3. Clear the workspace, check the cable slack, and keep other ID 0 markers out of view.

What it does: small moves around the start, a coarse calibration, about 25 varied views chosen with it, back to the
start pose, then scripts/calibrate_eye_to_hand.py on the session (configs/real_camera_extrinsics.json if it passes).

To stop: q or Esc in the image window, or Ctrl+C. The arm stops where it is and does not return by itself.

    python scripts/auto_handeye.py --side-mm 69.8
    python scripts/auto_handeye.py --side-mm 69.8 --step               # asks before every move; use for the first run
    python scripts/auto_handeye.py --side-mm 69.8 --ip 192.168.0.194 --table-z-mm 0
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.ur3e import UR3eModel  # noqa: E402
from src.real.auto_handeye import AutoCollector, AutoStop, ViewLimits  # noqa: E402
from src.real.robot import ROBOT_IP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WINDOW = "auto hand-eye"


def viewer(rgb, corners) -> bool:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if corners is not None:
        cv2.polylines(bgr, [corners.astype(np.int32)], True, (0, 255, 0), 2)
    else:
        cv2.putText(bgr, "marker not detected", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow(WINDOW, bgr)
    return (cv2.waitKey(1) & 0xFF) in (ord("q"), 27)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side-mm", type=float, required=True, help="measured black-square edge of the printed marker")
    p.add_argument("--ip", default=ROBOT_IP)
    p.add_argument("--views", type=int, default=25)
    p.add_argument("--table-z-mm", type=float, default=0.0, help="table surface z in the RTDE base frame")
    p.add_argument("--tool-mm", type=float, default=230.0, help="flange to beyond the fingertips")
    p.add_argument("--speed", type=float, default=0.20, help="leading joint speed (rad/s), at most 0.5")
    p.add_argument("--step", action="store_true", help="press Enter before every move")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--no-solve", action="store_true", help="only collect the views")
    args = p.parse_args()
    if not 0.0 < args.speed <= 0.5:
        sys.exit("--speed must be in (0, 0.5] rad/s")

    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    from src.real.camera import D435i

    limits = ViewLimits(table_z=args.table_z_mm / 1000.0, tool_length=args.tool_mm / 1000.0, speed=args.speed,
                        acceleration=2.0 * args.speed)
    out = ROOT / "results/handeye" / datetime.now().strftime("auto_%Y%m%d_%H%M%S")

    print(f"robot {args.ip}, marker {args.side_mm} mm, up to {args.views} views, speed {limits.speed} rad/s")
    print(f"limits: flange and fingertips {1000 * limits.clearance:.0f} mm above the table (z = {args.table_z_mm} mm), "
          f"flange within {1000 * limits.max_tcp_travel:.0f} mm of its start")
    print("The arm will move. Hand near the e-stop, workspace clear, pendant in Remote.")
    input("Press Enter to start (Ctrl+C to cancel): ")

    def confirm(label: str) -> None:
        input(f"  next: {label}. Enter to move, Ctrl+C to stop: ")

    try:
        recv = RTDEReceiveInterface(args.ip)
        ctrl = RTDEControlInterface(args.ip)
    except RuntimeError as exc:
        sys.exit(f"cannot connect to {args.ip}: {exc}\nCheck the IP, the cable, and that the pendant is in Remote.")
    collected = False
    try:
        with D435i() as cam:
            collector = AutoCollector(ctrl, recv, cam, UR3eModel(), args.side_mm / 1000.0, limits, out, args.views,
                                      confirm=confirm if args.step else None,
                                      on_frame=None if args.no_window else viewer)
            collector.run()
            collected = True
    except AutoStop as exc:
        print(f"\nSTOPPED: {exc}" + ("  (the arm stays where it is)" if exc.halt else ""))
    except KeyboardInterrupt:
        print("\nSTOPPED: Ctrl+C (the arm stays where it is)")
    finally:
        try:
            ctrl.stopJ(1.0)
        except Exception:
            pass
        cv2.destroyAllWindows()
        ctrl.stopScript()
        ctrl.disconnect()
        recv.disconnect()

    print(f"session: {out}")
    if not collected:
        print("Not solved. Views saved so far can still be solved with:")
        print(f"  python scripts/calibrate_eye_to_hand.py {out} --auto-drop")
        sys.exit(1)
    if args.no_solve:
        return
    done = subprocess.run([sys.executable, str(ROOT / "scripts/calibrate_eye_to_hand.py"), str(out), "--auto-drop"])
    sys.exit(done.returncode)


if __name__ == "__main__":
    main()
