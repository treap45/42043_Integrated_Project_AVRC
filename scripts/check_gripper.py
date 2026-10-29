"""Check the RG2 through src/real/gripper.py: read the width, and with --move open, close and open again.

Run at the robot with the fingers clear. Pendant: Installation -> Tool I/O set to OnRobot. Nothing here moves the arm.

    python scripts/check_gripper.py                  # read only
    python scripts/check_gripper.py --move           # open 80 mm, close 20 mm, open 80 mm
    python scripts/check_gripper.py --move --object-mm 60   # put the cube between the fingers first
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.real.gripper import DEFAULT_FORCE_N, RG2  # noqa: E402
from src.real.robot import ROBOT_IP  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ip", default=ROBOT_IP)
    p.add_argument("--move", action="store_true")
    p.add_argument("--object-mm", type=float, default=None, help="close on an object of this width instead")
    p.add_argument("--force", type=float, default=DEFAULT_FORCE_N)
    args = p.parse_args()

    rg = RG2(args.ip)
    print(f"RG2 at {args.ip}: width {rg.width():.1f} mm")
    if not args.move:
        return
    for label, fn in [("open 80", lambda: rg.open(80.0)),
                      ("close", (lambda: rg.close_on(args.object_mm, force_n=args.force)) if args.object_mm
                       else (lambda: rg.grip(20.0, args.force))),
                      ("open 80", lambda: rg.open(80.0))]:
        t = time.monotonic()
        out = fn()
        w, held = out if isinstance(out, tuple) else (out, None)
        print(f"  {label:8s}: {w:5.1f} mm after {time.monotonic() - t:.2f} s" + ("" if held is None else f", held={held}"))


if __name__ == "__main__":
    main()
