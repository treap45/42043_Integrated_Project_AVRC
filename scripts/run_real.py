"""Run the moving-cube task on the lab UR3e + D435i + RG2, or the same code against the simulation.

The pipeline is the simulation's (HSV cube detection, cube EKF, TRACK with PBVS, APPROACH, grasp and place), run
through src/real/: a fixed camera from the eye-to-hand calibration, joint-velocity commands (RTDE speedJ) instead
of torques, the RG2 over XML-RPC, and safety limits on every command (src/real/io.py).

Order of use on the robot (pendant in Remote, arm started where it does not hide the table from the camera):
    python scripts/tune_hsv.py                               # cube colour thresholds, saved for the runs below
    python scripts/run_real.py --sim                         # first, in simulation (fixed scene camera)
    python scripts/run_real.py --dry-run --ip 192.168.0.191  # camera + cube + EKF live, nothing is sent
    python scripts/run_real.py --mode track --hover-mm 150 --tcp-mm <measured> --table-z-mm <measured>
    python scripts/run_real.py --mode pick --place-xy -0.30 -0.12 --tcp-mm <measured> --table-z-mm <measured>

Any run that moves the robot needs --tcp-mm and --table-z-mm: the safety limits rely on them, so a wrong value would
not be caught. A grasp that closes on nothing ends the run with the arm stopped where it is.

Coordinates on the command line (--place-xy, --table-z-mm) are in the RTDE base frame, as the pendant shows them.
In the image window, q or Esc ends the run (the arm is stopped); Ctrl+C does the same from the terminal.
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
from src.model.ur3e import UR3eModel  # noqa: E402
from src.perception.segmentation import color_mask, largest_component  # noqa: E402
from src.real import config  # noqa: E402
from src.real.fixed_camera import FixedCameraModel  # noqa: E402
from src.real.frames import SIM_BASE_HEIGHT, sim_world_from_rtde  # noqa: E402
from src.real.io import DEPTH_FRAMES, QD_MAX, SIM_DEPTH_K, RealIO, SafetyLimits, SafetyStop, SimIO  # noqa: E402
from src.real.pick_place_real import RealPickPlace  # noqa: E402
from src.real.robot import ROBOT_IP  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parents[1] / "results/real_runs"
VIEW_PERIOD = 50  # control periods between display updates (10 Hz)
TCP_RANGE_MM = (120.0, 300.0)  # a flange-to-fingertip value outside this is a typo or the wrong unit


def rtde_xy_to_world(xy) -> np.ndarray:
    p = sim_world_from_rtde() @ np.array([xy[0], xy[1], 0.0, 1.0])
    return p[:2]


class Viewer:
    """Camera image with the cube mask outline, the state and the EKF; q / Esc asks the run to stop."""

    def __init__(self, show: bool):
        self.show = show
        self.quit = False
        self.k = 0
        self.last_state = None
        self.log = []

    def __call__(self, pp: RealPickPlace) -> None:
        self.k += 1
        if pp.state != self.last_state:
            extra = ""
            if pp.state == "TRACK" and pp.cube_estimate is not None:
                c = pp.cube_estimate
                extra = (f"  cube at ({c.pos[0]:.3f}, {c.pos[1]:.3f}, {c.pos[2]:.3f}) m world, top "
                         f"{1000 * c.top_size[0]:.0f} x {1000 * c.top_size[1]:.0f} mm")
            print(f"[{pp.elapsed:6.2f} s] {pp.state}{extra}")
            self.last_state = pp.state
        if self.k % VIEW_PERIOD:
            return
        ekf = pp.cube_ekf
        self.log.append({"t": pp.elapsed, "state": pp.state, "q": pp.ctrl.xhat[:, 0].tolist(),
                         "qd_cmd": pp.io.last_cmd.tolist(),
                         "ekf": None if ekf is None else [*ekf.position, ekf.yaw, *ekf.velocity]})
        if not self.show:
            return
        rgb, _ = pp.sim.read_camera()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = largest_component(color_mask(rgb, **sys.modules["src.task.pick_place"].CUBE_HSV), min_pixels=100)
        if mask is not None:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(bgr, contours, -1, (0, 255, 0), 2)
        lines = [f"{pp.state}  t {pp.elapsed:5.1f} s", f"|qd cmd| max {np.max(np.abs(pp.io.last_cmd)):.3f} rad/s"]
        if ekf is not None:
            lines.append(f"EKF speed {1000 * np.linalg.norm(ekf.velocity):4.1f} mm/s")
        for i, text in enumerate(lines):
            cv2.putText(bgr, text, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(bgr, text, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1,
                        cv2.LINE_AA)
        cv2.imshow("run_real", bgr)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            self.quit = True
            pp.timeout = 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", action="store_true", help="run against the simulation (scene camera fixed_cam)")
    p.add_argument("--ip", default=ROBOT_IP)
    p.add_argument("--dry-run", action="store_true", help="real camera and robot state, no motion commands")
    p.add_argument("--mode", choices=["track", "pick"], default="track")
    p.add_argument("--place-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
                   help="place position in the RTDE base frame (m)")
    p.add_argument("--cube-mm", type=float, default=None, help="cube edge (default 60, sim 50)")
    p.add_argument("--table-z-mm", type=float, default=None,
                   help="table surface z in the RTDE base frame; needed to move the robot (otherwise default 0)")
    p.add_argument("--tcp-mm", type=float, default=None,
                   help="flange to RG2 fingertip centre; needed to move the robot (otherwise default: sim 156)")
    p.add_argument("--cube-hsv", default=None,
                   help=f"cube colour thresholds from scripts/tune_hsv.py (default on the robot: {config.CUBE_HSV_FILE.name})")
    p.add_argument("--hover-mm", type=float, default=100.0, help="TCP above the grasp pose while tracking")
    p.add_argument("--v-max", type=float, default=0.08, help="TCP speed limit while tracking (m/s)")
    p.add_argument("--qd-max", type=float, default=QD_MAX, help="joint speed limit (rad/s)")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=0, help="sim only")
    p.add_argument("--static-cube", action="store_true", help="sim only: the cube does not move")
    p.add_argument("--no-camera-noise", action="store_true", help="sim only")
    p.add_argument("--depth-k", type=float, default=SIM_DEPTH_K, help="sim only: depth noise k_axial (1/m)")
    p.add_argument("--depth-frames", type=int, default=DEPTH_FRAMES, help="depth frames averaged per measurement")
    p.add_argument("--no-window", action="store_true")
    args = p.parse_args()

    if not args.sim and not args.dry_run:
        missing = [flag for flag, value in (("--tcp-mm", args.tcp_mm), ("--table-z-mm", args.table_z_mm))
                   if value is None]
        if missing:
            p.error(f"measure and pass {' and '.join(missing)} before moving the robot "
                    "(the safety limits rely on them, so a wrong default would not be caught)")
    if args.tcp_mm is not None and not TCP_RANGE_MM[0] <= args.tcp_mm <= TCP_RANGE_MM[1]:
        p.error(f"--tcp-mm {args.tcp_mm} is outside {TCP_RANGE_MM} mm; it is in millimetres")

    cube_hsv = None
    if args.cube_hsv is not None:
        cube_hsv = config.load_cube_hsv(args.cube_hsv)
    elif not args.sim:
        if config.CUBE_HSV_FILE.exists():
            cube_hsv = config.load_cube_hsv(config.CUBE_HSV_FILE)
            print(f"cube colour thresholds from {config.CUBE_HSV_FILE.name}: {cube_hsv}")
        else:
            print("warning: no thresholds tuned on the lab camera, using the simulation's; run scripts/tune_hsv.py")

    robot = UR3eModel()
    table_z = SIM_BASE_HEIGHT + (args.table_z_mm or 0.0) / 1000.0
    cube_mm = args.cube_mm if args.cube_mm is not None else (50.0 if args.sim else 60.0)
    settings = config.RealSettings(cube_side=cube_mm / 1000.0, table_z=table_z,
                                   tcp_offset=None if args.tcp_mm is None else args.tcp_mm / 1000.0,
                                   cube_hsv=cube_hsv, hover=args.hover_mm / 1000.0, track_v_max=args.v_max,
                                   lab_estimation=not args.sim)
    changed = config.apply(settings, robot)
    print("settings: " + ", ".join(f"{k} {old} -> {new}" for k, (old, new) in changed.items() if old != new))
    limits = SafetyLimits(table_z=table_z, qd_max=args.qd_max)

    if args.sim:
        io = SimIO(robot, limits, seed=args.seed, moving_cube=not args.static_cube,
                   camera_noise=not args.no_camera_noise, depth_k=args.depth_k, depth_frames=args.depth_frames)
        camera = FixedCameraModel.from_mujoco(io.sim.model, io.sim.data, "fixed_cam", io.K_true)
        place_world = (rtde_xy_to_world(args.place_xy) if args.place_xy is not None
                       else np.asarray(io.sim.trial.zone_xy, dtype=float))
    else:
        camera = FixedCameraModel.from_calibration()
        io = RealIO(robot, limits, args.ip, dry_run=args.dry_run, cube_side_mm=cube_mm, acceleration=1.5,
                    depth_frames=args.depth_frames)
        if np.max(np.abs(io.K - camera.K)) > 1.0:
            print("warning: the camera intrinsics differ from the calibration file (another camera?)")
        place_world = None if args.place_xy is None else rtde_xy_to_world(args.place_xy)
    print(f"camera: {camera.source}, at {np.round(camera.p_world_cam, 3)} m (world)")

    viewer = Viewer(show=not args.no_window)
    pp = RealPickPlace(io, robot, camera, mode=args.mode, place_xy=place_world, on_step=viewer)
    out = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    outcome, error = None, None
    wall = time.perf_counter()
    try:
        outcome = pp.run(timeout=args.timeout)
    except SafetyStop as exc:
        error = f"safety stop: {exc}"
    except KeyboardInterrupt:
        error = "interrupted"
    finally:
        io.stop()
        io.close()
        cv2.destroyAllWindows()

    wall = time.perf_counter() - wall
    if error:
        print(f"\nSTOPPED in {pp.state}: {error}")
    elif viewer.quit:
        print(f"\nSTOPPED in {pp.state}: quit from the window")
    else:
        print(f"\n{'OK' if outcome['ok'] else 'FAILED'}: {outcome['reason'] or 'done'} "
              f"({pp.elapsed:.1f} s pipeline time, {wall:.1f} s wall)")
    for r in pp.records:
        print(f"  {r['state']:10s} {'ok ' if r['ok'] else 'FAIL'} {r['elapsed']:6.2f} s  {r['reason'] or ''}")
    if pp.z_offsets:
        print(f"  camera cube height minus the known height: {1000 * np.median(pp.z_offsets):+.1f} mm median "
              f"({len(pp.z_offsets)} estimates); a few mm off means calibration or --table-z-mm is off")
    if pp.align_deg is not None:
        print(f"  grasp yaw changed by {pp.align_deg:.1f} deg after TRACK (cube yaw {np.degrees(pp.yaw_at_stop):+.1f} deg)")
    if pp.track_log:
        st = [e["status"] for e in pp.track_log]
        print(f"  TRACK: {len(st)} vision updates, {st.count('none')} without the cube, "
              f"{st.count('rejected')} rejected, {st.count('reinit')} re-initialised")
        if pp.track_stop_time is not None:
            print(f"  stop declared at {pp.track_stop_time:.2f} s"
                  + (f" (true stop {io.sim.cube_motion.stop_time:.2f} s)"
                     if args.sim and io.sim.cube_motion is not None else ""))
    if args.sim:
        gt = io.sim.ground_truth()
        held = gt["cube_pos"][2] > table_z + cube_mm / 2000.0 + 0.01
        print(f"  sim truth: cube at {np.round(gt['cube_pos'], 3)}, "
              f"{1000 * np.linalg.norm(gt['cube_pos'][:2] - place_world):.1f} mm from the place position"
              + (", still held" if held else ""))
    if not args.sim:
        print(f"  control overruns: {io.overruns}, stale camera frames: {io.stale_frames}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({
        "args": vars(args), "error": error, "outcome": None if outcome is None else
        {k: v for k, v in outcome.items() if k != "states"}, "records": pp.records, "track_log": pp.track_log,
        "samples": viewer.log, "settings": {k: list(v) for k, v in changed.items()},
    }, indent=1, default=float))
    print(f"  log: {out}/run.json")
    sys.exit(0 if outcome is not None and outcome["ok"] and not error else 1)


if __name__ == "__main__":
    main()
