"""Run the vision-guided pick-and-place: one command for the demo and for batches of trials.

    python main.py                                  # live window, L0, seed 0, LQR: the cube slides, then stops;
                                                    # follow it (cube EKF + PBVS) and pick once it has stopped
    python main.py --trials 20 --no-window          # seeds 0-19, prints a summary
    python main.py --record videos/pick_place.mp4 --no-window
    python main.py --viewer                         # also show the run in the interactive MuJoCo viewer
    python main.py --no-camera-noise                # exact wrist RGB-D (noise is on by default)
    python main.py --nominal-camera                 # nominal K and camera mount instead of the calibration
    python main.py --no-moving-cube --approach open    # static cube, look-then-move
    python main.py --no-moving-cube --approach pbvs    # static cube, PBVS from the pregrasp pose
    python main.py --no-moving-cube --approach pbvs --push-cube 0.025   # push the cube 25 mm as the approach starts
    python main.py --no-moving-cube --approach pbvs_full --push-cube 0.025 --push-at transit_mid
    python main.py --no-moving-cube --seed 7 --level L1 --mode pd --approach pbvs_full
    python main.py --no-moving-cube --handeye-error-deg 2   # experiment: tilt the calibrated camera mount by 2 deg

The pipeline (src/task/pick_place.py) only sees encoder readings and the wrist camera. Ground truth is read
here, in TrialMonitor, solely to judge success (project plan section 6) and to show the true state next to
the estimate on screen; it never reaches the estimator, planner or controller.
"""

import argparse
import dataclasses
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from src.control.arm_controller import ArmController
from src.model.ur3e import UR3eModel
from src.perception.camera_model import CameraModel
from src.sim import CameraNoiseConfig, CubeMotionConfig, TaskSim
from src.sim.randomize import CUBE_HALF, TABLE_Z
from src.task.pick_place import PickPlace

FPS = 30
PANEL_W, PANEL_H = 640, 480

# Success criteria from the project plan (section 6).
LIFT_HEIGHT = 0.05  # m above the table
LIFT_HOLD = 2.0  # s
PLACE_TOL = 0.02  # m, cube centre to zone centre
REST_TOL = 0.005  # m, cube centre height above its resting height


class TrialMonitor:
    """EVALUATION ONLY: reads ground truth to judge the trial against the success criteria."""

    def __init__(self, sim: TaskSim):
        self.sim = sim
        self.lift_time = 0.0
        self.max_lift_time = 0.0
        # True cube pose at the moment of the latest cube estimate, so the error shown is the estimation
        # error rather than how far the cube has since been carried.
        self._estimate_seen = None
        self.cube_truth_at_estimate = None
        # Horizontal cube-to-TCP offset when the cube first leaves the table: how well the grasp was centred.
        self.grasp_offset = None
        # Horizontal cube-to-TCP distance while the arm hovers above the moving cube (TRACK, after the move from
        # the observe pose): how closely the arm follows.
        self.track_lag = []

    def step(self, pp: PickPlace) -> None:
        if pp.cube_estimate is not None and pp.cube_estimate is not self._estimate_seen:
            self._estimate_seen = pp.cube_estimate
            gt = self.sim.ground_truth()
            self.cube_truth_at_estimate = (gt["cube_pos"].copy(), 2 * np.arctan2(gt["cube_quat"][3], gt["cube_quat"][0]))
        gt = self.sim.ground_truth()
        cube_z = gt["cube_pos"][2]
        if pp.state == "TRACK" and pp.progress >= 1.0 and np.linalg.norm(gt["cube_vel"][:2]) > 1e-3:
            self.track_lag.append(float(np.linalg.norm(gt["cube_pos"][:2] - gt["tcp_pos"][:2])))
        if self.grasp_offset is None and cube_z - (TABLE_Z + CUBE_HALF) > 0.01:
            self.grasp_offset = float(np.linalg.norm(gt["cube_pos"][:2] - gt["tcp_pos"][:2]))
        if cube_z - (TABLE_Z + CUBE_HALF) > LIFT_HEIGHT:
            self.lift_time += self.sim.dt
            self.max_lift_time = max(self.max_lift_time, self.lift_time)
        else:
            self.lift_time = 0.0

    def result(self) -> dict:
        gt = self.sim.ground_truth()
        place_err = float(np.linalg.norm(gt["cube_pos"][:2] - gt["zone_pos"][:2]))
        resting = abs(gt["cube_pos"][2] - (TABLE_Z + CUBE_HALF)) < REST_TOL
        grasped = self.max_lift_time >= LIFT_HOLD
        return {
            "grasped": grasped,
            "placed": grasped and resting and place_err < PLACE_TOL,
            "place_err": place_err,
            "max_lift_time": self.max_lift_time,
            "grasp_offset": self.grasp_offset,
            "track_lag_rms": float(np.sqrt(np.mean(np.square(self.track_lag)))) if self.track_lag else None,
        }


class CubePush:
    """DISTURBANCE FOR EXPERIMENTS ONLY: moves the cube by `distance` metres along the world +y axis once,
    like a person nudging it. Writes the simulator state directly; the pipeline only notices through the camera.

    at="approach_start": when the move to the pregrasp pose ends (start of APPROACH, or of DESCEND for "open").
    at="transit_mid": half-way through the move to the pregrasp pose (PREGRASP or TRANSIT progress >= 0.5);
    the same instant for every approach, since quintic timing puts s = 0.5 at t = T/2."""

    def __init__(self, sim: TaskSim, distance: float, at: str = "approach_start"):
        if at not in ("approach_start", "transit_mid"):
            raise ValueError("at must be 'approach_start' or 'transit_mid'")
        self.sim = sim
        self.distance = float(distance)
        self.at = at
        self.done = self.distance == 0.0

    def step(self, pp: PickPlace) -> None:
        if self.done:
            return
        if self.at == "approach_start":
            trigger = pp.state in ("APPROACH", "DESCEND")
        else:
            trigger = pp.state in ("PREGRASP", "TRANSIT") and pp.progress >= 0.5
        if trigger:
            adr = self.sim.model.joint("cube_joint").qposadr[0]
            self.sim.data.qpos[adr + 1] += self.distance
            mujoco.mj_forward(self.sim.model, self.sim.data)
            self.done = True


# ---- display -------------------------------------------------------------------------------------

def depth_to_colour(depth: np.ndarray) -> np.ndarray:
    """Grayscale depth for display: near is bright, far is dark; invalid pixels (0) are black.

    The scale follows each frame (5th to 99th percentile of the valid depths, so the gripper fingers at the
    image bottom do not set it), because the scene spans
    0.1 m when the cube is held and 0.4 m at the observe pose."""
    valid = depth > 0
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), np.uint8)
    lo, hi = np.percentile(depth[valid], [5, 99])
    d = np.clip((depth - lo) / max(hi - lo, 1e-3), 0, 1)
    gray = (255 * (1 - d)).astype(np.uint8)
    gray[depth <= 0] = 0
    return np.repeat(gray[..., None], 3, axis=2)


def label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (len(text) * 11 + 12, 28), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def put(img, text, x, y, colour=(230, 230, 230), scale=0.5):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def info_panel(pp: PickPlace, monitor: TrialMonitor, seed: int, level: str, mode: str) -> np.ndarray:
    img = np.full((PANEL_H, PANEL_W, 3), 30, np.uint8)
    put(img, f"t = {pp.elapsed:5.2f} s   seed {seed}   level {level}   controller {mode.upper()}", 14, 24, scale=0.55)
    put(img, f"state: {pp.state}", 14, 54, (150, 220, 150), 0.7)
    noise = "noisy" if pp.sim.camera_noise is not None else "exact"
    put(img, f"camera: {pp.camera.source}, {noise}", 410, 54, (200, 200, 200), 0.45)
    put(img, f"approach: {pp.approach}", 410, 74, (200, 200, 200), 0.45)

    gt = pp.sim.ground_truth()
    put(img, "object            estimate            true*            error", 14, 92, (180, 180, 255))
    cube = pp.cube_estimate
    if cube is not None and cube.ok and monitor.cube_truth_at_estimate is not None:
        true_pos, true_yaw = monitor.cube_truth_at_estimate
        yaw_err = (cube.yaw - true_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        put(img, f"cube x,y (mm)   {cube.pos[0]*1e3:6.1f},{cube.pos[1]*1e3:6.1f}"
                 f"   {true_pos[0]*1e3:6.1f},{true_pos[1]*1e3:6.1f}"
                 f"   {np.linalg.norm(cube.pos[:2] - true_pos[:2])*1e3:5.1f} mm", 14, 118)
        put(img, f"cube yaw (deg)  {np.rad2deg(cube.yaw):13.1f}   {np.rad2deg(true_yaw):13.1f}"
                 f"   {abs(np.rad2deg(yaw_err)):5.2f} deg", 14, 142)
    else:
        put(img, "cube            not estimated yet", 14, 118)
    ekf = pp.cube_ekf
    if ekf is not None:
        sigma_xy = 2e3 * np.sqrt(np.max(np.linalg.eigvalsh(ekf.P[:2, :2])))
        put(img, f"cube v (mm/s)   {ekf.velocity[0]*1e3:6.1f},{ekf.velocity[1]*1e3:6.1f}"
                 f"   {gt['cube_vel'][0]*1e3:6.1f},{gt['cube_vel'][1]*1e3:6.1f}"
                 f"   EKF 2-sigma xy {sigma_xy:4.1f} mm", 14, 210, (200, 200, 200), 0.45)
    zone = pp.zone_estimate
    if zone is not None and np.all(np.isfinite(zone.xy)):
        put(img, f"zone x,y (mm)   {zone.xy[0]*1e3:6.1f},{zone.xy[1]*1e3:6.1f}"
                 f"   {gt['zone_pos'][0]*1e3:6.1f},{gt['zone_pos'][1]*1e3:6.1f}"
                 f"   {np.linalg.norm(zone.xy - gt['zone_pos'][:2])*1e3:5.1f} mm", 14, 166)
        put(img, f"zone fully observed: {zone.complete}", 14, 188, (200, 200, 200), 0.45)
    else:
        put(img, "zone            not estimated yet", 14, 166)

    xhat = pp.ctrl.xhat
    put(img, "joint   est. qd (deg/s)   true qd*   est. d (N m)", 14, 230, (180, 180, 255))
    for i, name in enumerate(["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]):
        put(img, f"{name:7s} {np.rad2deg(xhat[i, 1]):12.2f} {np.rad2deg(gt['qd'][i]):12.2f} {xhat[i, 2]:12.2f}",
            14, 254 + 22 * i)

    put(img, f"cube lifted > 5 cm for {monitor.max_lift_time:4.1f} s", 14, 400, (200, 200, 200), 0.45)
    put(img, "* true values: simulator ground truth, shown for evaluation only", 14, 446, (150, 150, 150), 0.42)
    put(img, "  (cube: true pose at the time of the estimate)", 14, 466, (150, 150, 150), 0.42)
    return img


class Display:
    """Composite view (overview, wrist RGB, wrist depth, status) shown live and/or written to video."""

    def __init__(self, sim: TaskSim, window: bool, record: Path | None):
        self.window = window
        self.overview = mujoco.Renderer(sim.model, height=PANEL_H, width=PANEL_W)
        self.writer = None
        if record is not None:
            import imageio.v2 as imageio

            record.parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(record, fps=FPS)
        self.quit = False

    def show(self, pp: PickPlace, monitor: TrialMonitor, seed: int, level: str, mode: str, wall_start: float):
        rgb, depth = pp.sim.read_camera_for_display()
        self.overview.update_scene(pp.sim.data, "overview")
        top = np.hstack([label(self.overview.render().copy(), "overview"), label(rgb, "wrist RGB")])
        bottom = np.hstack([label(depth_to_colour(depth), "wrist depth"), info_panel(pp, monitor, seed, level, mode)])
        frame = np.vstack([top, bottom])
        if self.writer is not None:
            self.writer.append_data(frame)
        if self.window:
            lag = pp.elapsed - (time.perf_counter() - wall_start)  # pace the live view to real time
            if lag > 0:
                time.sleep(lag)
            cv2.imshow("AVRC pick and place", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                self.quit = True

    def close(self):
        if self.writer is not None:
            self.writer.close()
        self.overview.close()
        if self.window:
            cv2.destroyAllWindows()


# ---- trials ----------------------------------------------------------------------------------------

def run_trial(sim, robot, camera, seed, level, mode, timeout, approach="track", push=0.0,
              push_at="approach_start", display=None, viewer=None, fixed_gains=False, occluders="off",
              occluder_margin=0.0, cube_filter="cv") -> dict:
    sim.reset(seed, level)
    ctrl = ArmController(robot=robot, dt=sim.dt, y0=sim.read_joint_positions(), mode=mode,
                         schedule_period=0 if fixed_gains else 50)
    monitor = TrialMonitor(sim)
    disturbance = CubePush(sim, push, push_at)
    steps_per_frame = max(1, round(1.0 / (FPS * sim.dt)))
    wall_start = time.perf_counter()
    count = 0

    def on_step(pp: PickPlace) -> None:
        nonlocal count
        disturbance.step(pp)
        monitor.step(pp)
        count += 1
        if count % steps_per_frame:
            return
        if display is not None:
            display.show(pp, monitor, seed, level, mode, wall_start)
            if display.quit:
                pp.timeout = 0.0  # ends the run at the next timeout check
        if viewer is not None:
            if not viewer.is_running():
                pp.timeout = 0.0
                return
            if display is None or not display.window:  # otherwise Display already paces to real time
                lag = pp.elapsed - (time.perf_counter() - wall_start)
                if lag > 0:
                    time.sleep(lag)
            viewer.sync()

    pick_place = PickPlace(sim, robot, ctrl, on_step=on_step, camera=camera, approach=approach, occluders=occluders,
                           occluder_margin=occluder_margin, cube_filter=cube_filter)
    outcome = pick_place.run(timeout=timeout)
    result = monitor.result()
    failed_state = next((r["state"] for r in outcome["states"] if not r["ok"]), None)
    log = pick_place.transit_log
    attempts = [e["status"] for e in log if e["vision_attempt"]]
    transit = None if not log else {
        "frozen": sum(e["frozen"] for e in log) / len(log),
        "no_detection": attempts.count("none") / max(len(attempts), 1),
        "rejected": attempts.count("rejected"),
    }
    track = None
    if pick_place.track_log:
        statuses = [e["status"] for e in pick_place.track_log]
        stop = pick_place.track_stop_time
        true_stop = sim.cube_motion.stop_time if sim.cube_motion is not None else None
        track = {
            "stop_delay": None if stop is None or true_stop is None else stop - true_stop,
            "no_detection": statuses.count("none") / len(statuses),
            "rejected": statuses.count("rejected"),
            "reinit": statuses.count("reinit"),
        }
    return {"seed": seed, "pipeline_ok": outcome["ok"], "reason": outcome["reason"],
            "failed_state": failed_state, "sim_time": outcome["elapsed"], "transit": transit, "track": track,
            "extra_views": max(len(pick_place.occluder_views) - 1, 0),
            **result}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="first seed")
    p.add_argument("--trials", type=int, default=1, help="number of consecutive seeds to run")
    p.add_argument("--level", default="L0", choices=["L0", "L1"])
    p.add_argument("--mode", default="lqr", choices=["lqr", "pd"], help="joint feedback law")
    p.add_argument("--timeout", type=float, default=60.0, help="simulated seconds per trial")
    p.add_argument("--record", type=Path, help="write an mp4 of the composite view")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--no-camera-noise", action="store_true",
                   help="exact wrist RGB-D; by default the camera is noisy (src/sim/sensors.py CameraNoiseConfig)")
    p.add_argument("--nominal-camera", action="store_true",
                   help="nominal K and camera mount instead of configs/camera_calib.json")
    p.add_argument("--approach", default="track", choices=["track", "open", "pbvs", "pbvs_full"],
                   help="track (default): follow the cube with the cube EKF and PBVS, pick once it has stopped; "
                        "open: look-then-move; pbvs: visual servoing from the pregrasp pose; pbvs_full: also for the "
                        "move from the observe pose (for the occluded levels)")
    p.add_argument("--push-cube", type=float, default=0.0, metavar="M",
                   help="experiment: push the cube this many metres along +y (see --push-at)")
    p.add_argument("--push-at", default="approach_start", choices=["approach_start", "transit_mid"],
                   help="when to push: start of the approach (default) or half-way to the pregrasp pose")
    p.add_argument("--handeye-error-deg", type=float, default=0.0, metavar="DEG",
                   help="experiment: rotate the camera mount used by the pipeline by DEG about the camera x axis")
    p.add_argument("--viewer", action="store_true",
                   help="also show the run in the interactive MuJoCo viewer (display only, paced to real time)")
    p.add_argument("--moving-cube", action=argparse.BooleanOptionalAction, default=True,
                   help="moving-cube task (default): from 0.5 s the cube slides in a straight line at 1-1.5 cm/s for "
                        "9-11 s, then stops, and the place zone is fixed; --no-moving-cube: static cube, random zone")
    p.add_argument("--occluders", default="off", choices=["off", "conservative", "active"],
                   help="occluder handling: off (default); conservative: estimate occluders, keep grasp yaws that "
                        "clear them (unseen table counts as occupied), lift above them; active: also take up to two "
                        "more views, chosen by what the grasp decision lacks (src/active_vision/next_view.py)")
    p.add_argument("--occluder-margin", type=float, default=0.0, metavar="M",
                   help="extra clearance (m) the grasp must keep from the estimated occluders (default 0)")
    p.add_argument("--cube-filter", default="cv", choices=["cv", "ct"],
                   help="TRACK estimator: cv (constant-velocity CubeEKF, default) or ct (coordinated-turn CubeCTEKF)")
    p.add_argument("--fixed-gains", action="store_true",
                   help="comparison: joint models and gains from the observe pose only, no gain scheduling")
    args = p.parse_args()

    sim = TaskSim(camera_noise=None if args.no_camera_noise else CameraNoiseConfig(),
                  cube_motion=CubeMotionConfig() if args.moving_cube else None)
    robot = UR3eModel()
    camera = CameraModel.nominal(robot) if args.nominal_camera else CameraModel.from_calibration()
    if args.handeye_error_deg:
        # Tilt about the camera x axis: an error about the optical axis would barely move a centred cube.
        R_err = Rotation.from_rotvec(np.deg2rad(args.handeye_error_deg) * np.array([1.0, 0.0, 0.0])).as_matrix()
        camera = dataclasses.replace(camera, R_ee_cam=camera.R_ee_cam @ R_err,
                                     source=f"{camera.source} + {args.handeye_error_deg:g} deg error")
    show = not args.no_window
    display = Display(sim, show, args.record) if (show or args.record) else None
    viewer = None
    if args.viewer:
        import mujoco.viewer

        # Passive viewer: the pipeline keeps stepping the simulation, the viewer only draws sim.data.
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)

    results = []
    try:
        for seed in range(args.seed, args.seed + args.trials):
            r = run_trial(sim, robot, camera, seed, args.level, args.mode, args.timeout,
                          args.approach, args.push_cube, args.push_at, display, viewer, args.fixed_gains,
                          args.occluders, args.occluder_margin, args.cube_filter)
            results.append(r)
            verdict = "SUCCESS" if r["placed"] else "FAIL"
            detail = "" if r["placed"] else f"  ({r['reason'] or 'success criteria not met'})"
            offset = "  -  " if r["grasp_offset"] is None else f"{r['grasp_offset']*1e3:5.1f}"
            views = f"  extra views {r['extra_views']}" if args.occluders == "active" else ""
            print(f"seed {seed:4d}  {verdict:7s}  grasped={r['grasped']!s:5s}  grasp offset {offset} mm"
                  f"  place error {r['place_err']*1e3:6.1f} mm  sim time {r['sim_time']:5.1f} s{views}{detail}",
                  flush=True)
            if (display is not None and display.quit) or (viewer is not None and not viewer.is_running()):
                break
    finally:
        if display is not None:
            display.close()
        if viewer is not None:
            viewer.close()
        sim.close()

    if len(results) > 1:
        placed = [r for r in results if r["placed"]]
        push = f", push {args.push_cube*1e3:.0f} mm at {args.push_at}" if args.push_cube else ""
        if args.moving_cube:
            push += ", moving cube"
        if args.fixed_gains:
            push += ", fixed gains"
        if args.handeye_error_deg:
            push += f", hand-eye error {args.handeye_error_deg:g} deg"
        if args.occluders != "off":
            push += f", occluders {args.occluders} (margin {args.occluder_margin * 1e3:g} mm)"
        print(f"\n{args.level}, {args.mode.upper()}, {args.approach}{push}: {len(placed)}/{len(results)} placed, "
              f"{sum(r['grasped'] for r in results)}/{len(results)} grasped")
        if placed:
            err = np.array([r["place_err"] for r in placed]) * 1e3
            print(f"place error (successful trials): median {np.median(err):.1f} mm, max {err.max():.1f} mm")
        offsets = np.array([r["grasp_offset"] for r in results if r["grasp_offset"] is not None]) * 1e3
        if offsets.size:
            print(f"grasp offset (cube to TCP when lifted): median {np.median(offsets):.1f} mm, max {offsets.max():.1f} mm")
        times = np.array([r["sim_time"] for r in placed])
        if times.size:
            print(f"sim time (successful trials): median {np.median(times):.1f} s")
        transits = [r["transit"] for r in results if r["transit"] is not None]
        if transits:
            print(f"TRANSIT: frozen steps median {100*np.median([x['frozen'] for x in transits]):.1f} %, "
                  f"no detection median {100*np.median([x['no_detection'] for x in transits]):.0f} % of vision "
                  f"updates, rejected measurements {sum(x['rejected'] for x in transits)}")
        tracks = [r["track"] for r in results if r["track"] is not None]
        if tracks:
            delays = np.array([x["stop_delay"] for x in tracks if x["stop_delay"] is not None])
            lags = np.array([r["track_lag_rms"] for r in results if r["track_lag_rms"] is not None]) * 1e3
            line = (f"TRACK: rejected measurements {sum(x['rejected'] for x in tracks)}, "
                    f"re-initialisations {sum(x['reinit'] for x in tracks)}")
            if delays.size:
                line += (f", stop declared {np.median(delays):.2f} s after the true stop (median, "
                         f"min {delays.min():.2f}, max {delays.max():.2f})")
            if lags.size:
                line += f", TCP-to-cube lag while moving RMS median {np.median(lags):.1f} mm"
            print(line)
        failures = {}
        for r in results:
            if not r["placed"]:
                failures.setdefault(r["failed_state"] or "criteria", []).append(r["seed"])
        for state, seeds in failures.items():
            print(f"failed in {state}: seeds {seeds}")


if __name__ == "__main__":
    main()
