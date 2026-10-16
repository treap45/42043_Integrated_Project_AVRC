"""Run the vision-guided pick-and-place: one command for the demo and for batches of trials.

    python main.py                                  # live window, L0, seed 0, LQR
    python main.py --seed 7 --level L1 --mode pd
    python main.py --trials 20 --no-window          # seeds 0-19, prints a summary
    python main.py --record videos/pick_place.mp4 --no-window
    python main.py --no-camera-noise                # exact wrist RGB-D (noise is on by default)
    python main.py --nominal-camera                 # nominal K and camera mount instead of the calibration
    python main.py --approach open                  # look-then-move instead of PBVS
    python main.py --push-cube 0.02                 # push the cube 2 cm when the approach starts

The pipeline (src/task/pick_place.py) only sees encoder readings and the wrist camera. Ground truth is read
here, in TrialMonitor, solely to judge success (project plan section 6) and to show the true state next to
the estimate on screen; it never reaches the estimator, planner or controller.
"""

import argparse
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from src.control.arm_controller import ArmController
from src.model.ur3e import UR3eModel
from src.perception.camera_model import CameraModel
from src.sim import CameraNoiseConfig, TaskSim
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

    def step(self, pp: PickPlace) -> None:
        if pp.cube_estimate is not None and pp.cube_estimate is not self._estimate_seen:
            self._estimate_seen = pp.cube_estimate
            gt = self.sim.ground_truth()
            self.cube_truth_at_estimate = (gt["cube_pos"].copy(), 2 * np.arctan2(gt["cube_quat"][3], gt["cube_quat"][0]))
        gt = self.sim.ground_truth()
        cube_z = gt["cube_pos"][2]
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
        }


class CubePush:
    """DISTURBANCE FOR EXPERIMENTS ONLY: moves the cube by `distance` metres along the world +y axis once,
    as soon as the pipeline leaves PREGRASP (start of APPROACH, or of DESCEND without PBVS), like a person
    nudging it. Writes the simulator state directly; the pipeline only notices through the camera."""

    def __init__(self, sim: TaskSim, distance: float):
        self.sim = sim
        self.distance = float(distance)
        self.done = self.distance == 0.0
        self._seen_pregrasp = False

    def step(self, pp: PickPlace) -> None:
        if self.done:
            return
        if pp.state == "PREGRASP":
            self._seen_pregrasp = True
        elif self._seen_pregrasp:
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
    zone = pp.zone_estimate
    if zone is not None and np.all(np.isfinite(zone.xy)):
        put(img, f"zone x,y (mm)   {zone.xy[0]*1e3:6.1f},{zone.xy[1]*1e3:6.1f}"
                 f"   {gt['zone_pos'][0]*1e3:6.1f},{gt['zone_pos'][1]*1e3:6.1f}"
                 f"   {np.linalg.norm(zone.xy - gt['zone_pos'][:2])*1e3:5.1f} mm", 14, 166)
        put(img, f"zone fully observed: {zone.complete}", 14, 190, (200, 200, 200), 0.45)
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

def run_trial(sim, robot, camera, seed, level, mode, timeout, approach="pbvs", push=0.0, display=None) -> dict:
    sim.reset(seed, level)
    ctrl = ArmController(robot=robot, dt=sim.dt, y0=sim.read_joint_positions(), mode=mode)
    monitor = TrialMonitor(sim)
    disturbance = CubePush(sim, push)
    steps_per_frame = max(1, round(1.0 / (FPS * sim.dt)))
    wall_start = time.perf_counter()
    count = 0

    def on_step(pp: PickPlace) -> None:
        nonlocal count
        disturbance.step(pp)
        monitor.step(pp)
        count += 1
        if display is not None and count % steps_per_frame == 0:
            display.show(pp, monitor, seed, level, mode, wall_start)
            if display.quit:
                pp.timeout = 0.0  # ends the run at the next timeout check

    pick_place = PickPlace(sim, robot, ctrl, on_step=on_step, camera=camera, approach=approach)
    outcome = pick_place.run(timeout=timeout)
    result = monitor.result()
    failed_state = next((r["state"] for r in outcome["states"] if not r["ok"]), None)
    return {"seed": seed, "pipeline_ok": outcome["ok"], "reason": outcome["reason"],
            "failed_state": failed_state, "sim_time": outcome["elapsed"], **result}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="first seed")
    p.add_argument("--trials", type=int, default=1, help="number of consecutive seeds to run")
    p.add_argument("--level", default="L0", choices=["L0", "L1", "L2"])
    p.add_argument("--mode", default="lqr", choices=["lqr", "pd"], help="joint feedback law")
    p.add_argument("--timeout", type=float, default=60.0, help="simulated seconds per trial")
    p.add_argument("--record", type=Path, help="write an mp4 of the composite view")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--no-camera-noise", action="store_true",
                   help="exact wrist RGB-D; by default the camera is noisy (src/sim/sensors.py CameraNoiseConfig)")
    p.add_argument("--nominal-camera", action="store_true",
                   help="nominal K and camera mount instead of configs/camera_calib.json")
    p.add_argument("--approach", default="pbvs", choices=["pbvs", "open"],
                   help="pbvs: visual servoing from the pregrasp pose; open: look-then-move")
    p.add_argument("--push-cube", type=float, default=0.0, metavar="M",
                   help="experiment: push the cube this many metres along +y when the approach starts")
    args = p.parse_args()

    sim = TaskSim(camera_noise=None if args.no_camera_noise else CameraNoiseConfig())
    robot = UR3eModel()
    camera = CameraModel.nominal(robot) if args.nominal_camera else CameraModel.from_calibration()
    show = not args.no_window
    display = Display(sim, show, args.record) if (show or args.record) else None

    results = []
    try:
        for seed in range(args.seed, args.seed + args.trials):
            r = run_trial(sim, robot, camera, seed, args.level, args.mode, args.timeout,
                          args.approach, args.push_cube, display)
            results.append(r)
            verdict = "SUCCESS" if r["placed"] else "FAIL"
            detail = "" if r["placed"] else f"  ({r['reason'] or 'success criteria not met'})"
            offset = "  -  " if r["grasp_offset"] is None else f"{r['grasp_offset']*1e3:5.1f}"
            print(f"seed {seed:4d}  {verdict:7s}  grasped={r['grasped']!s:5s}  grasp offset {offset} mm"
                  f"  place error {r['place_err']*1e3:6.1f} mm  sim time {r['sim_time']:5.1f} s{detail}", flush=True)
            if display is not None and display.quit:
                break
    finally:
        if display is not None:
            display.close()
        sim.close()

    if len(results) > 1:
        placed = [r for r in results if r["placed"]]
        push = f", push {args.push_cube*1e3:.0f} mm" if args.push_cube else ""
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
        failures = {}
        for r in results:
            if not r["placed"]:
                failures.setdefault(r["failed_state"] or "criteria", []).append(r["seed"])
        for state, seeds in failures.items():
            print(f"failed in {state}: seeds {seeds}")


if __name__ == "__main__":
    main()
