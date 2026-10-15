"""Live simulation demo: overview camera, wrist RGB, wrist depth, and joint estimation/control status.

The arm is driven by the joint controller (src/control/arm_controller.py) using only encoder readings.
The scene is re-randomised every --reset-every seconds (or press "r"); press "q" or Esc to quit.

Run from the repo root with the venv active:
    python scripts/demo_sim.py                              # live window, L0, PD, hold the observe pose
    python scripts/demo_sim.py --level L2 --mode lqr --motion
    python scripts/demo_sim.py --record videos/demo.mp4 --duration 10 --no-window

Ground truth (true joint velocity) is read only to display it next to the estimate. --gt-hold replaces the
controller with MuJoCo's own bias forces (ground truth) for checking the scene only; it must not be used
for anything that is assessed.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.control.arm_controller import ArmController  # noqa: E402
from src.model.ur3e import DEFAULT_XML, UR3eModel  # noqa: E402
from src.sim import N_ARM, TaskSim  # noqa: E402

FPS = 15
PANEL_W, PANEL_H = 640, 480
DEPTH_RANGE = (0.15, 0.8)  # metres, for the colour map
JOINTS = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]
PLOT_JOINT, PLOT_SECONDS = 1, 3.0  # joint shown in the velocity trace, and its time window

# --motion: q_ref = q_observe + A * (1 - cos(2 pi t / T)) / 2, which starts at rest.
MOTION_AMP = np.array([0.30, 0.30, -0.30, 0.40, 0.40, 0.60])  # rad
MOTION_PERIOD = 4.0  # s


class Pipeline:
    """Estimator + controller, rebuilt at every trial reset from the first encoder reading."""

    def __init__(self, mode: str, use_ff: bool, motion: bool):
        self.robot = UR3eModel()
        # The commanded start pose is part of the task definition, not a measurement.
        self.q_home = mujoco.MjModel.from_xml_path(str(DEFAULT_XML)).key("observe").qpos.copy()
        self.mode, self.use_ff, self.motion = mode, use_ff, motion
        self.ctrl = None

    def reset(self, dt: float, y0: np.ndarray) -> None:
        self.ctrl = ArmController(robot=self.robot, dt=dt, y0=y0, mode=self.mode, use_disturbance_ff=self.use_ff)

    def reference(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if not self.motion:
            return self.q_home, np.zeros(N_ARM)
        w = 2 * np.pi / MOTION_PERIOD
        return (self.q_home + MOTION_AMP * (1 - np.cos(w * t)) / 2,
                MOTION_AMP * w * np.sin(w * t) / 2)

    def step(self, y: np.ndarray, t: float) -> np.ndarray:
        q_ref, qd_ref = self.reference(t)
        return self.ctrl.step(y, q_ref, qd_ref)


def depth_to_colour(depth: np.ndarray) -> np.ndarray:
    lo, hi = DEPTH_RANGE
    d = np.clip((depth - lo) / (hi - lo), 0, 1)
    img = cv2.applyColorMap((255 * (1 - d)).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (len(text) * 11 + 12, 28), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def put(img, text, x, y, colour=(230, 230, 230), scale=0.5):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def info_panel(sim, seed, level, status, q_meas, xhat, qd_true, history) -> np.ndarray:
    img = np.full((PANEL_H, PANEL_W, 3), 30, np.uint8)
    put(img, f"t = {sim.time:5.2f} s   seed {seed}   level {level}   occluders {len(sim.trial.occluders)}",
        14, 24, scale=0.55)
    test_mode = status.startswith("TEST")
    put(img, status, 14, 48, (255, 120, 120) if test_mode else (150, 220, 150), 0.55)

    put(img, "joint     encoder q   est. qd   true qd*   est. d", 14, 80, (180, 180, 255))
    put(img, "            (deg)    (deg/s)    (deg/s)    (N m)", 14, 98, (180, 180, 255), 0.45)
    for i, name in enumerate(JOINTS):
        y = 122 + 22 * i
        if xhat is None:
            row = f"{name:7s} {np.rad2deg(q_meas[i]):10.2f}        -    {np.rad2deg(qd_true[i]):8.2f}        -"
        else:
            row = (f"{name:7s} {np.rad2deg(q_meas[i]):10.2f} {np.rad2deg(xhat[i, 1]):9.2f} "
                   f"{np.rad2deg(qd_true[i]):9.2f} {xhat[i, 2]:9.2f}")
        put(img, row, 14, y)
    put(img, "* true qd: simulator ground truth, shown for evaluation only", 14, 262, (150, 150, 150), 0.42)

    # Rolling trace of one joint's velocity: estimate vs ground truth.
    x0, y0, w, h = 50, 290, PANEL_W - 70, 170
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (90, 90, 90), 1)
    put(img, f"{JOINTS[PLOT_JOINT]} qd (deg/s), last {PLOT_SECONDS:g} s:  true", 14, y0 - 8, (230, 230, 230), 0.45)
    put(img, "estimate", 330, y0 - 8, (80, 170, 255), 0.45)
    if len(history) > 2:
        t, qt, qe = (np.array(v) for v in zip(*history))
        vals = np.concatenate([qt, qe[np.isfinite(qe)]]) if np.any(np.isfinite(qe)) else qt
        span = max(np.abs(vals).max(), 5.0) * 1.1
        put(img, f"{span:+.0f}", 8, y0 + 12, (150, 150, 150), 0.4)
        put(img, f"{-span:+.0f}", 8, y0 + h, (150, 150, 150), 0.4)
        cv2.line(img, (x0, y0 + h // 2), (x0 + w, y0 + h // 2), (70, 70, 70), 1)
        px = x0 + (t - t[-1] + PLOT_SECONDS) / PLOT_SECONDS * w

        def poly(v, colour):
            ok = np.isfinite(v)
            if ok.sum() > 1:
                py = y0 + h / 2 - v[ok] / span * (h / 2)
                pts = np.stack([px[ok], py], axis=1).astype(np.int32)
                cv2.polylines(img, [pts], False, colour, 1, cv2.LINE_AA)

        poly(qt, (230, 230, 230))
        poly(qe, (80, 170, 255))
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--level", default="L0", choices=["L0", "L1", "L2"])
    p.add_argument("--mode", default="pd", choices=["pd", "lqr"], help="joint feedback law")
    p.add_argument("--no-ff", action="store_true", help="disable the disturbance feed-forward (-d_hat)")
    p.add_argument("--motion", action="store_true", help="track a slow sinusoidal reference instead of holding")
    p.add_argument("--reset-every", type=float, default=5.0, help="seconds between re-randomisations")
    p.add_argument("--duration", type=float, default=float("inf"), help="stop after this many sim seconds")
    p.add_argument("--record", type=Path, help="write an mp4 of the composite view")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--gt-hold", action="store_true", help="TEST ONLY: hold the arm with ground-truth bias forces")
    args = p.parse_args()

    sim = TaskSim()
    pipeline = Pipeline(args.mode, not args.no_ff, args.motion)
    if args.gt_hold:
        status = "TEST MODE: arm held with ground truth"
    else:
        status = (f"controller: {args.mode.upper()}, d_hat feed-forward {'off' if args.no_ff else 'on'}, "
                  f"{'tracking' if args.motion else 'holding'}")

    def new_trial(seed):
        sim.reset(seed, args.level)
        y0 = sim.read_joint_positions()
        pipeline.reset(sim.dt, y0)
        return y0

    seed = args.seed
    q_meas = new_trial(seed)
    history = deque(maxlen=int(PLOT_SECONDS / sim.dt))
    overview = mujoco.Renderer(sim.model, height=PANEL_H, width=PANEL_W)
    writer = None
    if args.record:
        import imageio.v2 as imageio

        args.record.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.record, fps=FPS)

    steps_per_frame = max(1, round(1.0 / (FPS * sim.dt)))
    next_reset = args.reset_every
    elapsed = 0.0  # sim.time restarts at every reset
    wall_start = time.perf_counter()
    first_step = True
    try:
        while elapsed < args.duration:
            for _ in range(steps_per_frame):
                if not first_step:
                    q_meas = sim.read_joint_positions()
                first_step = False
                if args.gt_hold:
                    tau = sim.data.qfrc_bias[:N_ARM].copy()
                else:
                    tau = pipeline.step(q_meas, sim.time)
                sim.step(tau)
                # Display only: true velocity next to the estimate.
                qd_true = sim.ground_truth()["qd"]
                qd_est = np.nan if args.gt_hold else pipeline.ctrl.xhat[PLOT_JOINT, 1]
                history.append((elapsed, np.rad2deg(qd_true[PLOT_JOINT]), np.rad2deg(qd_est)))
                elapsed += sim.dt

            rgb, depth = sim.read_camera()
            overview.update_scene(sim.data, "overview")
            top = np.hstack([label(overview.render().copy(), "overview"), label(rgb, "wrist RGB")])
            xhat = None if args.gt_hold else pipeline.ctrl.xhat
            bottom = np.hstack([label(depth_to_colour(depth), "wrist depth"),
                                info_panel(sim, seed, args.level, status, q_meas, xhat, qd_true, history)])
            frame = np.vstack([top, bottom])

            if writer is not None:
                writer.append_data(frame)
            key = -1
            if not args.no_window:
                # Pace the live view to real time (recording runs as fast as possible).
                lag = elapsed - (time.perf_counter() - wall_start)
                if lag > 0:
                    time.sleep(lag)
                cv2.imshow("AVRC simulation", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            if key == ord("r") or sim.time >= next_reset:
                seed += 1
                q_meas = new_trial(seed)
                first_step = True
                next_reset = sim.time + args.reset_every
    finally:
        if writer is not None:
            writer.close()
        overview.close()
        sim.close()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
