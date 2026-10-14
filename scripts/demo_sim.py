"""Live simulation demo: overview camera, wrist RGB, wrist depth and encoder readings in one window.

The scene is re-randomised every --reset-every seconds (or press "r"); press "q" or Esc to quit.

Run from the repo root with the venv active:
    python scripts/demo_sim.py                      # live window, L0
    python scripts/demo_sim.py --level L2 --seed 7
    python scripts/demo_sim.py --record videos/demo.mp4 --duration 10 --no-window

Plug the pipeline in at controller() below. Until then the arm has no controller and falls under
gravity. --gt-hold holds the arm with MuJoCo's own bias forces (ground truth) for checking the scene
only; it must not be used for anything that is assessed.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim import N_ARM, TaskSim  # noqa: E402

FPS = 15
PANEL_W, PANEL_H = 640, 480
DEPTH_RANGE = (0.15, 0.8)  # metres, for the colour map


def controller(sim: TaskSim, q_meas: np.ndarray) -> np.ndarray:
    """Placeholder: replace with the estimator + controller. Must use only sensor readings."""
    return np.zeros(N_ARM)


def depth_to_colour(depth: np.ndarray) -> np.ndarray:
    lo, hi = DEPTH_RANGE
    d = np.clip((depth - lo) / (hi - lo), 0, 1)
    img = cv2.applyColorMap((255 * (1 - d)).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (len(text) * 11 + 12, 28), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def info_panel(sim: TaskSim, q_meas: np.ndarray, seed: int, level: str, gt_hold: bool) -> np.ndarray:
    img = np.full((PANEL_H, PANEL_W, 3), 30, np.uint8)
    lines = [f"t = {sim.time:6.2f} s   seed {seed}   level {level}",
             f"occluders: {len(sim.trial.occluders)}",
             "",
             "joint encoders (deg):"]
    names = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]
    lines += [f"  {n:7s} {np.rad2deg(v):9.3f}" for n, v in zip(names, q_meas)]
    if gt_hold:
        lines += ["", "TEST MODE: arm held with ground truth"]
    for i, text in enumerate(lines):
        colour = (255, 120, 120) if text.startswith("TEST") else (230, 230, 230)
        cv2.putText(img, text, (16, 40 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 1, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--level", default="L0", choices=["L0", "L1", "L2"])
    p.add_argument("--reset-every", type=float, default=5.0, help="seconds between re-randomisations")
    p.add_argument("--duration", type=float, default=float("inf"), help="stop after this many sim seconds")
    p.add_argument("--record", type=Path, help="write an mp4 of the composite view")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--gt-hold", action="store_true", help="TEST ONLY: hold the arm with ground-truth bias forces")
    args = p.parse_args()

    sim = TaskSim()
    seed = args.seed
    sim.reset(seed, args.level)
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
    try:
        while elapsed < args.duration:
            for _ in range(steps_per_frame):
                q_meas = sim.read_joint_positions()
                tau = sim.data.qfrc_bias[:N_ARM].copy() if args.gt_hold else controller(sim, q_meas)
                sim.step(tau)
            elapsed += steps_per_frame * sim.dt

            rgb, depth = sim.read_camera()
            overview.update_scene(sim.data, "overview")
            top = np.hstack([label(overview.render().copy(), "overview"), label(rgb, "wrist RGB")])
            bottom = np.hstack([label(depth_to_colour(depth), "wrist depth"),
                                info_panel(sim, q_meas, seed, args.level, args.gt_hold)])
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
                sim.reset(seed, args.level)
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
