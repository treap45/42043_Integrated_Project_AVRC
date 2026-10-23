"""Plot the cube EKF during TRACK in the default task (moving cube, noisy camera, calibrated camera model).

Runs the full pipeline for one seed and logs, every control period while TRACK runs, the EKF estimate and covariance
next to the simulator ground truth. Ground truth is read here only, in the logging hook, to score the filter; it never
reaches the pipeline.

Writes results/check_cube_ekf/track_seed<N>.png (six panels) and track_seed<N>.csv (one row per control period).

    python scripts/plot_cube_ekf.py              # seed 0
    python scripts/plot_cube_ekf.py --seed 3
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.control.arm_controller import ArmController  # noqa: E402
from src.estimation.cube_ekf import GATE  # noqa: E402
from src.model.ur3e import UR3eModel  # noqa: E402
from src.perception.camera_model import CameraModel  # noqa: E402
from src.sim import CameraNoiseConfig, CubeMotionConfig, TaskSim  # noqa: E402
from src.task.pick_place import STOP_SPEED, PickPlace  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "results/check_cube_ekf"

# Reference palette (dataviz skill, light mode): estimate = series 1, measurement marks = series 2.
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#9a9893"
GRID = "#e4e3df"
EST = "#2a78d6"
MEAS = "#eb6834"


def wrap90(a):
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def run(seed: int):
    sim = TaskSim(camera_noise=CameraNoiseConfig(), cube_motion=CubeMotionConfig())
    robot = UR3eModel()
    camera = CameraModel.from_calibration()
    sim.reset(seed, "L0")
    ctrl = ArmController(robot=robot, dt=sim.dt, y0=sim.read_joint_positions())
    rows, updates = [], []

    def log(pp: PickPlace) -> None:
        if pp.state != "TRACK" or pp.cube_ekf is None:
            return
        ekf, gt = pp.cube_ekf, sim.ground_truth()
        t = sim.data.time
        P_xy = ekf.P[:2, :2]
        yaw_true = 2 * np.arctan2(gt["cube_quat"][3], gt["cube_quat"][0])
        rows.append({
            "t": t,
            "x_est": ekf.x[0], "y_est": ekf.x[1], "yaw_est": ekf.x[2], "vx_est": ekf.x[3], "vy_est": ekf.x[4],
            "x_true": gt["cube_pos"][0], "y_true": gt["cube_pos"][1], "yaw_true": wrap90(yaw_true),
            "vx_true": gt["cube_vel"][0], "vy_true": gt["cube_vel"][1],
            "sx": np.sqrt(P_xy[0, 0]), "sy": np.sqrt(P_xy[1, 1]),
            "s_xy_max": np.sqrt(np.max(np.linalg.eigvalsh(P_xy))),
            "tcp_x": gt["tcp_pos"][0], "tcp_y": gt["tcp_pos"][1], "progress": pp.progress,
        })
        # TRACK appends one track_log entry per camera frame, before this hook runs.
        if len(pp.track_log) > len(updates):
            entry = pp.track_log[-1]
            nis = ekf.nis if entry["status"] == "accepted" else np.nan
            updates.append({"t": t, "status": entry["status"], "nis": nis})

    pp = PickPlace(sim, robot, ctrl, on_step=log, camera=camera)
    outcome = pp.run(timeout=60.0)
    true_stop = sim.cube_motion.stop_time
    declared_stop = pp.track_stop_time
    sim.close()
    return rows, updates, true_stop, declared_stop, outcome


def plot(rows, updates, true_stop, declared_stop, seed, path):
    d = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    t = d["t"]
    t_upd = np.array([u["t"] for u in updates])
    status = np.array([u["status"] for u in updates])
    nis = np.array([u["nis"] for u in updates], dtype=float)
    hover = t[np.argmax(d["progress"] >= 1.0)] if np.any(d["progress"] >= 1.0) else None

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    })
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), sharex=True)

    def events(ax, label=False):
        ax.axvline(true_stop, color=INK_2, ls="--", lw=1)
        if declared_stop is not None:
            ax.axvline(declared_stop, color=EST, ls=":", lw=1.2)
        if label:
            ax.text(true_stop, 1.0, " true stop", transform=ax.get_xaxis_transform(), color=INK_2, va="top", fontsize=8)
            if declared_stop is not None:
                ax.text(declared_stop, 0.88, " declared stop", transform=ax.get_xaxis_transform(), color=EST,
                        va="top", fontsize=8)

    for ax, key, sig, name in ((axes[0, 0], "x", "sx", "x"), (axes[0, 1], "y", "sy", "y")):
        est, true, s = d[f"{key}_est"] * 1e3, d[f"{key}_true"] * 1e3, d[sig] * 1e3
        ax.fill_between(t, est - 2 * s, est + 2 * s, color=EST, alpha=0.18, lw=0, label="EKF ±2σ")
        ax.plot(t, true, color=INK, lw=1.2, label="true")
        ax.plot(t, est, color=EST, label="EKF estimate")
        ax.set_ylabel(f"cube {name} (mm)")
        events(ax, label=(name == "x"))
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=8)

    err = np.hypot(d["x_est"] - d["x_true"], d["y_est"] - d["y_true"]) * 1e3
    ax = axes[1, 0]
    ax.fill_between(t, 0, 2 * d["s_xy_max"] * 1e3, color=EST, alpha=0.18, lw=0, label="EKF 2σ (largest axis)")
    ax.plot(t, err, color=EST, label="|position error|")
    ax.set_ylabel("position error (mm)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    events(ax)

    ax = axes[1, 1]
    ax.plot(t, np.hypot(d["vx_true"], d["vy_true"]) * 1e3, color=INK, lw=1.2, label="true")
    ax.plot(t, np.hypot(d["vx_est"], d["vy_est"]) * 1e3, color=EST, label="EKF estimate")
    ax.axhline(STOP_SPEED * 1e3, color=MUTED, ls="--", lw=1)
    ax.text(t[0], STOP_SPEED * 1e3, " stop threshold 5 mm/s", color=INK_2, va="bottom", fontsize=8)
    ax.set_ylabel("cube speed (mm/s)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower center", frameon=False, fontsize=8)
    events(ax)

    ax = axes[2, 0]
    ok = status == "accepted"
    ax.plot(t_upd[ok], nis[ok], "o", ms=3.5, color=MEAS, label=f"NIS per frame (mean {np.nanmean(nis):.2f})")
    for s_name, marker in (("rejected", "x"), ("none", "|"), ("reinit", "s")):
        sel = status == s_name
        if np.any(sel):
            ax.plot(t_upd[sel], np.full(sel.sum(), GATE), marker, ms=7, color=INK, label=f"{s_name} ({sel.sum()})")
    ax.axhline(4.0, color=MUTED, ls=":", lw=1)
    ax.axhline(GATE, color=MUTED, ls="--", lw=1)
    ax.text(t[0], 4.0, " consistent: mean 4", color=INK_2, va="bottom", fontsize=8)
    ax.text(t[0], GATE, " gate 13.28 (χ², 4 dof, 99 %)", color=INK_2, va="bottom", fontsize=8)
    ax.set_ylabel("NIS")
    ax.set_xlabel("time (s)")
    ax.set_ylim(0, GATE * 1.25)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    events(ax)

    ax = axes[2, 1]
    lag = np.hypot(d["tcp_x"] - d["x_true"], d["tcp_y"] - d["y_true"]) * 1e3
    ax.plot(t, lag, color=EST)
    if hover is not None:
        ax.axvspan(hover, t[-1], color=GRID, alpha=0.6, lw=0)
        ax.text(hover, 0.98, " hovering (s = 1)", transform=ax.get_xaxis_transform(), color=INK_2, va="top", fontsize=8)
        ax.set_ylim(0, max(10.0, 1.2 * np.max(lag[t >= hover])))
    ax.set_ylabel("TCP to cube, horizontal (mm)")
    ax.set_xlabel("time (s)")
    events(ax)

    fig.suptitle(f"Cube EKF during TRACK, seed {seed} (noisy camera, calibrated camera model; true values: "
                 "simulator, for evaluation only)", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return err, lag, hover, t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows, updates, true_stop, declared_stop, outcome = run(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"track_seed{args.seed}.png"
    csv_path = OUT_DIR / f"track_seed{args.seed}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    err, lag, hover, t = plot(rows, updates, true_stop, declared_stop, args.seed, png)
    d_v = np.array([[r["vx_est"] - r["vx_true"], r["vy_est"] - r["vy_true"]] for r in rows])
    nis = np.array([u["nis"] for u in updates], dtype=float)
    statuses = [u["status"] for u in updates]
    print(f"seed {args.seed}: pipeline {'OK' if outcome['ok'] else 'FAILED: ' + str(outcome['reason'])}")
    print(f"  TRACK {t[0]:.1f}-{t[-1]:.1f} s, {len(updates)} camera frames: "
          + ", ".join(f"{s} {statuses.count(s)}" for s in sorted(set(statuses))))
    print(f"  position error RMS {np.sqrt(np.mean(err**2)):.2f} mm, max {err.max():.2f} mm")
    print(f"  velocity error RMS {1e3 * np.sqrt(np.mean(np.sum(d_v**2, axis=1))):.2f} mm/s")
    print(f"  NIS mean {np.nanmean(nis):.2f} (consistent: 4)")
    if declared_stop is not None:
        print(f"  stop: true {true_stop:.2f} s, declared {declared_stop:.2f} s (delay {declared_stop - true_stop:.2f} s)")
    if hover is not None:
        moving = (t >= hover) & (t < true_stop)
        if np.any(moving):
            print(f"  TCP to cube while hovering above the moving cube: RMS {np.sqrt(np.mean(lag[moving]**2)):.2f} mm")
    print(f"  wrote {png.relative_to(OUT_DIR.parents[1])} and {csv_path.relative_to(OUT_DIR.parents[1])}")


if __name__ == "__main__":
    main()
