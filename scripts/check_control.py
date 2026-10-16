"""Closed-loop check of the joint controller: hold, point-to-point tracking and torque-impulse rejection.

Expected interface (src/control/arm_controller.py, written by the author):

    ctrl = ArmController(...)                   # constructed in make_controller() below; edit it to match
    tau = ctrl.step(y, q_ref, qd_ref)           # one control period: KF update, control law, KF predict
                                                # y: encoder reading (6,); returns the torque to apply (6,)
    ctrl.xhat                                   # (6, 3) estimate [q, qd, d] per joint
    ctrl.use_disturbance_ff                     # optional bool; if present, the hold test is also run without
                                                # the d_hat feed-forward for comparison

The controller only ever sees y and the reference. Ground truth is read here for evaluation, and a torque
impulse is injected into the plant through qfrc_applied (an external disturbance, not a control input).

Run from the repo root with the venv active:
    python scripts/check_control.py              # LQR, plots in results/check_control_lqr/
    python scripts/check_control.py --mode pd    # PD, plots in results/check_control/
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.ur3e import N_JOINTS, UR3eModel  # noqa: E402
from src.sim import TaskSim  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODE = "lqr"  # set from --mode in main()
JOINTS = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]

HOLD_S = 5.0
MOVE_DELTA = np.array([0.3, 0.2, -0.2, 0.2, 0.3, 0.5])  # rad, target = observe + delta
MOVE_T, MOVE_S = 2.0, 4.0  # quintic duration, total test length
IMPULSE_JOINT, IMPULSE_NM, IMPULSE_AT, IMPULSE_LEN, IMPULSE_S = 1, 40.0, 0.5, 0.1, 3.0

# Pass thresholds (evaluation only).
HOLD_TOL = 0.01  # rad, steady-state error after the first second
TRACK_RMS_TOL = 0.02  # rad, during and after the move
RECOVER_TOL, RECOVER_S = 0.01, 1.0  # back within RECOVER_TOL rad of the reference within RECOVER_S


def make_controller(robot: UR3eModel, dt: float, y0: np.ndarray):
    """Build the controller under test. Edit to match the ArmController constructor."""
    from src.control.arm_controller import ArmController

    return ArmController(robot=robot, dt=dt, y0=y0, mode=MODE)


def out_dir() -> Path:
    """PD plots keep the original folder; LQR plots go to check_control_lqr."""
    return RESULTS / ("check_control" if MODE == "pd" else "check_control_lqr")


def quintic(q0, q1, T, t):
    """Rest-to-rest quintic reference (test helper)."""
    s = np.clip(t / T, 0.0, 1.0)
    pos = 10 * s**3 - 15 * s**4 + 6 * s**5
    vel = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    return q0 + (q1 - q0) * pos, (q1 - q0) * vel


def run(robot, duration, reference, impulse=None, ff=None):
    """Closed loop for `duration` s. reference(t, q0) -> (q_ref, qd_ref). Returns a log of arrays."""
    sim = TaskSim()
    sim.reset(0, "L0")
    dt = sim.dt
    lo, hi = sim.model.actuator_ctrlrange[:N_JOINTS].T
    y = sim.read_joint_positions()
    ctrl = make_controller(robot, dt, y)
    if ff is not None:
        ctrl.use_disturbance_ff = ff
    q0 = sim.ground_truth()["q"]

    log = {k: [] for k in ("t", "q", "qd", "q_ref", "qd_hat", "d_hat", "tau", "sat")}
    for k in range(int(duration / dt)):
        t = k * dt
        if k > 0:
            y = sim.read_joint_positions()
        q_ref, qd_ref = reference(t, q0)
        tau = np.asarray(ctrl.step(y, q_ref, qd_ref), dtype=float)
        if impulse is not None:
            sim.data.qfrc_applied[:N_JOINTS] = impulse(t)
        sim.step(tau)
        gt = sim.ground_truth()
        xh = np.asarray(ctrl.xhat)
        log["t"].append(t)
        log["q"].append(gt["q"])
        log["qd"].append(gt["qd"])
        log["q_ref"].append(q_ref)
        log["qd_hat"].append(xh[:, 1])
        log["d_hat"].append(xh[:, 2])
        log["tau"].append(tau)
        log["sat"].append((tau <= lo + 1e-9) | (tau >= hi - 1e-9))
        if not np.all(np.isfinite(gt["q"])):
            break
    sim.close()
    out = {k: np.array(v) for k, v in log.items()}
    out["limits"] = hi
    return out


def summarise(name, log, after=1.0):
    e = log["q"] - log["q_ref"]
    m = log["t"] > after
    rms = np.sqrt(np.mean(e[m] ** 2, axis=0))
    final = np.abs(e[-1])
    sat = 100 * log["sat"].mean(axis=0)
    peak_tau = np.abs(log["tau"]).max(axis=0)
    print(f"  {name}")
    print(f"    RMS error after {after:g} s (rad) {rms}")
    print(f"    final |error| (rad)           {final}")
    print(f"    peak |tau| (N m)              {peak_tau}   limits {log['limits']}")
    print(f"    time saturated (%)            {sat}")
    return rms, final


def plot(name, log, extra_title=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = log["t"]
    fig, axes = plt.subplots(4, N_JOINTS, figsize=(20, 11), sharex=True)
    for j in range(N_JOINTS):
        axes[0, j].plot(t, log["q_ref"][:, j], "k--", lw=1, label="reference")
        axes[0, j].plot(t, log["q"][:, j], lw=1.2, label="true q")
        axes[0, j].set_title(JOINTS[j])
        axes[1, j].plot(t, 1000 * (log["q"][:, j] - log["q_ref"][:, j]), lw=1)
        axes[2, j].plot(t, log["qd"][:, j], lw=1, label="true qd")
        axes[2, j].plot(t, log["qd_hat"][:, j], lw=0.8, alpha=0.8, label="estimated qd")
        axes[3, j].plot(t, log["tau"][:, j], lw=1, label="tau")
        axes[3, j].plot(t, log["d_hat"][:, j], lw=1, label="d_hat")
        for s in (-1, 1):
            axes[3, j].axhline(s * log["limits"][j], color="r", ls=":", lw=1)
        axes[3, j].set_xlabel("t (s)")
    axes[0, 0].set_ylabel("q (rad)")
    axes[1, 0].set_ylabel("q - q_ref (mrad)")
    axes[2, 0].set_ylabel("qd (rad/s)")
    axes[3, 0].set_ylabel("N m")
    for r in (0, 2, 3):
        axes[r, 0].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{name} {extra_title}")
    fig.tight_layout()
    out_dir().mkdir(parents=True, exist_ok=True)
    path = out_dir() / f"{name}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main():
    global MODE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="lqr", choices=["lqr", "pd"], help="joint feedback law")
    MODE = p.parse_args().mode
    print(f"controller mode: {MODE}\n")
    np.set_printoptions(precision=4, suppress=True)
    try:
        robot = UR3eModel()
        results = {}

        print(f"== 1. Hold at the observe pose ({HOLD_S:g} s) ==")
        hold = lambda t, q0: (q0, np.zeros(N_JOINTS))  # noqa: E731
        log = run(robot, HOLD_S, hold)
        _, final = summarise("with controller defaults", log)
        results["hold"] = bool(np.all(np.isfinite(final)) and np.all(final < HOLD_TOL))
        print(f"    plot {plot('hold', log)}")
        probe = make_controller(robot, 0.002, np.zeros(N_JOINTS))
        if hasattr(probe, "use_disturbance_ff"):
            log_noff = run(robot, HOLD_S, hold, ff=False)
            summarise("without d_hat feed-forward (comparison)", log_noff)
            print(f"    plot {plot('hold_no_ff', log_noff, '(no disturbance feed-forward)')}")

        print(f"\n== 2. Point-to-point: observe + {MOVE_DELTA} rad in {MOVE_T:g} s ==")
        move = lambda t, q0: quintic(q0, q0 + MOVE_DELTA, MOVE_T, t)  # noqa: E731
        log = run(robot, MOVE_S, move)
        rms, final = summarise("quintic move", log, after=0.0)
        results["track"] = bool(np.all(rms < TRACK_RMS_TOL) and np.all(final < HOLD_TOL))
        print(f"    plot {plot('move', log)}")

        print(f"\n== 3. Torque impulse: {IMPULSE_NM:g} N m for {1000 * IMPULSE_LEN:g} ms on "
              f"{JOINTS[IMPULSE_JOINT]} at t = {IMPULSE_AT:g} s ==")

        def impulse(t):
            f = np.zeros(N_JOINTS)
            if IMPULSE_AT <= t < IMPULSE_AT + IMPULSE_LEN:
                f[IMPULSE_JOINT] = IMPULSE_NM
            return f

        log = run(robot, IMPULSE_S, hold, impulse=impulse)
        e = np.abs(log["q"][:, IMPULSE_JOINT] - log["q_ref"][:, IMPULSE_JOINT])
        after = log["t"] >= IMPULSE_AT
        peak = e[after].max()
        out = np.where(after & (e > RECOVER_TOL))[0]
        recover = (log["t"][out[-1]] - IMPULSE_AT) if len(out) else 0.0
        print(f"    peak deviation {peak:.4f} rad, back within {RECOVER_TOL} rad after {recover:.3f} s")
        results["impulse"] = bool(np.isfinite(peak) and recover < RECOVER_S and e[-1] < RECOVER_TOL)
        print(f"    plot {plot('impulse', log)}")

    except ImportError as exc:
        print(f"Controller not available yet: {exc}")
        return

    print("\n== Summary ==")
    for k, v in results.items():
        print(f"  {k:8s} {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
