"""Check src/estimation/joint_kf.py against ground truth, and sweep the disturbance noise sigma_d.

Test harness, not the pipeline: the arm is held with MuJoCo's own bias forces (ground truth) and every
joint is driven by a known sinusoidal torque u, which is also fed to the filter. With gravity cancelled
exactly, the true disturbance is only inertia mismatch and coupling, d = J_i * qdd_i - u_i, which varies at
the excitation frequency; the random-walk disturbance model is therefore checked here for consistency of
q and qd, not for tracking d.

Run from the repo root with the venv active:
    python scripts/check_joint_kf.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.estimation.joint_kf import ArmObserver, discrete_process_noise  # noqa: E402
from src.model.joint_ss import JointSSModel, build_joint_models  # noqa: E402
from src.model.ur3e import N_JOINTS, UR3eModel  # noqa: E402
from src.sim import TaskSim  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "results/check_joint_kf"
JOINTS = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]
AMP = np.array([3.0, 3.0, 2.0, 0.8, 0.8, 0.5])  # N m
FREQ = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.7])  # Hz
DURATION, WARMUP = 4.0, 1.0  # s
SIGMA_D = [3.0, 10.0, 30.0, 100.0, 300.0]


def check_process_noise() -> bool:
    print("== 1. Van Loan process noise ==")
    js = JointSSModel(0.25, 0.002)
    Qc = np.diag([0.0, 0.05**2, 30.0**2])
    Qd = discrete_process_noise(js.A, Qc, js.dt)
    # Reference: trapezoidal integral of exp(A s) Qc exp(A s)^T over [0, dt].
    s = np.linspace(0, js.dt, 2001)
    Phi = [np.eye(3) + js.A * t + js.A @ js.A * t**2 / 2 for t in s]  # A is nilpotent: exact
    vals = np.array([P @ Qc @ P.T for P in Phi])
    Q_ref = np.trapezoid(vals, s, axis=0)
    ok = np.allclose(Qd, Q_ref, rtol=1e-6, atol=1e-18) and np.allclose(Qd, Qd.T) and np.all(np.linalg.eigvalsh(Qd) > -1e-18)
    print(f"  matches direct integration, symmetric, PSD: {'PASS' if ok else 'FAIL'}")
    return ok


def run(robot: UR3eModel, sigma_d: float) -> dict:
    sim = TaskSim()
    sim.reset(0, "L0")
    dt = sim.dt
    y0 = sim.read_joint_positions()
    models = build_joint_models(robot, y0, dt)
    obs = ArmObserver(models, y0, sigma_d=sigma_d)
    J = np.array([m.inertia for m in models])

    e_q, e_qd, e_fd, e_d, d_true_log, nees, nis = [], [], [], [], [], [], []
    y_prev = y0
    for k in range(int(DURATION / dt)):
        t = k * dt
        y = y0 if k == 0 else sim.read_joint_positions()
        if k > 0:
            obs.update(y)
            if t > WARMUP:
                nis.append(obs.nis)
        u = AMP * np.sin(2 * np.pi * FREQ * t)
        sim.step(sim.data.qfrc_bias[:N_JOINTS] + u)  # TEST ONLY: ground-truth gravity hold
        obs.predict(u)
        if t > WARMUP:
            gt = sim.ground_truth()
            xh, P = obs.xhat, obs.covariance
            eq, eqd = xh[:, 0] - gt["q"], xh[:, 1] - gt["qd"]
            e_q.append(eq)
            e_qd.append(eqd)
            e_fd.append((y - y_prev) / dt - gt["qd"])
            d_true = J * sim.data.qacc[:N_JOINTS] - u
            e_d.append(xh[:, 2] - d_true)
            d_true_log.append(d_true)
            e2 = np.stack([eq, eqd], axis=1)
            nees.append([e2[i] @ np.linalg.solve(P[i][:2, :2], e2[i]) for i in range(N_JOINTS)])
        y_prev = y
    sim.close()

    rms = lambda a: np.sqrt(np.mean(np.square(a), axis=0))  # noqa: E731
    return {"qd_rms": rms(e_qd), "fd_rms": rms(e_fd), "q_rms": rms(e_q), "d_rms": rms(e_d),
            "d_true_rms": rms(d_true_log), "nees": np.mean(nees, axis=0), "nis": np.mean(nis, axis=0)}


def plot(results: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = np.array(SIGMA_D)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for j, name in enumerate(JOINTS):
        axes[0].loglog(sig, [results[s]["nees"][j] for s in SIGMA_D], "o-", label=name)
        axes[1].loglog(sig, [results[s]["nis"][j] for s in SIGMA_D], "o-", label=name)
        axes[2].semilogx(sig, [results[s]["qd_rms"][j] for s in SIGMA_D], "o-", label=name)
    axes[0].axhline(2, color="k", ls="--", lw=1)
    axes[1].axhline(1, color="k", ls="--", lw=1)
    fd = np.mean([results[s]["fd_rms"] for s in SIGMA_D])
    axes[2].axhline(fd, color="k", ls="--", lw=1, label="finite difference")
    axes[0].set(title="NEES of [q, qd] (consistent = 2)", xlabel="sigma_d (N m / sqrt(s))")
    axes[1].set(title="NIS (consistent = 1)", xlabel="sigma_d (N m / sqrt(s))")
    axes[2].set(title="RMS joint velocity error (rad/s)", xlabel="sigma_d (N m / sqrt(s))")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "sigma_d_sweep.png"
    fig.savefig(path, dpi=120)
    return path


def main():
    np.set_printoptions(precision=3, suppress=True)
    ok_q = check_process_noise()

    print(f"\n== 2. Sinusoidal excitation, sigma_d sweep ({DURATION:g} s, first {WARMUP:g} s discarded) ==")
    robot = UR3eModel()
    results = {s: run(robot, s) for s in SIGMA_D}
    for s in SIGMA_D:
        r = results[s]
        print(f"  sigma_d {s:5g}:  qd RMS {r['qd_rms']}  NEES[q,qd] {r['nees']}  NIS {r['nis']}")
    print(f"  finite-difference qd RMS {results[SIGMA_D[0]]['fd_rms']}")

    default = results[30.0]
    ok_kf = (np.all(default["qd_rms"] < default["fd_rms"])
             and np.all((default["nees"] > 0.5) & (default["nees"] < 5))
             and np.all((default["nis"] > 0.5) & (default["nis"] < 2)))
    print(f"\n  default sigma_d = 30: KF beats finite differences on every joint, NEES and NIS in range: "
          f"{'PASS' if ok_kf else 'FAIL'}")
    print(f"  d tracking in this test (RMS error / RMS true d): {default['d_rms'] / default['d_true_rms']}  "
          "(not checked: d varies at the excitation frequency here)")
    print(f"  plot saved to {plot(results)}")

    print("\n== Summary ==")
    print(f"  process noise {'PASS' if ok_q else 'FAIL'}")
    print(f"  joint KF      {'PASS' if ok_kf else 'FAIL'}")


if __name__ == "__main__":
    main()
