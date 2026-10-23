"""Check src/estimation/cube_ekf.py: the Jacobian, and the filter on the moving cube with the arm held at observe.

1. H_cube against central finite differences of h_cube at random states and camera poses.
2. Moving cube (src/sim CubeMotionConfig), noisy wrist camera, calibrated camera model. The arm is held at the observe
   pose by the joint controller; the EKF predicts every control period and is updated at about 10 Hz from
   cube_measurement. Ground truth is read only to score the filter: NEES on the 5 states (consistent = 5), NIS
   (consistent = 4), position and velocity RMS error, and the stop test of docs/cube_ekf_design.md (speed below 5 mm/s
   and v^T P_v^-1 v below 5.99 for 10 consecutive updates): delay after the true stop, and false stops before it.

Run from the repo root with the venv active:
    python scripts/check_cube_ekf.py
    python scripts/check_cube_ekf.py --sigma-a 0.02 0.05 0.1    # sweep the process noise
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.control.arm_controller import ArmController  # noqa: E402
from src.estimation.cube_ekf import CubeEKF, H_cube, h_cube  # noqa: E402
from src.model.ur3e import UR3eModel  # noqa: E402
from src.perception.camera_model import CameraModel  # noqa: E402
from src.perception.cube_pose import cube_measurement, estimate_cube  # noqa: E402
from src.perception.projection import backproject, camera_to_world  # noqa: E402
from src.perception.segmentation import CUBE_HSV, color_mask, largest_component  # noqa: E402
from src.sim import CameraNoiseConfig, CubeMotionConfig, TaskSim  # noqa: E402

MJ_FROM_CV = np.diag([1.0, -1.0, -1.0])
UPDATE_PERIOD = 50  # control periods between camera frames (10 Hz at 500 Hz)
AFTER_STOP = 3.0  # s simulated after the true stop
STOP_SPEED = 0.005  # m/s
STOP_CHI2 = 5.99  # chi-square, 2 degrees of freedom, 95 %
STOP_COUNT = 10  # consecutive camera updates
N_SEEDS = 20


def wrap90(a):
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def check_jacobian(rng: np.random.Generator, n: int = 200) -> float:
    """Largest relative error between H_cube and central finite differences of h_cube."""
    K_cam = np.array([[625.0, 0.0, 319.5], [0.0, 625.0, 239.5], [0.0, 0.0, 1.0]])
    worst = 0.0
    for _ in range(n):
        x = np.array([rng.uniform(0.25, 0.4), rng.uniform(-0.15, 0.15), rng.uniform(-0.7, 0.7), 0.01, -0.01])
        # Camera above the cube looking roughly down, with a random tilt.
        cam_pos = np.array([x[0] + rng.uniform(-0.05, 0.05), x[1] + rng.uniform(-0.05, 0.05), rng.uniform(1.0, 1.2)])
        tilt = rng.uniform(-0.3, 0.3, 3)
        c, s = np.cos(tilt), np.sin(tilt)
        Rx = np.array([[1, 0, 0], [0, c[0], -s[0]], [0, s[0], c[0]]])
        Ry = np.array([[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]])
        Rz = np.array([[c[2], -s[2], 0], [s[2], c[2], 0], [0, 0, 1]])
        R_cv = Rz @ Ry @ Rx @ np.diag([1.0, -1.0, -1.0])  # optical axis pointing down
        H = H_cube(x, K_cam, cam_pos, R_cv)
        H_fd = np.zeros_like(H)
        for j in range(5):
            dx = np.zeros(5)
            dx[j] = 1e-6
            H_fd[:, j] = (h_cube(x + dx, K_cam, cam_pos, R_cv) - h_cube(x - dx, K_cam, cam_pos, R_cv)) / 2e-6
        worst = max(worst, float(np.max(np.abs(H - H_fd)) / max(np.max(np.abs(H_fd)), 1e-12)))
    return worst


def measure(sim, robot, camera, q_hat):
    """One camera frame -> (z, cam_pos, R_cv) or None, exactly as the pipeline would see it."""
    rgb, depth = sim.read_camera()
    mask = largest_component(color_mask(rgb, **CUBE_HSV), min_pixels=100)
    if mask is None:
        return None
    points_cv = backproject(mask, depth, camera.K)
    if len(points_cv) == 0:
        return None
    cam_pos, R_mj = camera.pose(robot, q_hat)
    points_w = camera_to_world(points_cv, cam_pos, R_mj)
    cube = estimate_cube(points_w)
    z = cube_measurement(points_cv, points_w, cube, camera.K)
    if z is None:
        return None
    return z, cube, cam_pos, R_mj @ MJ_FROM_CV


def run(sim, robot, camera, seed: int, sigma_a: float, R=None) -> dict:
    sim.reset(seed)
    motion = sim.cube_motion
    ctrl = ArmController(robot, sim.dt, sim.read_joint_positions())
    q_hold = ctrl.xhat[:, 0].copy()

    ekf = None
    nees, nis, pos_err, vel_err = [], [], [], []
    statuses = []
    stop_run = 0
    stop_declared = None
    false_stop = False
    k = 0
    while sim.time < motion.stop_time + AFTER_STOP:
        sim.step(ctrl.step(sim.read_joint_positions(), q_hold, np.zeros(6)))
        k += 1
        if ekf is not None:
            ekf.predict()
        if k % UPDATE_PERIOD:
            continue

        m = measure(sim, robot, camera, ctrl.xhat[:, 0].copy())
        if ekf is None:
            if m is not None and m[1].ok:
                ekf = CubeEKF(np.array([*m[1].pos[:2], m[1].yaw, 0.0, 0.0]), sim.dt, sigma_a=sigma_a, R=R)
            continue
        status = "none" if m is None else ekf.update(m[0], camera.K, m[2], m[3])
        statuses.append(status)
        if status == "accepted":
            nis.append(ekf.nis)

        # Score against ground truth (evaluation only).
        gt = sim.ground_truth()
        yaw_true = 2 * np.arctan2(gt["cube_quat"][3], gt["cube_quat"][0])
        e = np.array([*(ekf.position - gt["cube_pos"][:2]), wrap90(ekf.yaw - yaw_true),
                      *(ekf.velocity - gt["cube_vel"][:2])])
        nees.append(float(e @ np.linalg.solve(ekf.P, e)))
        pos_err.append(np.linalg.norm(e[:2]))
        vel_err.append(np.linalg.norm(e[3:]))

        stopped_now = np.linalg.norm(ekf.velocity) < STOP_SPEED and ekf.velocity_test() < STOP_CHI2
        stop_run = stop_run + 1 if stopped_now else 0
        if stop_run >= STOP_COUNT and stop_declared is None:
            if sim.time < motion.stop_time:
                false_stop = True
            else:
                stop_declared = sim.time

    rms = lambda a: float(np.sqrt(np.mean(np.square(a)))) if a else float("nan")  # noqa: E731
    return {
        "speed": float(np.linalg.norm(motion.velocity)), "stop_time": motion.stop_time,
        "nees": float(np.mean(nees)) if nees else float("nan"), "nis": float(np.mean(nis)) if nis else float("nan"),
        "pos_rms": rms(pos_err), "vel_rms": rms(vel_err),
        "delay": None if stop_declared is None else stop_declared - motion.stop_time, "false_stop": false_stop,
        "rejected": statuses.count("rejected") + statuses.count("reinit"), "missed": statuses.count("none"),
        "updates": len(statuses),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sigma-a", type=float, nargs="+", default=[0.01],
                   help="process noise (m/s^2); default 0.01 as in CubeEKF")
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--r-std", type=float, nargs=3, metavar=("UV_PX", "Z_M", "PSI_RAD"), default=None,
                   help="measurement std for u and v (px), Z (m), psi (rad)")
    p.add_argument("--nominal-camera", action="store_true",
                   help="nominal K and camera mount (the simulated mount is exact) instead of the calibration")
    args = p.parse_args()

    print("== 1. H_cube vs finite differences ==")
    worst = check_jacobian(np.random.default_rng(0))
    print(f"  largest relative error over 200 random states and camera poses: {worst:.2e}  "
          f"{'PASS' if worst < 1e-5 else 'FAIL'}")

    sim = TaskSim(camera_noise=CameraNoiseConfig(), cube_motion=CubeMotionConfig())
    robot = UR3eModel()
    camera = CameraModel.nominal(robot) if args.nominal_camera else CameraModel.from_calibration()
    R = None if args.r_std is None else np.diag(np.array([args.r_std[0], args.r_std[0], args.r_std[1], args.r_std[2]])**2)
    for sigma_a in args.sigma_a:
        print(f"\n== 2. Moving cube, arm at observe, sigma_a = {sigma_a:g} ==")
        rows = []
        for seed in range(args.seeds):
            r = run(sim, robot, camera, seed, sigma_a, R)
            rows.append(r)
            delay = "  never" if r["delay"] is None else f"{r['delay']:6.2f} s"
            print(f"  seed {seed:2d}: speed {r['speed']*100:3.1f} cm/s  NEES {r['nees']:6.2f}  NIS {r['nis']:5.2f}  "
                  f"pos RMS {r['pos_rms']*1e3:4.1f} mm  vel RMS {r['vel_rms']*1e3:4.1f} mm/s  stop delay {delay}  "
                  f"false stop {r['false_stop']!s:5s}  rejected {r['rejected']}/{r['updates']}  "
                  f"no measurement {r['missed']}")
        med = lambda key: float(np.nanmedian([r[key] for r in rows]))  # noqa: E731
        delays = [r["delay"] for r in rows if r["delay"] is not None]
        print(f"  median NEES {med('nees'):.2f} (consistent = 5), NIS {med('nis'):.2f} (consistent = 4), "
              f"pos RMS {med('pos_rms')*1e3:.1f} mm, vel RMS {med('vel_rms')*1e3:.1f} mm/s; "
              f"stop declared {len(delays)}/{len(rows)}, median delay "
              f"{np.median(delays) if delays else float('nan'):.2f} s, false stops {sum(r['false_stop'] for r in rows)}")
    sim.close()


if __name__ == "__main__":
    main()
