"""Test the automatic eye-to-hand collection (src/real/auto_handeye.py) with a fake robot and camera; no hardware.

The fake robot moves through moveJ on the controller-side model's kinematics; the fake camera renders the marker
at its true pose with image noise. Scenarios:
    normal      full collection and solve; T_base_cam against the truth, path limits, back at the start pose
    no marker   marker out of view at the start: refused before any move
    stop key    stop asked for in the image window during a move: stopJ, and no further move
    ctrl-c      Ctrl+C during a move: stopJ, and no further move
    protective  protective stop during a move: stopped, and no further move

    python scripts/check_auto_handeye.py
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.ur3e import UR3eModel  # noqa: E402
from src.real.aruco import dictionary  # noqa: E402
from src.real.auto_handeye import AT_TARGET_RAD, AutoCollector, AutoStop, ViewLimits  # noqa: E402
from src.real.camera import RGBDFrame  # noqa: E402
from src.real.frames import invert_T, make_T, rotation_angle_deg, sim_world_from_rtde  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIDE = 0.07
TIME_SCALE = 10.0  # the fake arm moves this much faster than commanded, to keep the test short
Q0 = np.radians([-47.3, -121.6, 82.6, -64.3, -76.1, -45.7])
PASS_TABLE_MM = 5.0  # half the RG2 tolerance on a 60 mm cube held in an 80 mm opening (10 mm each side)


def look_at(eye, target):
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


class World:
    """The truth: robot state, camera pose and marker on the flange."""

    def __init__(self, marker_in_view=True):
        self.robot = UR3eModel()
        self.T_b_w = invert_T(sim_world_from_rtde())
        self.q = Q0.copy()
        p0 = self.flange(Q0)
        eye = p0[:3, 3] + np.array([-0.55, 0.55, 0.15])
        self.T_base_cam = make_T(look_at(eye, p0[:3, 3] + 0.05 * p0[:3, 2]), eye)
        # Marker 80 mm along the flange z, facing the camera at about 25 degrees at the start pose.
        centre = p0[:3, 3] + 0.08 * p0[:3, 2]
        to_cam = (eye - centre) / np.linalg.norm(eye - centre)
        R_face = look_at(centre, centre + to_cam)
        tilt, _ = cv2.Rodrigues(np.array([0.0, np.radians(25.0), 0.0]))
        self.T_ee_marker = invert_T(p0) @ make_T(R_face @ tilt, centre)
        if not marker_in_view:
            self.T_ee_marker = self.T_ee_marker @ make_T(np.diag([1.0, -1.0, -1.0]), [0.0, 0.0, 0.0])  # faces away

    def flange(self, q):
        p, R = self.robot.site_pose(q, "attachment_site")
        return self.T_b_w @ make_T(R, p)


class FakeRobot:
    def __init__(self, world: World, protective_after_moves=None):
        self.w = world
        self.target, self.speed = None, 0.0
        self.t = self.t_command = time.monotonic()
        self.moves = 0
        self.stops = 0
        self.protective_after = protective_after_moves
        self.protective = False
        self.min_z = np.inf
        self.max_travel = 0.0
        self.p0 = world.flange(Q0)[:3, 3]

    def _advance(self):
        now = time.monotonic()
        dt, self.t = now - self.t, now
        if self.target is None:
            return
        d = self.target - self.w.q
        step = self.speed * TIME_SCALE * dt
        if np.max(np.abs(d)) <= step:
            self.w.q = self.target.copy()
            self.target = None
        else:
            self.w.q = self.w.q + d / np.max(np.abs(d)) * step
        T = self.w.flange(self.w.q)
        self.min_z = min(self.min_z, T[2, 3], T[2, 3] + 0.23 * T[2, 2])
        self.max_travel = max(self.max_travel, float(np.linalg.norm(T[:3, 3] - self.p0)))
        if self.protective_after is not None and self.moves > self.protective_after:
            self.protective = True
            self.target = None

    # RTDE receive
    def getActualQ(self):
        self._advance()
        return self.w.q.tolist()

    def getActualQd(self):
        self._advance()
        return [0.1] * 6 if self.target is not None else [0.0] * 6

    def getActualTCPPose(self):
        self._advance()
        T = self.w.flange(self.w.q)
        return [*T[:3, 3], *cv2.Rodrigues(T[:3, :3])[0].ravel()]

    def isProtectiveStopped(self):
        self._advance()
        return self.protective

    def isEmergencyStopped(self):
        return False

    # RTDE control
    def isJointsWithinSafetyLimits(self, q):
        return True

    def moveJ(self, q, speed, acceleration, asynchronous):
        assert asynchronous, "moves must be asynchronous so they can be stopped"
        assert speed <= 0.5, "too fast"
        self._advance()
        self.moves += 1
        self.t_command = time.monotonic()
        self.target, self.speed = np.asarray(q, dtype=float), speed
        return True

    def getAsyncOperationProgress(self):
        self._advance()
        # As on a real controller, the operation is not reported as running just after the command.
        if time.monotonic() - self.t_command < 0.05:
            return -1
        return 0 if self.target is not None else -1

    def stopJ(self, a=2.0):
        self._advance()
        self.target = None
        self.stops += 1


class FakeCamera:
    width, height = 640, 480
    depth_scale = 0.001
    serial = "fake"

    def __init__(self, world: World, K, seed=0):
        self.w, self.K, self.dist = world, K, np.zeros(5)
        self.rng = np.random.default_rng(seed)
        mk = cv2.aruco.generateImageMarker(dictionary(), 0, 600)
        self.tex = np.full((800, 800), 255, np.uint8)
        self.tex[100:700, 100:700] = mk
        self.m_per_px = SIDE / 600.0

    def capture(self):
        T_cm = invert_T(self.w.T_base_cam) @ self.w.flange(self.w.q) @ self.w.T_ee_marker
        img = np.full((self.height, self.width), 120, np.uint8)
        depth = np.zeros((self.height, self.width), np.float32)
        ray = T_cm[:3, 3] / np.linalg.norm(T_cm[:3, 3])
        if T_cm[2, 3] > 0.1 and T_cm[:3, 2] @ ray < 0:
            src = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
            local = np.column_stack([(src[:, 0] - 400) * self.m_per_px, (400 - src[:, 1]) * self.m_per_px,
                                     np.zeros(4)])
            pts = (T_cm[:3, :3] @ local.T).T + T_cm[:3, 3]
            uv = (self.K @ pts.T).T
            H = cv2.getPerspectiveTransform(src, np.float32(uv[:, :2] / uv[:, 2:]))
            mask = cv2.warpPerspective(np.full((800, 800), 255, np.uint8), H, (self.width, self.height))
            warped = cv2.warpPerspective(self.tex, H, (self.width, self.height), flags=cv2.INTER_LINEAR)
            img[mask > 127] = warped[mask > 127]
            depth[mask > 127] = T_cm[2, 3]
        img = cv2.GaussianBlur(img, (3, 3), 0.6)
        noisy = np.clip(img + self.rng.normal(0.0, 2.0, img.shape), 0, 255).astype(np.uint8)
        return RGBDFrame(rgb=cv2.cvtColor(noisy, cv2.COLOR_GRAY2RGB), depth_m=depth, timestamp_ms=0.0)


def collector(world, robot, K, out, **kw):
    return AutoCollector(robot, robot, FakeCamera(world, K), world.robot, SIDE, ViewLimits(), out, log=lambda m: None, **kw)


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")
    return ok


def main():
    K = np.asarray(json.loads((ROOT / "configs/real_camera_intrinsics.json").read_text())["K"])
    all_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        for seed in range(3):
            print(f"normal, seed {seed}")
            w = World()
            r = FakeRobot(w)
            out = tmp / f"normal{seed}"
            t = time.perf_counter()
            collector(w, r, K, out, n_views=25, seed=seed).run()
            extr = tmp / f"extrinsics{seed}.json"
            done = subprocess.run([sys.executable, str(ROOT / "scripts/calibrate_eye_to_hand.py"), str(out), "--auto-drop",
                                   "--out", str(extr)], capture_output=True, text=True)
            views = json.loads((out / "views.json").read_text())
            all_ok &= check("collected", len(views) >= 15, f"{len(views)} views, {r.moves} moves, "
                                                           f"{time.perf_counter() - t:.0f} s with the arm "
                                                           f"{TIME_SCALE:.0f}x faster")
            all_ok &= check("solver passed and saved", done.returncode == 0 and extr.exists(),
                            "" if done.returncode == 0 else done.stdout[-600:])
            if extr.exists():
                T = np.asarray(json.loads(extr.read_text())["T_base_cam"])
                e_p = 1000 * np.linalg.norm(T[:3, 3] - w.T_base_cam[:3, 3])
                e_r = rotation_angle_deg(T[:3, :3].T @ w.T_base_cam[:3, :3])
                # What the pipeline uses: where a point on the table lands when seen through the calibration.
                pts = np.array([[x, y, 0.06, 1.0] for x in np.linspace(-0.43, -0.20, 6)
                                for y in np.linspace(-0.15, 0.15, 6)])
                moved = (T @ invert_T(w.T_base_cam) @ pts.T).T
                e_w = 1000 * np.linalg.norm((moved - pts)[:, :3], axis=1)
                all_ok &= check("table points seen through the calibration", e_w.max() < PASS_TABLE_MM,
                                f"{np.median(e_w):.2f} mm median, {e_w.max():.2f} mm max")
                print(f"       camera pose itself: {e_p:.2f} mm, {e_r:.3f} deg; reprojection "
                      f"{json.loads(extr.read_text())['reprojection_px_mean']:.2f} px")
            all_ok &= check("back at the start pose", np.max(np.abs(w.q - Q0)) < AT_TARGET_RAD)
            all_ok &= check("limits held along the moves", r.min_z >= 0.05 - 1e-3 and r.max_travel <= 0.15 + 1e-3,
                            f"lowest point {1000 * r.min_z:.0f} mm above the table, "
                            f"flange travel {1000 * r.max_travel:.0f} mm")

        print("no marker at the start")
        w = World(marker_in_view=False)
        r = FakeRobot(w)
        try:
            collector(w, r, K, tmp / "none").run()
            all_ok &= check("refused", False)
        except AutoStop as exc:
            all_ok &= check("refused before any move", r.moves == 0, str(exc))

        print("stop key during a move")
        w = World()
        r = FakeRobot(w)
        frames = {"n": 0}

        def stop_key(rgb, corners):
            frames["n"] += 1
            return r.moves >= 3 and r.target is not None

        try:
            collector(w, r, K, tmp / "key", on_frame=stop_key).run()
            all_ok &= check("stopped", False)
        except AutoStop as exc:
            moves = r.moves
            all_ok &= check("stopJ and no further move", r.stops == 1 and moves == 3 and exc.halt,
                            f"{moves} moves, {r.stops} stopJ")

        print("Ctrl+C during a move")
        w = World()
        r = FakeRobot(w)

        def ctrl_c(rgb, corners):
            if r.moves >= 2 and r.target is not None:
                raise KeyboardInterrupt
            return False

        try:
            collector(w, r, K, tmp / "ctrlc", on_frame=ctrl_c).run()
            all_ok &= check("stopped", False)
        except KeyboardInterrupt:
            all_ok &= check("stopJ and no further move", r.stops == 1 and r.moves == 2, f"{r.moves} moves, {r.stops} stopJ")

        print("protective stop during a move")
        w = World()
        r = FakeRobot(w, protective_after_moves=2)
        try:
            collector(w, r, K, tmp / "protective").run()
            all_ok &= check("stopped", False)
        except AutoStop as exc:
            all_ok &= check("halted with no further move", exc.halt and r.moves == 3, f"{r.moves} moves: {exc}")

    print("\nPASS" if all_ok else "\nFAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
