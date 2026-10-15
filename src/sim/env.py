"""Simulation interface (hardware abstraction layer) for the task.

The pipeline talks to the robot only through:
    actuation   step(tau), set_gripper(closure)
    sensing     read_joint_positions(), read_camera()
Everything simulator-internal (true joint state, object poses, camera intrinsics) is available only
through ground_truth(), which is for evaluation and logging and must never feed the estimator or
controller.
"""

from pathlib import Path

import mujoco
import numpy as np

from .randomize import TrialConfig, WorkspaceConfig, apply_trial, sample_trial
from .sensors import CameraNoise, CameraNoiseConfig, EncoderConfig, JointEncoders, WristCamera

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = REPO_ROOT / "assets/scene.xml"

# Checkerboard on the calib_board body (assets/scene.xml, texture from scripts/make_calib_board.py).
CALIB_SQUARE = 0.02  # m
CALIB_SQUARES = (10, 7)  # along the board's x and y axes; inner corners (9, 6), asymmetric so the order is unique
CALIB_THICKNESS = 0.003  # m; the board frame is at the box centre, the printed face at z = +CALIB_THICKNESS / 2
PARK_BOARD = np.array([0.6, 0.9, -1.0])

ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
N_ARM = len(ARM_JOINTS)


class TaskSim:
    def __init__(self, encoder_cfg: EncoderConfig = EncoderConfig(),
                 workspace: WorkspaceConfig = WorkspaceConfig(), scene_xml: Path = SCENE_XML,
                 camera_noise: CameraNoiseConfig | None = None):
        """camera_noise=None gives exact RGB-D frames; pass CameraNoiseConfig() for the noisy camera."""
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        # Exact depth: with MSAA the resolved depth comes from an off-centre sub-sample.
        self.model.vis.quality.offsamples = 0
        self.data = mujoco.MjData(self.model)
        self.workspace = workspace
        self.encoder_cfg = encoder_cfg
        self.encoders = JointEncoders(encoder_cfg, np.random.default_rng(0))
        self.camera = WristCamera(self.model)
        self.camera_noise_cfg = camera_noise
        self.camera_noise: CameraNoise | None = None
        self._grip_act = self.model.actuator("gripper_fingers_actuator").id
        self._key = self.model.key("observe").id
        self.trial: TrialConfig | None = None
        self.reset(seed=0)

    # ---- trial management -------------------------------------------------------------------
    def reset(self, seed: int, level: str = "L0") -> TrialConfig:
        """Start a new trial: arm at the "observe" keyframe, gripper open, scene randomised from seed."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._key)
        self.trial = sample_trial(seed, level, self.workspace)
        apply_trial(self.model, self.data, self.trial)
        # Sensor noise is seeded per trial too, so a failure can be replayed exactly.
        self.encoders = JointEncoders(self.encoder_cfg, np.random.default_rng([seed, 1]))
        if self.camera_noise_cfg is not None:
            self.camera_noise = CameraNoise(self.camera_noise_cfg, self.camera._K_true,
                                            np.random.default_rng([seed, 2]))
            self._display_noise = CameraNoise(self.camera_noise_cfg, self.camera._K_true,
                                              np.random.default_rng([seed, 3]))
        mujoco.mj_forward(self.model, self.data)
        return self.trial

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    # ---- actuation --------------------------------------------------------------------------
    def step(self, tau: np.ndarray, n_steps: int = 1) -> None:
        """Apply joint torques (N m, clipped to the actuator limits) for n_steps physics steps."""
        self.data.ctrl[:N_ARM] = tau
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def set_gripper(self, closure: float) -> None:
        """0 = fully open, 1 = fully closed."""
        lo, hi = self.model.actuator_ctrlrange[self._grip_act]
        self.data.ctrl[self._grip_act] = lo + np.clip(closure, 0.0, 1.0) * (hi - lo)

    # ---- sensing ----------------------------------------------------------------------------
    def read_joint_positions(self) -> np.ndarray:
        """Encoder reading of the six arm joints (noisy, quantised)."""
        return self.encoders.read(self.data.qpos[:N_ARM].copy())

    def read_camera(self) -> tuple[np.ndarray, np.ndarray]:
        """Wrist camera: RGB (480, 640, 3) uint8 and depth (480, 640) float32 in metres
        (0 marks an invalid pixel when camera noise is on)."""
        rgb, depth = self.camera.render(self.data)
        if self.camera_noise is not None:
            rgb, depth = self.camera_noise.apply(rgb, depth)
        return rgb, depth

    def read_camera_for_display(self) -> tuple[np.ndarray, np.ndarray]:
        """Same as read_camera() but with noise from a separate RNG, so live views and recordings do not
        change the noise the pipeline sees."""
        rgb, depth = self.camera.render(self.data)
        if self.camera_noise is not None:
            rgb, depth = self._display_noise.apply(rgb, depth)
        return rgb, depth

    # ---- calibration setup ------------------------------------------------------------------
    def place_calib_board(self, pos, quat=(1.0, 0.0, 0.0, 0.0), clear_table: bool = True) -> None:
        """Put the checkerboard at pos (board centre, m) with orientation quat (w, x, y, z), like placing a
        printed board by hand. clear_table moves the cube off the table and parks the place zone and
        occluders so they do not cover the board. Call reset() to return to a task trial.

        A flat board on the table has its centre at z = TABLE_Z + CALIB_THICKNESS / 2 = 0.7515."""
        m, d = self.model, self.data
        board = m.body("calib_board").mocapid[0]
        d.mocap_pos[board] = pos
        d.mocap_quat[board] = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
        if clear_table:
            cube = m.joint("cube_joint").qposadr[0]
            d.qpos[cube:cube + 7] = [0.0, -0.9, 0.025, 1.0, 0.0, 0.0, 0.0]  # on the floor, out of view
            dof = m.joint("cube_joint").dofadr[0]
            d.qvel[dof:dof + 6] = 0.0
            for i, name in enumerate(["place_zone", "occluder_0", "occluder_1", "occluder_2"]):
                d.mocap_pos[m.body(name).mocapid[0]] = [0.2 * i, 0.9, -1.0]
        mujoco.mj_forward(m, d)

    # ---- ground truth: EVALUATION AND LOGGING ONLY ------------------------------------------
    def ground_truth(self) -> dict:
        """True state. K uses the OpenCV pixel convention (pixel centres at integer coordinates), so the
        principal point is ((W - 1) / 2, (H - 1) / 2) = (319.5, 239.5), checked against the rendered
        checkerboard corners and against mj_ray (scripts/check_calib_board.py)."""
        m, d = self.model, self.data
        cam = self.camera.cam_id
        board = m.body("calib_board").id
        flange = m.body("flange").id
        return {
            "q": d.qpos[:N_ARM].copy(),
            "qd": d.qvel[:N_ARM].copy(),
            "cube_pos": d.body("cube").xpos.copy(),
            "cube_quat": d.body("cube").xquat.copy(),
            "zone_pos": d.body("place_zone").xpos.copy(),
            "tcp_pos": d.site("tcp").xpos.copy(),
            "cam_pos": d.cam_xpos[cam].copy(),
            "cam_mat": d.cam_xmat[cam].reshape(3, 3).copy(),  # MuJoCo camera frame: -z forward, +y up
            "K": self.camera._K_true.copy(),
            "flange_pos": d.xpos[flange].copy(),
            "flange_mat": d.xmat[flange].reshape(3, 3).copy(),
            "board_pos": d.xpos[board].copy(),  # board centre; printed face at z = +CALIB_THICKNESS / 2
            "board_mat": d.xmat[board].reshape(3, 3).copy(),
        }

    def close(self):
        self.camera.close()
