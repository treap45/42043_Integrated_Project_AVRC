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
from .sensors import EncoderConfig, JointEncoders, WristCamera

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = REPO_ROOT / "assets/scene.xml"

ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
N_ARM = len(ARM_JOINTS)


class TaskSim:
    def __init__(self, encoder_cfg: EncoderConfig = EncoderConfig(),
                 workspace: WorkspaceConfig = WorkspaceConfig(), scene_xml: Path = SCENE_XML):
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        # Exact depth: with MSAA the resolved depth comes from an off-centre sub-sample.
        self.model.vis.quality.offsamples = 0
        self.data = mujoco.MjData(self.model)
        self.workspace = workspace
        self.encoder_cfg = encoder_cfg
        self.encoders = JointEncoders(encoder_cfg, np.random.default_rng(0))
        self.camera = WristCamera(self.model)
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
        """Wrist camera: RGB (480, 640, 3) uint8 and depth (480, 640) float32 in metres."""
        return self.camera.render(self.data)

    # ---- ground truth: EVALUATION AND LOGGING ONLY ------------------------------------------
    def ground_truth(self) -> dict:
        m, d = self.model, self.data
        cam = self.camera.cam_id
        h = self.camera.height
        f = 0.5 * h / np.tan(np.deg2rad(m.cam_fovy[cam]) / 2)
        return {
            "q": d.qpos[:N_ARM].copy(),
            "qd": d.qvel[:N_ARM].copy(),
            "cube_pos": d.body("cube").xpos.copy(),
            "cube_quat": d.body("cube").xquat.copy(),
            "zone_pos": d.body("place_zone").xpos.copy(),
            "tcp_pos": d.site("tcp").xpos.copy(),
            "cam_pos": d.cam_xpos[cam].copy(),
            "cam_mat": d.cam_xmat[cam].reshape(3, 3).copy(),  # MuJoCo camera frame: -z forward, +y up
            "K": np.array([[f, 0, self.camera.width / 2], [0, f, h / 2], [0, 0, 1]]),
        }

    def close(self):
        self.camera.close()
