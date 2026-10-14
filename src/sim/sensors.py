"""Simulated sensors: joint encoders and the wrist RGB-D camera.

These model the measurement process on the plant side. Estimators must only ever see what these
classes return, never data.qpos / data.qvel.
"""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class EncoderConfig:
    """Joint encoder model: additive Gaussian noise, then quantisation to the encoder resolution.

    Defaults are placeholders chosen so that finite-differencing q is visibly noisy; they are design
    parameters to be justified (or changed) by the author.
    """

    noise_std: float = 5e-4  # rad
    resolution: float = 2 * np.pi / 2**14  # rad per count (14-bit absolute encoder, ~0.022 deg)


class JointEncoders:
    """Measures the six arm joint angles."""

    def __init__(self, cfg: EncoderConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def read(self, q_true: np.ndarray) -> np.ndarray:
        q = q_true + self.rng.normal(0.0, self.cfg.noise_std, size=q_true.shape)
        res = self.cfg.resolution
        return np.round(q / res) * res if res > 0 else q


class WristCamera:
    """Renders the wrist camera. Returns RGB (H, W, 3) uint8 and linear metric depth (H, W) float32.

    Depth is noise-free here; a view-dependent depth noise model can be added on top of render().
    Intrinsics are deliberately not exposed: they are to be found by calibration.
    """

    def __init__(self, model: mujoco.MjModel, name: str = "wrist_cam"):
        self.cam_id = model.camera(name).id
        self.width, self.height = (int(v) for v in model.cam_resolution[self.cam_id])
        self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)

    def render(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        r = self._renderer
        r.disable_depth_rendering()
        r.update_scene(data, self.cam_id)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, self.cam_id)
        depth = r.render().copy()
        r.disable_depth_rendering()
        return rgb, depth

    def close(self):
        self._renderer.close()
