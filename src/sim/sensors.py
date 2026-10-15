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


@dataclass(frozen=True)
class CameraNoiseConfig:
    """Wrist RGB-D noise model, applied on top of the exact render.

    Defaults are placeholders of the order of a RealSense D435 (depth accuracy within 2% at 2 m, 1 mm depth
    unit, roughly 0.1 m minimum range); they are design parameters to be justified (or changed) by the author.

    Depth (z along the optical axis, metres), in this order:
      1. Gaussian error with std  sigma_z = k_axial * z^2 / max(cos(theta), cos_min),  where theta is the angle
         between the viewing ray and the surface normal. Stereo depth error grows with z^2 (triangulation)
         and with grazing incidence.
      2. Pixels on depth discontinuities (jump > edge_jump to a 4-neighbour) are dropped with probability
         edge_dropout, as stereo matching fails at object edges.
      3. Readings outside [min_range, max_range] are invalid.
      4. Quantisation to depth_resolution.
    Invalid pixels are returned as 0, like a RealSense depth frame.

    RGB: additive Gaussian noise with std rgb_std (0-255 scale), then clipped and rounded.
    """

    k_axial: float = 0.01  # 1/m; sigma_z = 1.4 mm at the observe distance 0.37 m, 40 mm at 2 m
    cos_min: float = np.cos(np.deg2rad(75.0))
    edge_jump: float = 0.01  # m
    edge_dropout: float = 0.5
    min_range: float = 0.105  # m
    max_range: float = 3.0  # m
    depth_resolution: float = 0.001  # m
    rgb_std: float = 2.0


class CameraNoise:
    """Applies CameraNoiseConfig to one RGB-D frame. Needs the true intrinsics, so it lives on the plant side."""

    def __init__(self, cfg: CameraNoiseConfig, K: np.ndarray, rng: np.random.Generator):
        self.cfg = cfg
        self.K = K
        self.rng = rng

    def _cos_incidence(self, z: np.ndarray) -> np.ndarray:
        """|cos| of the angle between each pixel's viewing ray and the surface normal from the depth image."""
        h, w = z.shape
        u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        rays = np.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], np.ones_like(u)], -1)
        points = rays * z[..., None]
        normal = np.cross(np.gradient(points, axis=1), np.gradient(points, axis=0))
        denom = np.linalg.norm(normal, axis=-1) * np.linalg.norm(rays, axis=-1)
        return np.abs(np.sum(normal * rays, axis=-1)) / np.maximum(denom, 1e-12)

    def apply(self, rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg, rng = self.cfg, self.rng
        z = depth.astype(float)

        cos_theta = np.maximum(self._cos_incidence(z), cfg.cos_min)
        noisy = z + rng.normal(0.0, 1.0, z.shape) * cfg.k_axial * z**2 / cos_theta

        jump = np.zeros(z.shape, dtype=bool)
        dv = np.abs(np.diff(z, axis=0)) > cfg.edge_jump
        du = np.abs(np.diff(z, axis=1)) > cfg.edge_jump
        jump[1:] |= dv
        jump[:-1] |= dv
        jump[:, 1:] |= du
        jump[:, :-1] |= du
        dropped = jump & (rng.random(z.shape) < cfg.edge_dropout)

        invalid = dropped | (noisy < cfg.min_range) | (noisy > cfg.max_range)
        if cfg.depth_resolution > 0:
            noisy = np.round(noisy / cfg.depth_resolution) * cfg.depth_resolution
        noisy[invalid] = 0.0

        rgb_noisy = rgb.astype(float) + rng.normal(0.0, cfg.rgb_std, rgb.shape)
        return np.clip(np.round(rgb_noisy), 0, 255).astype(np.uint8), noisy.astype(np.float32)


class WristCamera:
    """Renders the wrist camera. Returns RGB (H, W, 3) uint8 and linear metric depth (H, W) float32.

    render() is exact; TaskSim adds CameraNoise on top when it is configured.
    Intrinsics are deliberately not exposed: they are to be found by calibration.
    """

    def __init__(self, model: mujoco.MjModel, name: str = "wrist_cam"):
        self.cam_id = model.camera(name).id
        self.width, self.height = (int(v) for v in model.cam_resolution[self.cam_id])
        self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        # True intrinsics, plant side only. OpenCV pixel convention: pixel centres at integer coordinates, so
        # the principal point of the rendered image is ((W - 1) / 2, (H - 1) / 2).
        f = 0.5 * self.height / np.tan(np.deg2rad(model.cam_fovy[self.cam_id]) / 2)
        self._K_true = np.array([[f, 0, (self.width - 1) / 2], [0, f, (self.height - 1) / 2], [0, 0, 1]])

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
