"""The lab D435i through pyrealsense2: RGB with depth registered to it, in the OpenCV camera frame.

Depth is aligned to the colour stream, so the HSV mask, the back-projection and the intrinsics all use the RGB pixel
grid. The factory RGB intrinsics have zero distortion (configs/real_camera_intrinsics.json), as projection.py assumes.
"""

from dataclasses import dataclass

import numpy as np
import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 640, 480, 30
WARMUP_FRAMES = 30  # auto-exposure settles in about a second


@dataclass
class RGBDFrame:
    rgb: np.ndarray       # (H, W, 3) uint8, RGB order
    depth_m: np.ndarray   # (H, W) float32, Z depth in metres, 0 where missing
    timestamp_ms: float


class D435i:
    """Start with `with D435i() as cam:` so the pipeline is always stopped."""

    def __init__(self, width: int = WIDTH, height: int = HEIGHT, fps: int = FPS):
        self.width, self.height, self.fps = width, height, fps
        self._pipeline = None
        self._align = rs.align(rs.stream.color)
        self.K = None
        self.dist = None
        self.depth_scale = None
        self.serial = None

    def __enter__(self) -> "D435i":
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)
        self.serial = profile.get_device().get_info(rs.camera_info.serial_number)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]])
        self.dist = np.asarray(intr.coeffs, dtype=float)
        for _ in range(WARMUP_FRAMES):
            self._pipeline.wait_for_frames()
        return self

    def __exit__(self, *exc) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def capture(self, timeout_ms: int = 2000) -> RGBDFrame:
        frames = self._align.process(self._pipeline.wait_for_frames(timeout_ms))
        colour, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not colour or not depth:
            raise RuntimeError("missing colour or depth frame")
        rgb = np.asanyarray(colour.get_data()).copy()
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        return RGBDFrame(rgb=rgb, depth_m=depth_m, timestamp_ms=colour.get_timestamp())
