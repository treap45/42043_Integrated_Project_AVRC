"""Camera model for a camera fixed in the world (eye-to-hand), with the interface of perception.CameraModel.

PickPlace asks camera.pose(robot, q) for the camera pose at every vision update; for a fixed camera that is a
constant, from the eye-to-hand calibration (configs/real_camera_extrinsics.json, T_world_cam in the simulation
world frame). The rotation is returned in the MuJoCo camera convention, as camera_to_world expects.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXTRINSICS_FILE = Path(__file__).resolve().parents[2] / "configs/real_camera_extrinsics.json"
_MJ_FROM_CV = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class FixedCameraModel:
    K: np.ndarray
    p_world_cam: np.ndarray  # camera centre in the world frame
    R_world_cv: np.ndarray   # OpenCV camera axes (x right, y down, z forward) in the world frame
    source: str = "eye-to-hand"

    @classmethod
    def from_calibration(cls, path: Path = EXTRINSICS_FILE) -> "FixedCameraModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run scripts/calibrate_eye_to_hand.py first")
        data = json.loads(path.read_text())
        if np.any(np.abs(data["dist"]) > 0):
            raise NotImplementedError("back-projection assumes zero distortion")
        T = np.asarray(data["T_world_cam"], dtype=float)
        return cls(K=np.asarray(data["K"], dtype=float), p_world_cam=T[:3, 3], R_world_cv=T[:3, :3],
                   source=f"eye-to-hand {data.get('date', '')}")

    @classmethod
    def from_mujoco(cls, model, data, name: str, K: np.ndarray) -> "FixedCameraModel":
        """Exact pose of a MuJoCo camera, for the simulation test of the lab pipeline."""
        cam = model.camera(name).id
        R_mj = data.cam_xmat[cam].reshape(3, 3).copy()
        return cls(K=np.asarray(K, dtype=float), p_world_cam=data.cam_xpos[cam].copy(),
                   R_world_cv=R_mj @ _MJ_FROM_CV, source=f"mujoco {name}")

    def pose(self, robot=None, q=None) -> tuple[np.ndarray, np.ndarray]:
        """Camera position and rotation (MuJoCo convention) in the world frame; independent of q."""
        return self.p_world_cam.copy(), self.R_world_cv @ _MJ_FROM_CV
