"""Camera parameters used by perception: the intrinsics K and the hand-eye transform.

Two sources:
    from_calibration()  the result of scripts/calibrate_camera.py (configs/camera_calib.json)
    nominal(robot)      K from the field of view and the camera mount of the controller-side model,
                        kept to compare against the calibrated values

The camera pose for a joint estimate q is FK(q) of the end-effector site composed with the hand-eye
transform, so a calibration error moves every back-projected point consistently.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.perception.projection import K_NOMINAL

CALIB_FILE = Path(__file__).resolve().parents[2] / "configs/camera_calib.json"
_MJ_FROM_CV = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class CameraModel:
    """K (3, 3) and the OpenCV camera frame expressed in the end-effector site frame."""

    K: np.ndarray
    R_ee_cam: np.ndarray
    t_ee_cam: np.ndarray
    ee_site: str = "tcp"
    source: str = "calibrated"

    @classmethod
    def from_calibration(cls, path: Path = CALIB_FILE) -> "CameraModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run python scripts/calibrate_camera.py first")
        data = json.loads(path.read_text())
        if np.any(np.abs(data["dist"]) > 0):
            raise NotImplementedError("back-projection assumes zero distortion; undistort the image first")
        return cls(
            K=np.asarray(data["K"], dtype=float),
            R_ee_cam=np.asarray(data["R_ee_cam"], dtype=float),
            t_ee_cam=np.asarray(data["t_ee_cam"], dtype=float),
            ee_site=data["ee_site"],
            source="calibrated",
        )

    @classmethod
    def nominal(cls, robot, ee_site: str = "tcp") -> "CameraModel":
        """K_NOMINAL and the camera mount written in the controller-side model."""
        q = np.zeros(6)  # any configuration: the camera is rigidly mounted on the end-effector
        p_ee, R_ee = robot.site_pose(q, ee_site)
        p_cam, R_cam_mj = robot.camera_pose(q)
        return cls(
            K=K_NOMINAL.copy(),
            R_ee_cam=R_ee.T @ R_cam_mj @ _MJ_FROM_CV,
            t_ee_cam=R_ee.T @ (p_cam - p_ee),
            ee_site=ee_site,
            source="nominal",
        )

    def pose(self, robot, q) -> tuple[np.ndarray, np.ndarray]:
        """Camera position and rotation in the world frame for joint angles q.

        The rotation uses the MuJoCo camera convention (looks along -z, +y up), as camera_to_world expects.
        """
        p_ee, R_ee = robot.site_pose(q, self.ee_site)
        R_cam_cv = R_ee @ self.R_ee_cam
        return p_ee + R_ee @ self.t_ee_cam, R_cam_cv @ _MJ_FROM_CV
