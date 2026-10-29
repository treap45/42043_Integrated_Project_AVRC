"""Eye-to-hand calibration: the fixed camera's pose T_base_cam from a marker carried by the gripper.

For each view i the chain base -> ee -> marker and base -> cam -> marker must agree:
    T_base_ee_i @ T_ee_marker = T_base_cam @ T_cam_marker_i
with two unknown constants, T_base_cam and T_ee_marker. This is the eye-in-hand problem with the robot poses
inverted: cv2.calibrateHandEye, given T_ee_base_i (instead of T_base_ee_i) and T_cam_marker_i, returns T_base_cam.
That is why calibrate_hand_eye() in src/perception/calibration.py is reused as is, fed the inverted poses.

With one marker the per-view PnP pose is noisy (four corners; the rotation especially), and the closed-form solvers
only see those poses. refine() then minimises the corner reprojection error over all views at once, jointly in
T_base_cam and T_ee_marker (12 parameters, scipy least_squares), starting from the closed-form solution.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.perception.calibration import calibrate_hand_eye
from src.real.frames import invert_T, make_T, rotation_angle_deg

METHODS = {
    "Park": cv2.CALIB_HAND_EYE_PARK,
    "Tsai": cv2.CALIB_HAND_EYE_TSAI,
    "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class EyeToHandResult:
    T_base_cam: np.ndarray
    T_ee_marker: np.ndarray
    pos_err_mm: np.ndarray   # per view: predicted vs measured marker position in the camera frame
    rot_err_deg: np.ndarray  # per view: same for orientation
    reproj_px: np.ndarray    # per view: RMS corner reprojection error with the calibrated chain


def solve_T_base_cam(T_base_ee: list[np.ndarray], T_cam_marker: list[np.ndarray], method: str = "Park") -> np.ndarray:
    if len(T_base_ee) < 3 or len(T_base_ee) != len(T_cam_marker):
        raise ValueError("need at least three paired views")
    T_ee_base = [invert_T(T) for T in T_base_ee]
    if method == "Park":
        R, t = calibrate_hand_eye(
            [T[:3, :3] for T in T_ee_base], [T[:3, 3] for T in T_ee_base],
            [T[:3, :3] for T in T_cam_marker], [T[:3, 3] for T in T_cam_marker],
        )
    else:
        R, t = cv2.calibrateHandEye(
            [T[:3, :3] for T in T_ee_base], [T[:3, 3].reshape(3, 1) for T in T_ee_base],
            [T[:3, :3] for T in T_cam_marker], [T[:3, 3].reshape(3, 1) for T in T_cam_marker],
            method=METHODS[method],
        )
    return make_T(R, t)


def average_T(Ts: list[np.ndarray]) -> np.ndarray:
    """Mean translation and the rotation closest (Frobenius) to the mean rotation matrix."""
    U, _, Vt = np.linalg.svd(np.mean([T[:3, :3] for T in Ts], axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return make_T(R, np.mean([T[:3, 3] for T in Ts], axis=0))


def evaluate(T_base_cam: np.ndarray, T_base_ee: list[np.ndarray], T_cam_marker: list[np.ndarray],
             corners: list[np.ndarray], object_points: np.ndarray, K: np.ndarray,
             dist: np.ndarray) -> EyeToHandResult:
    """Fit T_ee_marker for this T_base_cam and report how well the closed chain predicts every view."""
    T_cam_base = invert_T(T_base_cam)
    T_ee_marker = average_T([invert_T(Tb) @ T_base_cam @ Tc for Tb, Tc in zip(T_base_ee, T_cam_marker)])
    pos, rot, rep = [], [], []
    for Tb, Tc, c in zip(T_base_ee, T_cam_marker, corners):
        T_pred = T_cam_base @ Tb @ T_ee_marker
        pos.append(1000.0 * np.linalg.norm(T_pred[:3, 3] - Tc[:3, 3]))
        rot.append(rotation_angle_deg(T_pred[:3, :3].T @ Tc[:3, :3]))
        rvec, _ = cv2.Rodrigues(T_pred[:3, :3])
        proj, _ = cv2.projectPoints(object_points, rvec, T_pred[:3, 3], K, dist)
        rep.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - c) ** 2, axis=1)))))
    return EyeToHandResult(T_base_cam, T_ee_marker, np.array(pos), np.array(rot), np.array(rep))


def rotation_spread_deg(T_base_ee: list[np.ndarray]) -> float:
    """Largest rotation between any two gripper poses; small values leave the problem ill-conditioned."""
    Rs = [T[:3, :3] for T in T_base_ee]
    return max((rotation_angle_deg(a.T @ b) for i, a in enumerate(Rs) for b in Rs[i + 1:]), default=0.0)


def _T_from_params(x: np.ndarray) -> np.ndarray:
    return make_T(cv2.Rodrigues(x[:3])[0], x[3:])


def _params_from_T(T: np.ndarray) -> np.ndarray:
    return np.concatenate([cv2.Rodrigues(T[:3, :3])[0].reshape(3), T[:3, 3]])


def refine(T_base_cam: np.ndarray, T_ee_marker: np.ndarray, T_base_ee: list[np.ndarray],
           corners: list[np.ndarray], object_points: np.ndarray, K: np.ndarray,
           dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Jointly refine T_base_cam and T_ee_marker on the pixel residuals of every corner in every view."""
    def residuals(x):
        T_cam_base = invert_T(_T_from_params(x[:6]))
        T_ee_m = _T_from_params(x[6:])
        out = []
        for Tb, c in zip(T_base_ee, corners):
            T_cm = T_cam_base @ Tb @ T_ee_m
            proj, _ = cv2.projectPoints(object_points, cv2.Rodrigues(T_cm[:3, :3])[0], T_cm[:3, 3], K, dist)
            out.append((proj.reshape(-1, 2) - c).ravel())
        return np.concatenate(out)

    x0 = np.concatenate([_params_from_T(T_base_cam), _params_from_T(T_ee_marker)])
    sol = least_squares(residuals, x0, method="lm")
    return _T_from_params(sol.x[:6]), _T_from_params(sol.x[6:])
