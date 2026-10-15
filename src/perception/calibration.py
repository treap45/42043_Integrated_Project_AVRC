from __future__ import annotations

import cv2
import numpy as np


PATTERN = (9, 6)  # Number of inner corners: 10 x 7 squares.
SQUARE = 0.02     # Checkerboard square size in metres.


def board_object_points(
    pattern: tuple[int, int] = PATTERN,
    square: float = SQUARE,
) -> np.ndarray:
    """Return (N, 3) checkerboard-corner coordinates in the board frame."""
    cols, rows = pattern
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    points[:, 0] = u.reshape(-1) * square
    points[:, 1] = v.reshape(-1) * square
    return points


def detect_corners(
    rgb: np.ndarray,
    pattern: tuple[int, int] = PATTERN,
) -> np.ndarray | None:
    """Detect checkerboard corners in an RGB image, or return None."""
    rgb = np.asarray(rgb)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    found, corners = cv2.findChessboardCornersSB(
        gray,
        pattern,
        flags=(
            cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
            | cv2.CALIB_CB_NORMALIZE_IMAGE
        ),
    )

    if not found:
        return None

    # Shape (N, 2) in OpenCV 5; calibrateCamera and solvePnP accept it as is.
    return corners.astype(np.float32)


def calibrate_intrinsics(
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    pattern: tuple[int, int] = PATTERN,
    square: float = SQUARE,
    zero_distortion: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate camera intrinsics and distortion from checkerboard detections.

    zero_distortion fixes all distortion coefficients at zero. The simulated camera is an ideal pinhole, and
    leaving k1-k3 free lets them absorb noise (k3 = -0.19 on a distortion-free camera, worsening cy from
    0.19 to 0.57 px). Set it to False for a real camera.
    """
    if len(image_points) < 3:
        raise ValueError("At least three valid checkerboard views are required")

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be (width, height)")

    object_points = board_object_points(pattern, square)
    object_points_all = [object_points.copy() for _ in image_points]

    flags = 0
    if zero_distortion:
        flags = (
            cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K1
            | cv2.CALIB_FIX_K2
            | cv2.CALIB_FIX_K3
        )

    rms, K, dist, _, _ = cv2.calibrateCamera(
        object_points_all,
        image_points,
        image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags,
    )

    return K, dist.reshape(-1), float(rms)


def board_pose(
    corners: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    pattern: tuple[int, int] = PATTERN,
    square: float = SQUARE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return T_cam_board as R_cam_board, t_cam_board in the OpenCV camera frame."""
    object_points = board_object_points(pattern, square)

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed")

    R_cam_board, _ = cv2.Rodrigues(rvec)
    return R_cam_board, tvec.reshape(3)


def calibrate_hand_eye(
    R_base_ee: list[np.ndarray],
    t_base_ee: list[np.ndarray],
    R_cam_board: list[np.ndarray],
    t_cam_board: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate T_ee_cam in the OpenCV camera frame using hand-eye calibration."""
    n = len(R_base_ee)

    if n < 3:
        raise ValueError("At least three hand-eye observations are required")
    if not (
        len(t_base_ee) == n
        and len(R_cam_board) == n
        and len(t_cam_board) == n
    ):
        raise ValueError("All hand-eye observation lists must have equal length")

    R_gripper2base = [np.asarray(R, dtype=float) for R in R_base_ee]
    t_gripper2base = [
        np.asarray(t, dtype=float).reshape(3, 1)
        for t in t_base_ee
    ]
    R_target2cam = [np.asarray(R, dtype=float) for R in R_cam_board]
    t_target2cam = [
        np.asarray(t, dtype=float).reshape(3, 1)
        for t in t_cam_board
    ]

    R_ee_cam, t_ee_cam = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_PARK,
    )

    return R_ee_cam, t_ee_cam.reshape(3)