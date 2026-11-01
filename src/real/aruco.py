"""A single ArUco marker on the gripper: detection with sub-pixel corners and its pose in the camera frame.

The marker frame has its origin at the marker centre, x to the right and y up as printed, z out of the paper. With
four coplanar corners solvePnP has two near-equal solutions when the marker faces the camera squarely, so
SOLVEPNP_IPPE_SQUARE is used and views tilted by at least about 20 degrees give the stable pose.
"""

from dataclasses import dataclass

import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_ID = 0


def dictionary():
    return cv2.aruco.getPredefinedDictionary(DICTIONARY)


def make_detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(dictionary(), params)


def marker_object_points(side: float) -> np.ndarray:
    """(4, 3) corners in the marker frame, in OpenCV's detection order: top-left, top-right, bottom-right, bottom-left."""
    h = side / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)


@dataclass
class MarkerObservation:
    corners: np.ndarray       # (4, 2) pixels
    R_cam_marker: np.ndarray  # (3, 3)
    t_cam_marker: np.ndarray  # (3,) metres
    reproj_px: float          # RMS reprojection error of the four corners
    side_px: float            # mean edge length in the image


def tilt_deg(obs: MarkerObservation) -> float:
    """Angle between the marker normal and the ray from the camera to the marker centre."""
    normal = obs.R_cam_marker[:, 2]
    ray = obs.t_cam_marker / np.linalg.norm(obs.t_cam_marker)
    return float(np.degrees(np.arccos(np.clip(abs(normal @ ray), 0.0, 1.0))))


def detect(detector: cv2.aruco.ArucoDetector, rgb: np.ndarray, marker_id: int = MARKER_ID) -> np.ndarray | None:
    """(4, 2) corners of marker_id, or None. Expects an RGB image."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(grey)
    if ids is None:
        return None
    for c, i in zip(corners, ids.reshape(-1)):
        if i == marker_id:
            return c.reshape(4, 2).astype(np.float64)
    return None


def marker_pose(corners: np.ndarray, side: float, K: np.ndarray, dist: np.ndarray) -> MarkerObservation:
    obj = marker_object_points(side)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = proj.reshape(4, 2) - corners
    R, _ = cv2.Rodrigues(rvec)
    edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    return MarkerObservation(
        corners=corners,
        R_cam_marker=R,
        t_cam_marker=tvec.reshape(3),
        reproj_px=float(np.sqrt(np.mean(np.sum(err**2, axis=1)))),
        side_px=float(edges.mean()),
    )


def marker_depth(depth_m: np.ndarray, corners: np.ndarray, shrink: float = 0.5) -> float:
    """Median valid depth inside the marker, shrunk toward its centre to stay off the edges; nan if none."""
    centre = corners.mean(axis=0)
    poly = (centre + shrink * (corners - centre)).astype(np.int32)
    mask = np.zeros(depth_m.shape, np.uint8)
    cv2.fillConvexPoly(mask, poly, 1)
    values = depth_m[(mask > 0) & (depth_m > 0)]
    return float(np.median(values)) if values.size else float("nan")
