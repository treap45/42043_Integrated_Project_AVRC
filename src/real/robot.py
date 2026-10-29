"""Read-only access to the UR3e over RTDE receive: joint angles and the TCP pose.

No RTDE control interface is opened here, so nothing in this module can move the robot. It works with the pendant in
Local or Remote mode, which lets the calibration poses be jogged on the pendant.
"""

import numpy as np
from rtde_receive import RTDEReceiveInterface

from src.real.frames import rtde_pose_to_T

ROBOT_IP = "192.168.0.191"


class URReader:
    def __init__(self, ip: str = ROBOT_IP):
        self.ip = ip
        self._r = RTDEReceiveInterface(ip)

    def q(self) -> np.ndarray:
        return np.asarray(self._r.getActualQ(), dtype=float)

    def qd(self) -> np.ndarray:
        return np.asarray(self._r.getActualQd(), dtype=float)

    def tcp_pose(self) -> np.ndarray:
        """[x, y, z, rx, ry, rz] of the pendant TCP in the RTDE base frame."""
        return np.asarray(self._r.getActualTCPPose(), dtype=float)

    def T_base_tcp(self) -> np.ndarray:
        return rtde_pose_to_T(self.tcp_pose())

    def close(self) -> None:
        self._r.disconnect()

    def __enter__(self) -> "URReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
