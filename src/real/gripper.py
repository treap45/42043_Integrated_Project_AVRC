"""The OnRobot RG2 on the UR3e, through the XML-RPC server of the OnRobot URCap on the robot (port 41414).

This is the protocol of the lab's example 15 (UTS-RI/UR3_RG2_rosbridge, which uses the onRobot package: one HTTP
POST per call of rg_grip(id, width_mm, force_n) or rg_get_width(id)), spoken here with the standard library's
xmlrpc.client instead of pycurl. It needs Installation -> Tool I/O set to OnRobot on the pendant.

The URCap passes values straight to the gripper without range checks, so width and force are clamped here.
rg_grip returns before the fingers stop; grip() polls the width until it settles.
"""

import time
import xmlrpc.client

from src.real.robot import ROBOT_IP

PORT = 41414
GRIPPER_ID = 0
WIDTH_RANGE_MM = (0.0, 100.0)   # RG2 stroke; the onRobot package documents 0-100
FORCE_RANGE_N = (3.0, 40.0)     # RG2 grip force
DEFAULT_FORCE_N = 20.0          # the lab example's value
SETTLE_MM = 0.3                 # width change between polls that counts as stopped
POLL_S = 0.1


def _clamp(x: float, lo_hi: tuple[float, float]) -> float:
    return min(max(float(x), lo_hi[0]), lo_hi[1])


class RG2:
    def __init__(self, ip: str = ROBOT_IP, gripper_id: int = GRIPPER_ID, timeout_s: float = 5.0):
        self.gripper_id = gripper_id
        # The trailing slash matters: without a path xmlrpc.client posts to /RPC2, and the URCap answers only on /.
        self._proxy = xmlrpc.client.ServerProxy(f"http://{ip}:{PORT}/", allow_none=True)
        self.timeout_s = timeout_s

    def width(self) -> float:
        """Current finger opening in mm."""
        return float(self._proxy.rg_get_width(self.gripper_id))

    def command(self, width_mm: float, force_n: float = DEFAULT_FORCE_N) -> None:
        """Send one grip command and return immediately."""
        self._proxy.rg_grip(self.gripper_id, _clamp(width_mm, WIDTH_RANGE_MM), _clamp(force_n, FORCE_RANGE_N))

    def grip(self, width_mm: float, force_n: float = DEFAULT_FORCE_N) -> float:
        """Move to width_mm (or until the force stops it) and wait for the fingers to settle; returns the width."""
        target = _clamp(width_mm, WIDTH_RANGE_MM)
        self.command(target, force_n)
        t0 = time.monotonic()
        last = self.width()
        still = 0
        while time.monotonic() - t0 < self.timeout_s:
            time.sleep(POLL_S)
            w = self.width()
            still = still + 1 if abs(w - last) < SETTLE_MM else 0
            last = w
            if still >= 3 and (abs(w - target) < 1.0 or time.monotonic() - t0 > 0.5):
                break
        return last

    def open(self, width_mm: float = 80.0) -> float:
        return self.grip(width_mm)

    def close_on(self, object_mm: float, squeeze_mm: float = 10.0, force_n: float = DEFAULT_FORCE_N) -> tuple[float, bool]:
        """Close to object_mm - squeeze_mm; the object stops the fingers early. Returns (width, held)."""
        w = self.grip(object_mm - squeeze_mm, force_n)
        held = w > object_mm - squeeze_mm + 2.0
        return w, held
