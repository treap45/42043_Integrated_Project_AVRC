"""Per-trial randomisation of the task scene.

A trial is fully determined by (seed, level). Difficulty levels follow the project plan:
    L0  no occluders; the place zone lies fully inside the wrist camera's view from the "observe" keyframe
    L1  one occluder near the cube; the place zone may lie anywhere in its range
    L2  two or three occluders near the cube; the place zone may lie anywhere in its range
L3 (cube fully hidden from the initial view) is not implemented.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

TABLE_Z = 0.75
CUBE_HALF = 0.025
ZONE_HALF = 0.05
N_OCCLUDERS = 3
PARK_POS = np.array([0.0, 0.9, -1.0])  # unused occluders are parked under the floor

LEVEL_OCCLUDERS = {"L0": (0, 0), "L1": (1, 1), "L2": (2, 3)}
LEVEL_ZONE_IN_VIEW = {"L0": True, "L1": False, "L2": False}


@dataclass(frozen=True)
class WorkspaceConfig:
    """Sampling ranges (metres, radians, world frame). Cube ranges keep the cube inside the wrist camera's
    view from the "observe" keyframe and inside the UR3e's reach."""

    cube_x: tuple = (0.25, 0.40)
    cube_y: tuple = (-0.15, 0.15)
    cube_yaw: tuple = (-np.pi / 4, np.pi / 4)
    zone_x: tuple = (0.20, 0.42)
    zone_y: tuple = (-0.28, 0.28)
    # Zone-centre range for which the whole zone is in the wrist image from "observe" (computed by projecting
    # its corners with the nominal intrinsics: x 0.27-0.44, y +/-0.155 with an 8 px margin), slightly shrunk.
    zone_x_in_view: tuple = (0.28, 0.42)
    zone_y_in_view: tuple = (-0.15, 0.15)
    zone_min_dist: float = 0.15  # zone centre to cube centre
    occ_dist: tuple = (0.07, 0.12)  # occluder centre to cube centre
    occ_half_xy: tuple = (0.02, 0.04)
    occ_height: tuple = (0.08, 0.15)
    clearance: float = 0.01  # minimum gap between objects (bounding circles)
    base_clearance: float = 0.20  # occluder bounding circle to robot base axis, keeps them clear of the arm
    max_tries: int = 1000


@dataclass
class Occluder:
    x: float
    y: float
    yaw: float
    half_x: float
    half_y: float
    height: float

    @property
    def radius(self) -> float:
        return float(np.hypot(self.half_x, self.half_y))


@dataclass
class TrialConfig:
    seed: int
    level: str
    cube_xy: np.ndarray
    cube_yaw: float
    zone_xy: np.ndarray
    occluders: list = field(default_factory=list)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def sample_trial(seed: int, level: str = "L0", ws: WorkspaceConfig = WorkspaceConfig()) -> TrialConfig:
    if level not in LEVEL_OCCLUDERS:
        raise ValueError(f"unknown level {level!r}; expected one of {list(LEVEL_OCCLUDERS)}")
    rng = np.random.default_rng(seed)
    u = lambda lo_hi: rng.uniform(*lo_hi)  # noqa: E731

    cube_xy = np.array([u(ws.cube_x), u(ws.cube_y)])
    cube_yaw = u(ws.cube_yaw)
    cube_r = np.sqrt(2) * CUBE_HALF

    lo, hi = LEVEL_OCCLUDERS[level]
    n_occ = int(rng.integers(lo, hi + 1))
    occluders = []
    for _ in range(ws.max_tries):
        if len(occluders) == n_occ:
            break
        ang, dist = rng.uniform(0, 2 * np.pi), u(ws.occ_dist)
        occ = Occluder(cube_xy[0] + dist * np.cos(ang), cube_xy[1] + dist * np.sin(ang), rng.uniform(0, np.pi),
                       u(ws.occ_half_xy), u(ws.occ_half_xy), u(ws.occ_height))
        if dist < occ.radius + cube_r + ws.clearance:
            continue
        if np.hypot(occ.x, occ.y) < occ.radius + ws.base_clearance:
            continue
        if any(np.hypot(occ.x - o.x, occ.y - o.y) < occ.radius + o.radius + ws.clearance for o in occluders):
            continue
        occluders.append(occ)
    else:
        raise RuntimeError(f"could not place {n_occ} occluders for seed {seed}")

    zone_r = np.sqrt(2) * ZONE_HALF
    if LEVEL_ZONE_IN_VIEW[level]:
        zone_x, zone_y = ws.zone_x_in_view, ws.zone_y_in_view
    else:
        zone_x, zone_y = ws.zone_x, ws.zone_y
    for _ in range(ws.max_tries):
        zone_xy = np.array([u(zone_x), u(zone_y)])
        if np.linalg.norm(zone_xy - cube_xy) < ws.zone_min_dist:
            continue
        if any(np.hypot(zone_xy[0] - o.x, zone_xy[1] - o.y) < zone_r + o.radius + ws.clearance for o in occluders):
            continue
        break
    else:
        raise RuntimeError(f"could not place the place zone for seed {seed}")

    return TrialConfig(seed, level, cube_xy, cube_yaw, zone_xy, occluders)


def apply_trial(model: mujoco.MjModel, data: mujoco.MjData, trial: TrialConfig) -> None:
    """Write a sampled trial into the model/data. Call mj_forward afterwards."""
    jnt = model.joint("cube_joint")
    qadr, vadr = model.jnt_qposadr[jnt.id], model.jnt_dofadr[jnt.id]
    data.qpos[qadr:qadr + 3] = [*trial.cube_xy, TABLE_Z + CUBE_HALF]
    data.qpos[qadr + 3:qadr + 7] = _yaw_quat(trial.cube_yaw)
    data.qvel[vadr:vadr + 6] = 0.0

    zone = model.body("place_zone").mocapid[0]
    data.mocap_pos[zone] = [*trial.zone_xy, TABLE_Z + 0.0005]

    for i in range(N_OCCLUDERS):
        body = model.body(f"occluder_{i}")
        geom = model.geom(f"occluder_{i}").id
        mocap = body.mocapid[0]
        if i < len(trial.occluders):
            o = trial.occluders[i]
            size = np.array([o.half_x, o.half_y, o.height / 2])
            data.mocap_pos[mocap] = [o.x, o.y, TABLE_Z + o.height / 2]
            data.mocap_quat[mocap] = _yaw_quat(o.yaw)
        else:
            size = model.geom_size[geom].copy()
            data.mocap_pos[mocap] = PARK_POS + [0.2 * i, 0, 0]
            data.mocap_quat[mocap] = [1, 0, 0, 0]
        # Collision broad-phase uses the bounding sphere and box, so they must follow the new size.
        model.geom_size[geom] = size
        model.geom_rbound[geom] = np.linalg.norm(size)
        model.geom_aabb[geom] = [0, 0, 0, *size]
