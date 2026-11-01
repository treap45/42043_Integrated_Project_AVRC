"""Per-trial randomisation of the task scene.

A trial is fully determined by (seed, level). Difficulty levels follow the project plan:
    L0  no occluders; the place zone lies fully inside the wrist camera's view from the "observe" keyframe
    L1  one to three occluders near the cube; the place zone may lie anywhere in its range
(L1 and L2 of the first design, one occluder and two or three, were merged into this one occluded level.)
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

LEVEL_OCCLUDERS = {"L0": (0, 0), "L1": (1, 3)}
LEVEL_ZONE_IN_VIEW = {"L0": True, "L1": False}


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


@dataclass(frozen=True)
class CubeMotionConfig:
    """Moving-cube task: the cube slides along the table at constant speed, by default in a straight line, then stops.

    The cube keeps moving well after the arm has reached it (about 3-4 s), so the arm is seen following it. The
    place zone is fixed, since the task is about tracking the cube; the cube start stays the trial's random one
    unless its path would pass the zone, in which case a new start is drawn. All of it comes from its own random
    stream [seed, 4].
    """

    start_time: float = 0.5  # s; the OBSERVE capture happens at 0.2 s, so the first frame sees the cube at rest
    speed: tuple = (0.01, 0.015)  # m/s
    duration: tuple = (9.0, 11.0)  # s: 9-16.5 cm in all, at least 5 cm of it with the arm already above (from about 4 s)
    # m, end-point range: wider than the start range, since the arm follows the cube and it need not stay in the
    # observe view. None keeps the cube sampling range.
    end_x: tuple | None = None
    end_y: tuple | None = None
    max_radius: float = 0.43  # m, end point to the base axis; past about 0.45 m the shoulder_pan joint oscillates
    zone_xy: tuple | None = (0.30, 0.12)  # m, fixed place zone (inside the observe view); None keeps the trial's
    zone_clearance: float = 0.12  # m, whole path to zone centre, so the cube never slides onto the zone
    max_tries: int = 1000
    tries_per_start: int = 1000  # a new start is drawn after this many rejected paths from the current one
    # "line"; "arc": constant turn rate speed / radius; "s_curve": turn rate speed / radius * sin(2 pi t / period);
    # "variable": an S-curve whose speed swings by +/- speed_swing and that pauses briefly, like a push by hand;
    # "long_pause": a straight line with one pause of pause_s, after the arm has reached the cube.
    # Only "variable" and "long_pause" change the speed. The cube translates without spinning, so its yaw stays the trial's.
    kind: str = "line"
    speed_swing: float = 0.5  # fraction of the speed
    speed_period: tuple = (2.0, 4.0)  # s
    short_pauses: tuple = (1, 2)  # number of pauses in "variable"
    short_pause_s: tuple = (0.4, 0.8)  # s
    pause_s: float = 2.0  # s, the pause in "long_pause"
    pause_at: tuple = (5.0, 6.0)  # s after the start of the motion, in "long_pause"
    turn_radius: tuple = (0.10, 0.16)  # m
    s_period: tuple = (5.0, 8.0)  # s, about one full S within the motion


@dataclass(frozen=True)
class CubeMotion:
    start_time: float  # s
    start_xy: np.ndarray  # m
    velocity: np.ndarray  # m/s, world frame; for a curved path, the initial velocity
    duration: float  # s
    path: tuple | None = None  # curved path: (time since the start, xy, velocity) tables; None is a straight line

    @property
    def stop_time(self) -> float:
        return self.start_time + self.duration

    @property
    def end_xy(self) -> np.ndarray:
        if self.path is not None:
            return self.path[1][-1].copy()
        return self.start_xy + self.velocity * self.duration

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Prescribed cube xy position and velocity at time t (at rest before the start and after the stop)."""
        if t < self.start_time:
            return self.start_xy.copy(), np.zeros(2)
        if t >= self.stop_time:
            return self.end_xy, np.zeros(2)
        if self.path is not None:
            tau, xy, v = self.path
            dt = t - self.start_time
            return (np.array([np.interp(dt, tau, xy[:, 0]), np.interp(dt, tau, xy[:, 1])]),
                    np.array([np.interp(dt, tau, v[:, 0]), np.interp(dt, tau, v[:, 1])]))
        return self.start_xy + self.velocity * (t - self.start_time), self.velocity.copy()


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to the segment ab."""
    ab = b - a
    s = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + s * ab)))


def _pause_factor(tau: np.ndarray, start: float, length: float, ramp: float = 0.1) -> np.ndarray:
    """0 outside a pause and 1 inside it, with smooth ramps of the given time on both sides."""
    t = np.minimum(tau - start, start + length - tau)
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t / ramp, 0.0, 1.0)))


def _speed_profile(rng, kind: str, tau: np.ndarray, speed: float, cfg: "CubeMotionConfig") -> np.ndarray:
    """Speed over time for the kinds that change it; a constant for the others."""
    if kind == "variable":
        profile = speed * (1.0 + cfg.speed_swing * np.sin(2.0 * np.pi * tau / rng.uniform(*cfg.speed_period)
                                                          + rng.uniform(0.0, 2.0 * np.pi)))
        for _ in range(int(rng.integers(cfg.short_pauses[0], cfg.short_pauses[1] + 1))):
            profile = profile * (1.0 - _pause_factor(tau, rng.uniform(1.0, max(tau[-1] - 1.5, 1.0)),
                                                     rng.uniform(*cfg.short_pause_s)))
        return profile
    if kind == "long_pause":
        return speed * (1.0 - _pause_factor(tau, rng.uniform(*cfg.pause_at), cfg.pause_s))
    return np.full_like(tau, speed)


def _curved_path(rng, cfg: "CubeMotionConfig", start: np.ndarray, heading: float, speed: float, duration: float,
                 omega: float, period: float, dt: float = 0.002) -> tuple:
    """Time, xy and velocity tables of a path with the heading in closed form and the position integrated."""
    kind = cfg.kind
    tau = np.linspace(0.0, duration, int(np.ceil(duration / dt)) + 1)
    if kind == "arc":
        theta = heading + omega * tau
    elif kind == "long_pause":
        theta = np.full_like(tau, heading)
    else:
        theta = heading + omega * period / (2.0 * np.pi) * (1.0 - np.cos(2.0 * np.pi * tau / period))
    v = _speed_profile(rng, kind, tau, speed, cfg)[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))
    steps = 0.5 * (v[1:] + v[:-1]) * np.diff(tau)[:, None]
    return tau, start + np.vstack((np.zeros(2), np.cumsum(steps, axis=0))), v


def sample_cube_motion(seed: int, trial: TrialConfig, cfg: CubeMotionConfig = CubeMotionConfig(),
                       ws: WorkspaceConfig = WorkspaceConfig()) -> tuple[TrialConfig, CubeMotion]:
    """Random heading, speed and duration from the cube start. Returns the trial with the fixed zone (and a new
    cube start if the old one is too close to it) and the motion. The end point stays in the cube sampling range
    and within reach, the whole path keeps clear of the zone, and the path keeps clear of occluders."""
    rng = np.random.default_rng([seed, 4])
    cube_r = np.sqrt(2) * CUBE_HALF
    zone = np.asarray(trial.zone_xy if cfg.zone_xy is None else cfg.zone_xy, dtype=float)
    start = np.asarray(trial.cube_xy, dtype=float)
    end_x, end_y = cfg.end_x or ws.cube_x, cfg.end_y or ws.cube_y
    for i in range(cfg.max_tries):
        if np.linalg.norm(start - zone) < cfg.zone_clearance or (i + 1) % cfg.tries_per_start == 0:
            start = np.array([rng.uniform(*ws.cube_x), rng.uniform(*ws.cube_y)])
            continue
        heading, speed, duration = rng.uniform(0, 2 * np.pi), rng.uniform(*cfg.speed), rng.uniform(*cfg.duration)
        direction = np.array([np.cos(heading), np.sin(heading)])
        if cfg.kind != "line":
            omega = rng.choice((-1.0, 1.0)) * speed / rng.uniform(*cfg.turn_radius)
            path = _curved_path(rng, cfg, start, heading, speed, duration, omega, rng.uniform(*cfg.s_period))
            xy = path[1]
            if (xy[:, 0].min() < end_x[0] or xy[:, 0].max() > end_x[1] or xy[:, 1].min() < end_y[0]
                    or xy[:, 1].max() > end_y[1] or np.hypot(xy[:, 0], xy[:, 1]).max() > cfg.max_radius
                    or np.linalg.norm(xy - zone, axis=1).min() < cfg.zone_clearance
                    or any(np.hypot(xy[:, 0] - o.x, xy[:, 1] - o.y).min() < o.radius + cube_r + ws.clearance
                           for o in trial.occluders)):
                continue
            trial = TrialConfig(trial.seed, trial.level, start.copy(), trial.cube_yaw, zone.copy(), trial.occluders)
            return trial, CubeMotion(cfg.start_time, start.copy(), speed * direction, duration, path)
        end = start + speed * duration * direction
        if not (end_x[0] <= end[0] <= end_x[1] and end_y[0] <= end[1] <= end_y[1]):
            continue
        if np.hypot(*end) > cfg.max_radius:
            continue
        if _segment_distance(zone, start, end) < cfg.zone_clearance:
            continue
        if any(_segment_distance(np.array([o.x, o.y]), start, end) < o.radius + cube_r + ws.clearance
               for o in trial.occluders):
            continue
        trial = TrialConfig(trial.seed, trial.level, start.copy(), trial.cube_yaw, zone.copy(), trial.occluders)
        return trial, CubeMotion(cfg.start_time, start.copy(), speed * direction, duration)
    raise RuntimeError(f"could not place a cube motion for seed {seed}")


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
