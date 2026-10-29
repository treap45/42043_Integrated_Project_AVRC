"""Settings that differ between the simulation and the lab, applied on top of the simulation pipeline.

The perception, estimation and planning modules keep the simulation's 50 mm cube and 0.75 m table as module
constants (read at call time). apply() overwrites those constants for a run on the lab setup, so the same code
serves both without editing the simulation files; it is called once by scripts/run_real.py before anything runs.
"""

from dataclasses import dataclass, field

from src.real.frames import SIM_BASE_HEIGHT


@dataclass
class RealSettings:
    cube_side: float = 0.060                 # m, the printed cube; measure it and pass --cube-mm
    table_z: float = SIM_BASE_HEIGHT         # m, in the simulation world: the cube surface is level with the base
    tcp_offset: float | None = None          # m, flange to the RG2 fingertip centre along the flange z; None keeps 0.156
    cube_hsv: dict | None = None             # overrides segmentation.CUBE_HSV (thresholds tuned on lab images)
    hover: float = 0.10                      # m, TCP above the grasp pose while tracking (grasp.HOVER)
    track_v_max: float = 0.08                # m/s, TCP speed limit while tracking (sim 0.15); quintic peak 75 % of it
    applied: dict = field(default_factory=dict)


def apply(settings: RealSettings, robot=None) -> dict:
    """Overwrite the geometry constants of the pipeline modules; returns what was changed (for the log)."""
    from src.estimation import cube_ekf
    from src.perception import cube_pose, segmentation
    from src.planning import grasp

    half = settings.cube_side / 2.0
    changes = {
        (cube_pose, "CUBE_SIDE"): settings.cube_side,
        (cube_pose, "CUBE_HALF_HEIGHT"): half,
        (grasp, "CUBE_HALF_HEIGHT"): half,
        (grasp, "TABLE_Z"): settings.table_z,
        (cube_ekf, "Z_TOP"): settings.table_z + settings.cube_side,
    }
    from src.planning import trajectory
    from src.task import pick_place
    changes[(grasp, "HOVER")] = settings.hover
    changes[(pick_place, "TRANSIT_V_MAX")] = settings.track_v_max
    if settings.cube_hsv is not None:
        changes[(segmentation, "CUBE_HSV")] = dict(settings.cube_hsv)
        changes[(pick_place, "CUBE_HSV")] = dict(settings.cube_hsv)
    log = {}
    for (module, name), value in changes.items():
        if not hasattr(module, name):
            raise AttributeError(f"{module.__name__}.{name} no longer exists; update src/real/config.py")
        log[f"{module.__name__}.{name}"] = (getattr(module, name), value)
        setattr(module, name, value)

    if settings.tcp_offset is not None:
        if robot is None:
            raise ValueError("a tcp_offset needs the robot model")
        site = robot.model.site("tcp")
        log["tcp site z"] = (float(site.pos[2]), settings.tcp_offset)
        site.pos[2] = settings.tcp_offset
    # The TRACK reference is built with GoalRelativeQuintic's default peak speed; slow it with the TCP limit.
    defaults = trajectory.GoalRelativeQuintic.__init__.__defaults__  # (v_peak, w_peak, t_min)
    new_defaults = (0.75 * settings.track_v_max, *defaults[1:])
    log["GoalRelativeQuintic v_peak"] = (defaults[0], new_defaults[0])
    trajectory.GoalRelativeQuintic.__init__.__defaults__ = new_defaults
    settings.applied = log
    return log
