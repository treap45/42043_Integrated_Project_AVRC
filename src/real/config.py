"""Settings that differ between the simulation and the lab, applied on top of the simulation pipeline.

The perception, estimation and planning modules keep the simulation's 50 mm cube and 0.75 m table as module
constants (read at call time). apply() overwrites those constants for a run on the lab setup, so the same code
serves both without editing the simulation files; it is called once by scripts/run_real.py before anything runs.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.real.frames import SIM_BASE_HEIGHT

CUBE_HSV_FILE = Path(__file__).resolve().parents[2] / "configs/real_cube_hsv.json"  # written by scripts/tune_hsv.py


def load_cube_hsv(path: Path = CUBE_HSV_FILE) -> dict:
    """Thresholds in the form segmentation.color_mask takes: h_ranges, s_min, v_min."""
    data = json.loads(Path(path).read_text())
    return {"h_ranges": tuple(tuple(r) for r in data["h_ranges"]), "s_min": data["s_min"], "v_min": data["v_min"]}


def save_cube_hsv(hsv: dict, path: Path = CUBE_HSV_FILE) -> None:
    Path(path).write_text(json.dumps(hsv, indent=2) + "\n")


@dataclass
class RealSettings:
    cube_side: float = 0.060                 # m, the printed cube; measure it and pass --cube-mm
    table_z: float = SIM_BASE_HEIGHT         # m, in the simulation world: the cube surface is level with the base
    tcp_offset: float | None = None          # m, flange to the RG2 fingertip centre along the flange z; None keeps 0.156
    cube_hsv: dict | None = None             # overrides segmentation.CUBE_HSV (thresholds tuned on lab images)
    hover: float = 0.10                      # m, TCP above the grasp pose while tracking (grasp.HOVER)
    track_v_max: float = 0.08                # m/s, TCP speed limit while tracking (sim 0.15); quintic peak 75 % of it
    lab_estimation: bool = True              # the cube estimate and EKF noise of src/real/cube_top.py; False keeps the sim's
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

    if settings.lab_estimation:
        # The simulation's modules are unchanged; only the names PickPlace uses point elsewhere for this run.
        from src.real import cube_top
        log["cube top band"] = (0.005, cube_top.TOP_BAND)
        log["cube size tolerance"] = (0.005, cube_top.SIZE_TOL)
        log["cube EKF yaw sigma"] = (0.01, cube_top.YAW_SIGMA)
        pick_place.estimate_cube = cube_top.estimate_cube
        pick_place.cube_measurement = cube_top.cube_measurement
        pick_place.CubeEKF = cube_top.LabCubeEKF
        # Set the cube down rather than drop it: the simulation releases it 3 mm above the table.
        log["release gap"] = (grasp.RELEASE_GAP, 0.001)
        grasp.RELEASE_GAP = 0.001

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
