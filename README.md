# AVRC

## Overview

**Project page: <https://treap45.github.io/42043_Integrated_Project_AVRC/>** (design, parameters, results and statements).

UTS 41014 / 42043 Integrated Project, Track B, personalised. A UR3e follows a cube that is pushed across a table, decides from its own estimate when the cube has stopped, then picks and places it. Joints: one Kalman filter per joint and LQR on the estimated state. Cube: an EKF through the camera projection and PBVS. It runs in MuJoCo and on the lab UR3e with an RG2 and a fixed D435i. Ground truth from the simulator is used only to score trials and for the display panel.

Branches: `main` has the simulation; `real` adds the lab robot code; `advanced` adds the occluder extension to `main`; `github_page` holds the project page, its logs and its build script.

## Installation

Requires git and either [uv](https://docs.astral.sh/uv/) (recommended) or Python 3.12.

```bash
git clone --recursive https://github.com/treap45/42043_Integrated_Project_AVRC.git
cd 42043_Integrated_Project_AVRC
# if cloned without --recursive:
git submodule update --init --recursive
```

With uv (downloads a standalone Python 3.12 if needed):

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or with plain pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Rendering backend (add to the end of `.venv/bin/activate`):

```bash
# NVIDIA GPU available (nvidia-smi works)
export MUJOCO_GL=egl
# otherwise: sudo apt install libosmesa6-dev, then
# export MUJOCO_GL=osmesa
```

Verify the environment:

```bash
python scripts/check_env.py
python scripts/check_scene.py
python scripts/check_sim.py
python scripts/check_controller_model.py
python scripts/check_joint_ss.py
python scripts/check_joint_kf.py
python scripts/check_control.py             # LQR; add --mode pd for the PD baseline
python scripts/check_camera_noise.py
python scripts/check_calib_board.py
python -m mujoco.viewer --mjcf assets/scene.xml
```

In the viewer, load the `observe` keyframe to see the start pose. The arm actuators are torque motors,
so with zero control the arm falls under gravity until a controller is running.

## Usage

Vision-guided pick and place with the noisy wrist RGB-D camera and the calibrated camera parameters (live view of the overview camera, wrist RGB and depth, the current state, and
estimates next to the true values; press `q` to quit). Ground truth is read only to judge success and for display:

```bash
python main.py                                  # L0, seed 0, LQR: the cube slides, then stops; follow it (cube EKF + PBVS), pick once it stops
python main.py --trials 20 --no-window          # seeds 0-19, prints success rate and failure states
python main.py --record videos/pick_place.mp4 --no-window
python main.py --viewer                         # also show the run in the interactive MuJoCo viewer
python main.py --no-camera-noise                # exact wrist RGB-D (by default the camera is noisy)
python main.py --nominal-camera                 # nominal K and camera mount instead of the calibration
python main.py --no-moving-cube --approach open    # static cube, look-then-move
python main.py --no-moving-cube --approach pbvs    # static cube, PBVS from the pregrasp pose down to 60 mm above the cube
python main.py --no-moving-cube --approach pbvs --push-cube 0.025   # push the cube 25 mm as the approach starts
python main.py --no-moving-cube --approach open --push-cube 0.025   # same push with look-then-move, for comparison
python main.py --no-moving-cube --approach pbvs_full --push-cube 0.025 --push-at transit_mid   # push half-way there
python main.py --no-moving-cube --seed 7 --level L1 --mode pd --approach pbvs_full
python main.py --no-moving-cube --handeye-error-deg 2 --trials 20 --no-window   # experiment: 2 deg camera-mount error
python scripts/calibrate_camera.py              # recalibrate: writes configs/camera_calib.json used by main.py
```

Live simulation view (overview, wrist RGB, wrist depth, joint estimates; press `r` to re-randomise, `q` to quit).
The arm is driven by the joint controller from encoder readings only:

```bash
python scripts/demo_sim.py --level L1                    # LQR, hold the observe pose
python scripts/demo_sim.py --level L1 --mode pd --motion   # PD, track a slow sinusoidal reference
python scripts/demo_sim.py --record videos/demo.mp4 --duration 10 --no-window
```

## Project Structure

```
main.py          entry point and evaluator (the only place the pipeline's ground truth is read)
assets/          scene and controller models
configs/         calibration results
src/sim/         simulation interface, sensor noise, randomisation
src/model/       joint state-space model, controller-side robot model
src/perception/  segmentation, projection, cube and zone estimates, calibration
src/estimation/  joint KF, cube EKF, world-frame KF, coordinated-turn EKF
src/control/     LQR, PD, arm controller, PBVS
src/planning/    IK, trajectories, grasp poses
src/task/        task state machine
scripts/         checks with PASS or FAIL, experiments
```

## Results

See the [project page](https://treap45.github.io/42043_Integrated_Project_AVRC/). To reproduce them:

```bash
python scripts/watch_motion.py --motion arc      # harder motion, one live run in a window
python scripts/watch_motion.py --motion s_curve
```

The full set of experiments runs from branch `real`:

```bash
git checkout real
bash scripts/evaluate_all.sh   # the simulation experiments, about an hour; logs in results/eval/
```

## Contributions

See the [project page](https://treap45.github.io/42043_Integrated_Project_AVRC/#statements).

## Generative AI Declaration

Tool: Claude Code (Anthropic).

Purpose: setting up the environment and simulation; writing test, demo and evaluation scripts; reviewing and revising the author's code; writing parts of the controller, estimator and extension code, and the real-robot interface and calibration tooling, at the author's direction; building the project page; drafting the demo video's voice-over script and cutting the video; and explaining concepts. The video's voice-over is synthesised with Kokoro, a text-to-speech model, run locally. All AI-assisted code was reviewed and tested by the author.

## Acknowledgements

- **Robot model:** UR3e MJCF from [SouthColumn76/universal_robots_ur3e](https://github.com/SouthColumn76/universal_robots_ur3e) (BSD-3-Clause), the model used in the course tutorial [rFalque/mujoco_UR3_tutorial](https://github.com/rFalque/mujoco_UR3_tutorial); arm actuators changed to torque motors (see the header of `assets/scene.xml`).
- **Gripper and camera models:** Robotiq 2F-85 (BSD-2-Clause) and RealSense D435i (Apache-2.0) from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
- **Simulator:** [MuJoCo](https://mujoco.org) (rendering, physics, and the controller-side model for kinematics and dynamics).
- **OpenCV:** colour segmentation (HSV conversion, morphology, connected components), `minAreaRect`, checkerboard detection (`findChessboardCornersSB`), `calibrateCamera`, `solvePnP` and `calibrateHandEye` (Park and Martin method: F. C. Park and B. J. Martin, "Robot sensor calibration: solving AX = XB on the Euclidean group", IEEE Trans. Robotics and Automation, 1994).
- **SciPy:** `solve_discrete_are` (LQR gains), `expm` (discretisation and Van Loan process noise), `CubicSpline` (Cartesian paths).
- **Robotics Toolbox for Python:** used only in the environment check `scripts/check_env.py`; the pipeline does not use it.
