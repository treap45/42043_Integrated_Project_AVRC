# AVRC

## Overview

## Installation

Requires git and either [uv](https://docs.astral.sh/uv/) (recommended) or Python 3.12.

```bash
git clone --recursive https://github.com/treap45/AVRC.git
cd AVRC
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
python scripts/check_control.py
python -m mujoco.viewer --mjcf assets/scene.xml
```

In the viewer, load the `observe` keyframe to see the start pose. The arm actuators are torque motors,
so with zero control the arm falls under gravity until a controller is running.

## Usage

Live simulation view (overview, wrist RGB, wrist depth, joint estimates; press `r` to re-randomise, `q` to quit).
The arm is driven by the joint controller from encoder readings only:

```bash
python scripts/demo_sim.py --level L1                    # PD, hold the observe pose
python scripts/demo_sim.py --level L2 --mode lqr --motion  # LQR, track a slow sinusoidal reference
python scripts/demo_sim.py --record videos/demo.mp4 --duration 10 --no-window
```

## Project Structure

## Results

## Contributions

## Generative AI Declaration

Tool: Claude Code (Anthropic).

Purpose: setting up the environment and the simulation (scene, simulation interface, sensor models, randomisation); writing test, demo and evaluation scripts; reviewing and revising the author's code; and explaining concepts. All AI-assisted code was reviewed and tested by the author.

## Acknowledgements
