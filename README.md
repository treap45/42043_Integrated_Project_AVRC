# AVRC

## Overview

## Installation

Requires Python 3.12 and git.

```bash
git clone --recursive https://github.com/treap45/AVRC.git
cd AVRC
# if cloned without --recursive:
git submodule update --init --recursive

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
python -m mujoco.viewer --mjcf third_party/mujoco_menagerie/universal_robots_ur5e/scene.xml
```

## Usage

## Project Structure

## Results

## Contributions

## Generative AI Declaration

## Acknowledgements
