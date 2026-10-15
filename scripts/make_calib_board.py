"""Write the checkerboard texture used by the calib_board body in assets/scene.xml.

The board has CALIB_SQUARES squares of CALIB_SQUARE metres plus a one-square white margin. The box size in
scene.xml must match: half extents ((nx + 2) * square / 2, (ny + 2) * square / 2). Texture columns run along the
board's x axis.

Run from the repo root with the venv active:
    python scripts/make_calib_board.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim.env import CALIB_SQUARE, CALIB_SQUARES  # noqa: E402

PX_PER_SQUARE = 64
OUT = Path(__file__).resolve().parents[1] / "assets/calib_board.png"


def main() -> None:
    nx, ny = CALIB_SQUARES
    img = np.full(((ny + 2) * PX_PER_SQUARE, (nx + 2) * PX_PER_SQUARE), 255, np.uint8)
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                img[(j + 1) * PX_PER_SQUARE:(j + 2) * PX_PER_SQUARE, (i + 1) * PX_PER_SQUARE:(i + 2) * PX_PER_SQUARE] = 0
    cv2.imwrite(str(OUT), img)
    print(f"wrote {OUT}: {nx} x {ny} squares of {CALIB_SQUARE * 1000:.0f} mm, "
          f"box half extents {(nx + 2) * CALIB_SQUARE / 2:.3f} {(ny + 2) * CALIB_SQUARE / 2:.3f} m")


if __name__ == "__main__":
    main()
