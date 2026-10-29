"""Write a printable A4 page with one ArUco marker for the eye-to-hand calibration.

The marker (DICT_4X4_50, src/real/aruco.py) is drawn at its exact physical size in a PDF, with a white margin of one
cell around it and a 100 mm scale bar. Print at "actual size" (no fit-to-page), then measure the black square with
calipers and pass that value to capture_handeye.py / calibrate_eye_to_hand.py as --side-mm.

    python scripts/make_aruco.py                 # 80 mm, id 0
    python scripts/make_aruco.py --side-mm 100
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.real.aruco import MARKER_ID, dictionary  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "results/aruco"
A4_MM = (210.0, 297.0)
MM_PER_IN = 25.4


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side-mm", type=float, default=80.0, help="black square edge length")
    p.add_argument("--id", type=int, default=MARKER_ID)
    args = p.parse_args()

    cells = 6  # 4x4 data bits plus the one-cell black border on each side
    img = cv2.aruco.generateImageMarker(dictionary(), args.id, cells * 100)
    W, H = A4_MM
    fig = plt.figure(figsize=(W / MM_PER_IN, H / MM_PER_IN))

    def axes_mm(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    s = args.side_mm
    x0, y0 = (W - s) / 2, (H - s) / 2 + 20
    ax = axes_mm(x0, y0, s, s)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.axis("off")

    bar = axes_mm((W - 100) / 2, y0 - 35, 100, 4)
    bar.add_patch(plt.Rectangle((0, 0), 1, 1, color="black"))
    bar.set_xlim(0, 1)
    bar.set_ylim(0, 1)
    bar.axis("off")

    fig.text(0.5, (y0 - 42) / H, "scale bar: 100 mm", ha="center", va="top", fontsize=10)
    fig.text(0.5, (y0 - 52) / H,
             f"ArUco DICT_4X4_50, id {args.id}, black square {s:.1f} mm (one-cell white margin needed)",
             ha="center", va="top", fontsize=10)
    fig.text(0.5, (y0 - 60) / H, "Print at actual size; measure the black square and use that value as --side-mm.",
             ha="center", va="top", fontsize=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"aruco_4x4_50_id{args.id}_{s:.0f}mm"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    print(f"wrote {stem}.pdf and .png: {s:.1f} mm marker on A4; print at actual size")


if __name__ == "__main__":
    main()
