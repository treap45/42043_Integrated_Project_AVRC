"""Tune the cube colour thresholds on the lab camera, with sliders, and save them for scripts/run_real.py.

Only the camera is used; nothing here connects to the robot. Put the cube on the table under the lab's lighting. The
largest blob, which is the one the pipeline takes as the cube, has a green outline; other blobs that pass the
thresholds are tinted yellow. Move the cube around the workspace, and bring in the hand or ruler that will push it:
they must not become the largest blob, and should not touch the cube's blob.

    python scripts/tune_hsv.py
    python scripts/tune_hsv.py --image frame.png       # tune on a saved RGB image instead of the camera

Keys in the window: s saves to configs/real_cube_hsv.json (run_real.py loads it), q or Esc quits.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.perception.segmentation import CUBE_HSV, color_mask, largest_component  # noqa: E402
from src.real import config  # noqa: E402

WINDOW = "cube HSV"
MIN_PIXELS = 100  # the pipeline's smallest cube blob (src/task/pick_place.py)
H_LOW_MAX, H_HIGH_MIN = 40, 140  # red wraps around hue 0, so there is one range at each end of the 0-179 scale


def read_sliders() -> dict:
    return {"h_ranges": ((0, cv2.getTrackbarPos("H low: 0 to", WINDOW)),
                         (cv2.getTrackbarPos("H high: from", WINDOW), 179)),
            "s_min": cv2.getTrackbarPos("S min", WINDOW),
            "v_min": cv2.getTrackbarPos("V min", WINDOW)}


def render(rgb: np.ndarray, hsv: dict) -> tuple[np.ndarray, str]:
    """The image with the thresholded blobs marked, and a one-line summary."""
    mask = color_mask(rgb, **hsv)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    blobs = sorted((int(a) for a in stats[1:n, cv2.CC_STAT_AREA] if a >= MIN_PIXELS), reverse=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr[mask] = (0.5 * bgr[mask] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
    largest = largest_component(mask, min_pixels=MIN_PIXELS)
    if largest is not None:
        contours, _ = cv2.findContours(largest.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, (0, 255, 0), 2)
    text = f"{len(blobs)} blobs, largest {blobs[0] if blobs else 0} px" + (f", next {blobs[1]} px" if len(blobs) > 1 else "")
    cv2.putText(bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr, text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, default=None, help="a saved image instead of the camera")
    p.add_argument("--out", type=Path, default=config.CUBE_HSV_FILE)
    args = p.parse_args()

    start = config.load_cube_hsv(args.out) if args.out.exists() else CUBE_HSV
    (_, h_low), (h_high, _) = start["h_ranges"]
    cv2.namedWindow(WINDOW)
    cv2.createTrackbar("H low: 0 to", WINDOW, min(h_low, H_LOW_MAX), H_LOW_MAX, lambda _: None)
    cv2.createTrackbar("H high: from", WINDOW, h_high, 179, lambda _: None)
    cv2.setTrackbarMin("H high: from", WINDOW, H_HIGH_MIN)
    cv2.createTrackbar("S min", WINDOW, start["s_min"], 255, lambda _: None)
    cv2.createTrackbar("V min", WINDOW, start["v_min"], 255, lambda _: None)

    cam = None
    if args.image is not None:
        still = cv2.imread(str(args.image))
        if still is None:
            sys.exit(f"cannot read {args.image}")
        still = cv2.cvtColor(still, cv2.COLOR_BGR2RGB)
    else:
        from src.real.camera import D435i
        cam = D435i().__enter__()

    try:
        while True:
            rgb = still if cam is None else cam.capture().rgb
            hsv = read_sliders()
            bgr, _ = render(rgb, hsv)
            cv2.imshow(WINDOW, bgr)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                config.save_cube_hsv(hsv, args.out)
                print(f"saved to {args.out}: {hsv}")
            elif key in (ord("q"), 27):
                break
    finally:
        if cam is not None:
            cam.__exit__(None, None, None)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
