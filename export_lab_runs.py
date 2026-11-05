"""Summarise run.json logs of scripts/run_real.py into one CSV row per run, for website/build.py.

    python website/export_lab_runs.py hardware results/real_runs/20261102_* >  website/data/lab_runs.csv
    python website/export_lab_runs.py --no-header sim_fast <run dirs>       >> website/data/lab_runs.csv

Runs with --sim or --dry-run are left out of the "hardware" group, and only --sim runs go in any other group.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

FIELDS = ["group", "id", "mode", "v_max", "ok", "elapsed_s", "reason", "updates", "accepted", "rejected", "reinit",
          "no_cube", "tcp_err_median_mm", "tcp_err_p95_mm", "tcp_err_max_mm", "cube_speed_max_mm_s", "stop_time_s",
          "grasp_width_mm", "placed_x_mm", "placed_y_mm", "placed_yaw_deg"]


def row(group: str, run_dir: Path) -> dict | None:
    log = json.loads((run_dir / "run.json").read_text())
    args = log["args"]
    if args.get("dry_run") or bool(args.get("sim")) != (group != "hardware"):
        return None
    outcome = log.get("outcome") or {}
    track = log.get("track_log") or []
    status = [t["status"] for t in track]
    err = np.array([t["e_p"] for t in track if t.get("e_p") is not None]) * 1e3
    speed = [t["speed"] for t in track if t.get("speed") is not None]
    placed = log.get("cube_placed_rtde")
    out = {
        "group": group, "id": run_dir.name, "mode": args["mode"], "v_max": args.get("v_max"),
        "ok": int(bool(outcome.get("ok")) and not log.get("error")), "elapsed_s": outcome.get("elapsed"),
        "reason": log.get("error") or outcome.get("reason") or "", "updates": len(status),
        "accepted": status.count("accepted"), "rejected": status.count("rejected"), "reinit": status.count("reinit"),
        "no_cube": status.count("none"), "stop_time_s": log.get("track_stop_time"),
        "grasp_width_mm": log.get("grasp_width_mm"),
    }
    if err.size:
        out.update(tcp_err_median_mm=np.median(err), tcp_err_p95_mm=np.percentile(err, 95), tcp_err_max_mm=err.max())
    if speed:
        out["cube_speed_max_mm_s"] = 1e3 * max(speed)
    if placed:
        out.update(placed_x_mm=1e3 * placed[0], placed_y_mm=1e3 * placed[1], placed_yaw_deg=placed[2])
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-header", action="store_true")
    p.add_argument("group")
    p.add_argument("runs", nargs="+", type=Path)
    args = p.parse_args()
    writer = csv.DictWriter(sys.stdout, FIELDS)
    if not args.no_header:
        writer.writeheader()
    for run_dir in args.runs:
        if (run_dir / "run.json").exists():
            r = row(args.group, run_dir)
            if r is not None:
                writer.writerow(r)


if __name__ == "__main__":
    main()
