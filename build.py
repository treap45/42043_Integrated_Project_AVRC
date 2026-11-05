"""Build website/index.html from template.html and the evaluation logs in website/data/.

    python website/build.py

Every table and most numbers on the page come from website/data/logs/ (scripts/evaluate_all.sh) and
website/data/lab_runs.csv (website/export_lab_runs.py), so the page cannot drift from the logs. Numbers from earlier
development runs are written in the template and labelled there as development measurements.
"""

import csv
import json
import re
import zipfile
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "data/logs"


def main_log(name: str) -> dict:
    """Per-seed lines and the summary printed by main.py."""
    text = (LOGS / f"{name}.log").read_text()
    seeds = re.findall(r"^seed +(\d+) +(SUCCESS|FAIL) +grasped=(\w+) +grasp offset +(\S+) +mm +place error +([\d.]+) mm"
                       r" +sim time +([\d.]+) s(.*)$", text, re.M)
    if not seeds:
        raise ValueError(f"{name}: no trials in the log")
    placed = [s for s in seeds if s[1] == "SUCCESS"]
    offsets = [float(s[3]) for s in seeds if s[3] != "-"]
    errs = [float(s[4]) for s in placed]
    out = {
        "n": len(seeds), "placed": len(placed), "grasped": sum(s[2] == "True" for s in seeds),
        "omed": f"{median(offsets):.1f}" if offsets else "-", "omax": f"{max(offsets):.1f}" if offsets else "-",
        "pmed": f"{median(errs):.1f}" if errs else "-", "pmax": f"{max(errs):.1f}" if errs else "-",
        "tmed": f"{median(float(s[5]) for s in placed):.1f}" if placed else "-",
        "offsets": [None if s[3] == "-" else float(s[3]) for s in seeds],
        "failures": [s[6].strip() for s in seeds if s[1] != "SUCCESS"],
    }
    m = re.search(r"TRACK: rejected measurements (\d+), re-initialisations (\d+)(?:, stop declared ([\d.]+) s after the "
                  r"true stop \(median, min ([\d.]+), max ([\d.]+)\))?(?:, TCP-to-cube lag while moving RMS median ([\d.]+) mm)?",
                  text)
    if m:
        out.update(rej=m[1], reinit=m[2], delay=m[3] or "-", delay_min=m[4] or "-", delay_max=m[5] or "-", lag=m[6] or "-")
    out["fail_states"] = {k: json.loads(v) for k, v in re.findall(r"failed in (\w+): seeds (\[.*\])", text)}
    return out


def motion_log(name: str) -> dict:
    text = (LOGS / f"{name}.log").read_text()
    m = re.search(r"placed (\d+)/(\d+), grasped (\d+)/\d+\n\s+stop declared early (\d+), late delay median ([\d.nan]+) s", text)
    o = re.search(r"grasp offset median ([\d.]+) mm, max ([\d.]+) mm", text)
    p = re.search(r"place error median ([\d.]+) mm, max ([\d.]+) mm", text)
    r = re.search(r"rejected (\d+), reinit (\d+)", text)
    return {"placed": m[1], "n": m[2], "grasped": m[3], "early": m[4], "delay": m[5], "omed": o[1], "omax": o[2],
            "pmed": p[1], "pmax": p[2], "rej": r[1], "reinit": r[2]}


def filter_rows(name: str) -> dict:
    """{(section, filter): row} from compare_cube_filters.py."""
    rows, section = {}, None
    for line in (LOGS / f"{name}.log").read_text().splitlines():
        head = re.match(r"(\w+): \d+ seeds", line)
        if head:
            section = head[1]
        m = re.match(r"\s+(.+?)\s+pos\s+([\d.]+) mm\s+vel\s+([\d.]+) mm/s\s+NEES\s+([\d.]+)\s+NIS/m\s+([\d.]+)\s+rej\s+(\d+) "
                     r"reinit\s+(\d+)\s+stop\s+([\d.nan]+) s \((\d+)/(\d+)\)\s+gap\s+([\d.]+) mm \(max\s+([\d.]+)\)", line)
        if m:
            rows[(section, m[1])] = dict(zip(
                ["pos", "vel", "nees", "nis", "rej", "reinit", "stop", "stops", "n", "gap", "gapmax"], m.groups()[1:]))
    return rows


def table(head: list, rows: list, numeric_from: int = 1, highlight: tuple = ()) -> str:
    def cells(row, tag):
        return "".join(f'<{tag}{" class=n" if i >= numeric_from else ""}>{c}</{tag}>' for i, c in enumerate(row))
    body = "".join(f'<tr{" class=hi" if i in highlight else ""}>{cells(r, "td")}</tr>' for i, r in enumerate(rows))
    return f'<div class="table-wrap"><table><thead><tr>{cells(head, "th")}</tr></thead><tbody>{body}</tbody></table></div>'


def lab_runs() -> list:
    rows = list(csv.DictReader((ROOT / "data/lab_runs.csv").open()))
    for r in rows:
        r["ok"] = r["ok"] == "1"
        for key in ("elapsed_s", "updates", "accepted", "rejected", "reinit", "no_cube", "tcp_err_median_mm",
                    "tcp_err_max_mm", "cube_speed_max_mm_s", "placed_x_mm", "placed_y_mm", "placed_yaw_deg"):
            r[key] = float(r[key]) if r[key] != "" else None
    return rows


def ekf_trace() -> dict:
    rows = list(csv.DictReader((ROOT / "data/ekf_trace_seed0.csv").open()))
    col = lambda k, scale=1e3: [round(float(r[k]) * scale, 3) for r in rows]  # noqa: E731
    x, sx = col("x_est"), col("sx")
    hyp = lambda a, b: round((a * a + b * b) ** 0.5, 3)  # noqa: E731
    text = (LOGS / "plot_ekf.log").read_text()
    stop = re.search(r"stop: true ([\d.]+) s, declared ([\d.]+) s", text)
    return {
        "t": col("t", 1.0), "x_hat": x, "x_true": col("x_true"),
        "x_lo": [round(a - 2 * s, 3) for a, s in zip(x, sx)], "x_hi": [round(a + 2 * s, 3) for a, s in zip(x, sx)],
        "err": [hyp(a - b, c - d) for a, b, c, d in zip(x, col("x_true"), col("y_est"), col("y_true"))],
        "sigma2": [round(2 * s, 3) for s in col("s_xy_max")],
        "v_hat": [hyp(a, b) for a, b in zip(col("vx_est"), col("vy_est"))],
        "v_true": [hyp(a, b) for a, b in zip(col("vx_true"), col("vy_true"))],
        "true_stop": float(stop[1]), "declared_stop": float(stop[2]),
    }


def build() -> None:
    v = {}
    logs = {name: main_log(name) for name in [
        "moving_lqr", "moving_pd", "moving_fixed", "moving_noiseless", "moving_nominal", "moving_ct", "static_open",
        "static_pbvs", "static_pd", "push_open", "push_pbvs", "push_open15", "push_pbvs15", "he2_open", "he2_pbvs",
        "he2_track", "l1_off", "l1_conservative", "l1_active", "l0_active"]}
    for name, log in logs.items():
        for key, value in log.items():
            if isinstance(value, (str, int)):
                v[f"{name}.{key}"] = value

    def task_row(label, name):
        r = logs[name]
        return [label, f'{r["placed"]}/{r["n"]}', r["omed"], r["omax"], r["pmed"], r["pmax"], r["tmed"]]

    head = ["Condition (L0, noisy camera, calibrated model)", "Placed", "Grasp offset median (mm)", "max", "Place error median (mm)", "max", "Time median (s)"]
    v["tbl_batch"] = table(head, [
        task_row("<strong>Moving cube, TRACK, LQR</strong> (default task)", "moving_lqr"),
        task_row("Moving cube, PD baseline", "moving_pd"),
        task_row("Moving cube, gain scheduling off", "moving_fixed"),
        task_row("Moving cube, coordinated-turn EKF", "moving_ct"),
        task_row("Moving cube, noise-free camera", "moving_noiseless"),
        task_row("Moving cube, nominal camera (no calibration)", "moving_nominal"),
        task_row("Static cube, look-then-move", "static_open"),
        task_row("Static cube, PBVS approach", "static_pbvs"),
        task_row("Static cube, look-then-move, PD", "static_pd"),
    ], highlight=(0,))

    head = ["Disturbance (static cube, seeds 0-19)", "Look-then-move placed", "grasp offset median (mm)", "PBVS placed", "grasp offset median (mm)"]
    def pair(label, a, b):
        return [label, f'{logs[a]["placed"]}/{logs[a]["n"]}', logs[a]["omed"], f'{logs[b]["placed"]}/{logs[b]["n"]}', logs[b]["omed"]]
    v["tbl_disturb"] = table(head, [
        pair("Cube pushed 15 mm at the start of the approach", "push_open15", "push_pbvs15"),
        pair("Cube pushed 25 mm at the start of the approach", "push_open", "push_pbvs"),
        pair("Camera mount wrong by 2&deg;", "he2_open", "he2_pbvs"),
    ])
    v["he2_track_row"] = f'{logs["he2_track"]["placed"]}/{logs["he2_track"]["n"]}'

    motions = {"line": "Straight", "arc": "Arc, radius 10-16 cm", "s": "S-curve", "var": "Variable speed, short pauses"}
    rows = []
    for key, label in motions.items():
        for flt, flt_label in (("cv", "EKF"), ("ct", "CT-EKF")):
            r = motion_log(f"motion_{key}_{flt}")
            rows.append([f"{label}, {flt_label}", f'{r["placed"]}/{r["n"]}', f'{r["omed"]} / {r["omax"]}', f'{r["pmed"]} / {r["pmax"]}', r["delay"], f'{r["rej"]} / {r["reinit"]}'])
    v["tbl_motion"] = table(["Motion at 3-4 cm/s, closed loop, seeds 0-19", "Placed", "Grasp offset median / max (mm)", "Place error median / max (mm)", "Stop delay median (s)", "Rejected / re-initialised"], rows)

    rows = []
    for pause in ("1.0", "1.5", "2.0", "2.5", "3.0", "4.0"):
        r = motion_log(f"pause_{pause}")
        rows.append([f"{pause} s", f'{r["early"]}/{r["n"]}', f'{r["placed"]}/{r["n"]}', r["omax"]])
    v["tbl_pause"] = table(["Mid-path pause", "Stop declared early", "Placed", "Largest grasp offset (mm)"], rows)

    flt = filter_rows("filters")
    rows = []
    for motion, label in (("line", "Straight"), ("arc", "Arc"), ("s_curve", "S-curve")):
        for name in ("KF", "EKF", "CT-EKF", "EKF sigma_a 0.03"):
            r = flt[(motion, name)]
            shown = name.replace("sigma_a", "&sigma;<sub>a</sub>")
            rows.append([f"{label}, {shown}", r["pos"], r["vel"], r["nees"], f'{r["stop"]} ({r["stops"]}/{r["n"]})', f'{r["gap"]} ({r["gapmax"]})'])
    v["tbl_filters"] = table(["Motion, filter", "Position RMS (mm)", "Velocity RMS (mm/s)", "NEES median (ideal 5)", "Stop delay (s), stops found", "Error after 1 s without frames, median (max) (mm)"], rows)

    fr = filter_rows("filters_r")
    rows = []
    for scale in ("0.1", "1", "10"):
        r = fr[("line", f"EKF, R x {scale}")]
        rows.append([f"R &times; {scale}", r["pos"], r["vel"], r["nees"], r["nis"], f'{r["rej"]} / {r["reinit"]}', f'{r["stop"]} ({r["stops"]}/{r["n"]})'])
    v["tbl_r_scale"] = table(["Cube EKF, straight motion", "Position RMS (mm)", "Velocity RMS (mm/s)", "NEES median (ideal 5)", "NIS / dim (ideal 1)", "Rejected / re-initialised", "Stop delay (s), stops found"], rows, highlight=(1,))
    v["f_line_ekf_nees"] = flt[("line", "EKF")]["nees"]
    v["r10_nees"] = fr[("line", "EKF, R x 10")]["nees"]
    v["f_arc_ct"], v["f_arc_ekf"] = flt[("arc", "CT-EKF")]["gap"], flt[("arc", "EKF")]["gap"]

    ekf_text = (LOGS / "chk_cube_ekf.log").read_text()
    nominal_text = (LOGS / "chk_cube_ekf_nominal.log").read_text()
    summary = re.compile(r"median NEES ([\d.]+) \(consistent = 5\), NIS ([\d.]+) \(consistent = 4\), pos RMS ([\d.]+) mm, vel RMS ([\d.]+) mm/s; stop declared (\d+)/(\d+), median delay ([\d.]+) s, false stops (\d+)")
    a, b = summary.search(ekf_text), summary.search(nominal_text)
    v["tbl_nees"] = table(["Camera model used by the filter", "NEES median (ideal 5)", "NIS (ideal 4)", "Position RMS (mm)", "Velocity RMS (mm/s)", "Stops found", "Stop delay (s)", "False stops"], [
        ["Calibrated (1.3 mm, 0.11&deg; off the truth)", a[1], a[2], a[3], a[4], f"{a[5]}/{a[6]}", a[7], a[8]],
        ["Exact mount and nominal K", b[1], b[2], b[3], b[4], f"{b[5]}/{b[6]}", b[7], b[8]],
    ])
    for key, group in (("nees_cal", 1), ("rms_cal", 3), ("rms_nom", 3), ("nees_nom", 1)):
        v[key] = (b if "nom" in key else a)[group]

    kf_text = (LOGS / "chk_joint_kf.log").read_text()
    rows = []
    for m in re.finditer(r"sigma_d\s+(\d+):\s+qd RMS \[(.*?)\]\s+NEES\[q,qd\] \[(.*?)\]\s+NIS \[(.*?)\]", kf_text):
        rng = lambda s, d: "{:.{d}f}-{:.{d}f}".format(min(map(float, s.split())), max(map(float, s.split())), d=d)  # noqa: E731
        rows.append([m[1], rng(m[2], 3), rng(m[3], 2), rng(m[4], 2)])
    fd = re.search(r"finite-difference qd RMS \[(.*?)\]", kf_text)[1].split()
    rows.append(["finite differences", f"{min(map(float, fd)):.3f}-{max(map(float, fd)):.3f}", "-", "-"])
    v["tbl_sigma_d"] = table(["&sigma;<sub>d</sub> (N m/&radic;s), all joints", "q&#775; RMS (rad/s)", "NEES[q, q&#775;] (ideal 2)", "NIS (ideal 1)"], rows, highlight=(2,))

    def control(name):
        t = (LOGS / f"{name}.log").read_text()
        rms = re.findall(r"RMS error after \d s \(rad\) \[(.*?)\]", t)
        peak = re.findall(r"peak \|tau\| \(N m\)\s+\[(.*?)\]", t)
        imp = re.search(r"peak deviation ([\d.]+) rad, back within 0.01 rad after ([\d.]+) s", t)
        mx = lambda s: 1e3 * max(map(float, s.split()))  # noqa: E731
        return {"hold": mx(rms[0]), "noff": mx(rms[1]), "move": mx(rms[2]), "peak": peak[0].split(), "imp": 1e3 * float(imp[1]), "rec": imp[2]}
    q, d = control("chk_control_lqr"), control("chk_control_pd")
    v["tbl_control"] = table(["Test", "LQR (Bryson)", "PD (5 Hz)"], [
        ["Hold at observe, largest joint RMS error (mrad)", f'{q["hold"]:.1f}', f'{d["hold"]:.1f}'],
        ["Same, without the d&#770; feed-forward (mrad)", f'{q["noff"]:.1f}', f'{d["noff"]:.1f}'],
        ["Quintic move in 2 s, largest joint RMS error (mrad)", f'{q["move"]:.1f}', f'{d["move"]:.1f}'],
        ["40 N m, 100 ms impulse on lift: peak deviation (mrad)", f'{q["imp"]:.1f}', f'{d["imp"]:.1f}'],
        ["Recovery to within 10 mrad (s)", q["rec"], d["rec"]],
        ["Peak torque while holding, lift / elbow (N m)", f'{float(q["peak"][1]):.1f} / {float(q["peak"][2]):.1f}', f'{float(d["peak"][1]):.1f} / {float(d["peak"][2]):.1f}'],
    ])

    runs = lab_runs()
    hw = [r for r in runs if r["group"] == "hardware"]
    def by_mode(mode):
        sel = [r for r in hw if r["mode"] == mode]
        return sel, sum(r["ok"] for r in sel)
    def hw_row(mode, label, note):
        sel, ok = by_mode(mode)
        return [label, f"{ok}/{len(sel)}", note]
    v["tbl_hw"] = table(["Mode", "Completed", "What stopped the others"], [
        hw_row("moving", "<strong>Moving cube</strong>: track, measure, grasp, place", "Measurement after TRACK succeeded on 1 of 21 frames, cause not established (1); cube beyond IK reach (1); cube not detected at the start, suspected tilted on a slot (1); stopped by hand because the 5 mm/s stop test never held (1)"),
        hw_row("static", "<strong>Static cube</strong>: measure, grasp, place", "Cube moved away after it was measured; the gripper closed on nothing and the program stopped itself (1)"),
        hw_row("track", "Track only", "-"),
        hw_row("pick", "First pick, starting above the cube", "Fingertip contact, UR protective stop (1)"),
    ], numeric_from=9)
    moving_ok = [r for r in hw if r["mode"] == "moving" and r["ok"]]
    placed = [r for r in moving_ok if r["placed_x_mm"] is not None]
    v["hw_moving_ok"], v["hw_moving_n"] = by_mode("moving")[1], len(by_mode("moving")[0])
    v["hw_static_ok"], v["hw_static_n"] = by_mode("static")[1], len(by_mode("static")[0])
    v["placed_n"] = len(placed)
    v["placed_x"] = f'{max(r["placed_x_mm"] for r in placed) - min(r["placed_x_mm"] for r in placed):.1f}'
    v["placed_y"] = f'{max(r["placed_y_mm"] for r in placed) - min(r["placed_y_mm"] for r in placed):.1f}'
    v["placed_yaw_lo"] = f'{min(r["placed_yaw_deg"] for r in placed):+.1f}'.replace("-", "&minus;")
    v["placed_yaw_hi"] = f'{max(r["placed_yaw_deg"] for r in placed):+.1f}'

    def sim_place_err(prefix):
        out = []
        for path in sorted((LOGS / "lab_sim").glob(f"{prefix}_*.log")):
            m = re.search(r"([\d.]+) mm from the place position", path.read_text())
            if m and re.search(r"^OK:", path.read_text(), re.M):
                out.append(float(m[1]))
        return out

    def group_col(sel, place):
        ok = [r for r in sel if r["ok"]]
        per = lambda key: f'{median(r[key] for r in ok):.1f}'  # noqa: E731
        share = sum(r["accepted"] for r in ok) / sum(r["updates"] for r in ok)
        return [f"{len(ok)}/{len(sel)}", f'{max(r["cube_speed_max_mm_s"] for r in ok):.0f}', per("elapsed_s"), f"{100 * share:.0f} %", per("no_cube"),
                per("rejected"), per("reinit"), per("tcp_err_median_mm"), f'{max(r["tcp_err_max_mm"] for r in ok):.0f}', place]

    slow = [r for r in runs if r["group"] == "sim_slow" and r["mode"] == "moving"]
    fast = [r for r in runs if r["group"] == "sim_fast"]
    final = [r for r in hw if r["id"] in ("20261102_155933", "20261102_160052")]
    se, fe = sim_place_err("moving"), sim_place_err("fast")
    cols = [
        group_col(slow, f"{median(se):.1f} / {max(se):.1f} (truth)"),
        group_col(fast, f"{median(fe):.1f} / {max(fe):.1f} (truth)"),
        group_col(by_mode("moving")[0], "not measured"),
        group_col(final, "not measured"),
    ]
    labels = ["Runs completed", "Largest estimated cube speed (mm/s)", "Run time median (s)", "Frames accepted by the EKF",
              "Frames without a cube estimate, median per run", "Rejected, median per run", "Re-initialised, median per run",
              "TCP tracking error, median of run medians (mm)", "TCP tracking error, largest (mm)", "Place error median / max (mm)"]
    v["tbl_simhw"] = table(["Moving cube, lab pipeline", "Sim, scripted 1-1.5 cm/s", "Sim, scripted 5-6 cm/s", "Hardware, all runs", "Hardware, final code"],
                           [[labels[i]] + [c[i] for c in cols] for i in range(len(labels))])
    v["sim_static"] = f"{sum(r['ok'] for r in runs if r['group'] == 'sim_slow' and r['mode'] == 'static')}/{sum(1 for r in runs if r['group'] == 'sim_slow' and r['mode'] == 'static')}"
    v["fast_ok"] = sum(r["ok"] for r in fast)
    v["fast_n"] = len(fast)

    def occ(name, label):
        log = logs[name]
        states = log["fail_states"]
        collided = sum(len(states.get(s, [])) for s in ("APPROACH", "DESCEND", "TRANSPORT", "LIFT", "PLACE"))
        refused = sum("no collision-free grasp yaw" in f for f in log["failures"])
        return {"name": label, "placed": log["placed"], "collided": collided, "refused": refused,
                "other": log["n"] - log["placed"] - collided - refused}
    occ_rows = [occ("l1_off", "Ignore occluders"), occ("l1_conservative", "Conservative"), occ("l1_active", "Active perception")]
    for r, key in zip(occ_rows, ("off", "conservative", "active")):
        for field in ("placed", "collided", "refused", "other"):
            v[f"occ_{key}_{field}"] = r[field]

    site_data = {
        "ekf_trace": ekf_trace(),
        "push_rows": [
            {"name": "No push", "values": {"open": logs["static_open"]["offsets"][:20], "pbvs": logs["static_pbvs"]["offsets"][:20]}},
            {"name": "Push 15 mm", "values": {"open": logs["push_open15"]["offsets"], "pbvs": logs["push_pbvs15"]["offsets"]}},
            {"name": "Push 25 mm", "values": {"open": logs["push_open"]["offsets"], "pbvs": logs["push_pbvs"]["offsets"]}},
        ],
        "occluder_rows": occ_rows,
    }
    v["site_data"] = json.dumps(site_data, separators=(",", ":")).replace("</", "<\\/")
    v["push25_open_missed"] = sum(o is None for o in logs["push_open"]["offsets"])

    template = (ROOT / "template.html").read_text()
    used = set()

    def fill(match):
        used.add(match[1])
        if match[1] not in v:
            raise KeyError(f"template.html uses {{{{{match[1]}}}}}, which build.py does not define")
        return str(v[match[1]])

    (ROOT / "index.html").write_text(re.sub(r"\{\{([\w.]+)\}\}", fill, template))
    print(f"wrote website/index.html, {len(used)} values filled")

    # The Data button downloads this. Fixed timestamps keep the archive identical when the logs have not changed.
    logs = ROOT / "data/logs"
    files = sorted(f for f in logs.rglob("*") if f.is_file())
    with zipfile.ZipFile(ROOT / "data/logs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for f in files:
            info = zipfile.ZipInfo(str(Path("logs") / f.relative_to(logs)), (2026, 11, 6, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, f.read_bytes())
    print(f"wrote website/data/logs.zip, {len(files)} files")


if __name__ == "__main__":
    build()
