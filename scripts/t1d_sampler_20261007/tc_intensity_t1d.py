"""Per-draw storm intensity of the T1d stage A box runs (2026-10-07).

Copy of the T1 stage 3 tc_intensity.py (docs 20260930_T1_stage1_results/stage3/scripts/common/) with one change: the
runs are arguments (name=run_root) instead of a hard-coded table. Same reduction: for each event box of the tc
evaluator (support_mode native; boxes from <run_root>/<eval subdir>/evaluators/tc/stats.json), date, lead (24 and 120)
and member, the maximum of sqrt(10u^2 + 10v^2) and the minimum of msl (hPa) over the native points in the box. The truth
(y) is written once, from the first run, as member "truth". Check: the max over members and files equals the
evaluator's pooled wind_max / mslp_min (printed at the end; the evaluator pools the same two leads here).
Usage: python tc_intensity_t1d.py --out <tsv> [--eval-subdir eval_stageA] name=<run root> [name=<run root> ...]
"""
import argparse
import glob
import json
import os
import sys

import netCDF4
import numpy as np


def boxes(eval_dir):
    ev = json.load(open(eval_dir + "/evaluators/tc/stats.json"))["events"]
    out = {}
    for name, e in ev.items():
        b = e["comparison_contract"]["geographic_box"]
        exp = [r for r in e["extreme_tail"]["rows"] if r["exp"] not in ("input O320", "target O1280")][0]
        out[name] = (b, exp["wind_max"], exp["mslp_min"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-subdir", default="eval_stageA")
    ap.add_argument("runs", nargs="+", help="name=run_root (run_root/predictions and run_root/<eval subdir>)")
    args = ap.parse_args()
    runs = [r.split("=", 1) for r in args.runs]
    truth_arm = runs[0][0]
    rows = ["arm\tstorm\tdate\tlead\tmember\twind_max\tmslp_min"]
    checks = []
    for arm, root in runs:
        pdir, edir = os.path.join(root, "predictions"), os.path.join(root, args.eval_subdir)
        bx = boxes(edir)
        agg = {s: [-np.inf, np.inf] for s in bx}
        files = sorted(glob.glob(pdir + "/predictions_*_step*.nc"))
        if not files:
            sys.exit(f"FATAL no prediction files in {pdir}")
        for f in files:
            base = os.path.basename(f)
            date, lead = base.split("_")[1], int(base.split("step")[1][:3])
            if lead not in (24, 120):
                continue
            with netCDF4.Dataset(f) as d:
                d.set_auto_mask(False)
                lat = np.asarray(d["lat_hres"][:], dtype=np.float64)
                lon = np.asarray(d["lon_hres"][:], dtype=np.float64)
                lon = np.where(lon > 180, lon - 360, lon)
                ws = list(d["weather_state"][:])
                mem = [int(m) for m in d["ensemble_member"][:]]
                iu, iv, im = ws.index("10u"), ws.index("10v"), ws.index("msl")
                yp = d["y_pred"][0]
                yt = d["y"][0]
                for storm, (b, _, _) in bx.items():
                    sel = (lat >= b["south"]) & (lat <= b["north"]) & (lon >= b["west"]) & (lon <= b["east"])
                    for j, m in enumerate(mem):
                        wmax = float(np.nanmax(np.hypot(yp[j, sel, iu], yp[j, sel, iv])))
                        pmin = float(np.nanmin(yp[j, sel, im])) / 100.0
                        agg[storm][0] = max(agg[storm][0], wmax)
                        agg[storm][1] = min(agg[storm][1], pmin)
                        rows.append(f"{arm}\t{storm}\t{date}\t{lead}\t{m}\t{wmax:.3f}\t{pmin:.2f}")
                    if arm == truth_arm:
                        wt = float(np.nanmax(np.hypot(yt[0, sel, iu], yt[0, sel, iv])))
                        pt = float(np.nanmin(yt[0, sel, im])) / 100.0
                        rows.append(f"truth\t{storm}\t{date}\t{lead}\ttruth\t{wt:.3f}\t{pt:.2f}")
        for storm, (b, ew, ep) in bx.items():
            checks.append(f"CHECK {arm} {storm}: pooled wind_max={agg[storm][0]:.3f} (evaluator {ew:.3f})  "
                          f"mslp_min={agg[storm][1]:.2f} (evaluator {ep:.2f})")
        print(f"done {arm}", flush=True)
    open(args.out, "w").write("\n".join(rows) + "\n")
    print("\n".join(checks))
    print(f"rows={len(rows) - 1}")


if __name__ == "__main__":
    main()
