#!/usr/bin/env python3
"""Station scores of one row of the extremes set XSET-1: eval.cli's obs_points evaluator with the set's configuration.

Fair CRPS and spread of 2t (raw and height-corrected), 10ff, msl and 12 h tp at every STVL station of the cache; rain
tail: fair twCRPS, fair Brier, observed count and forecast frequency above 20, 50 and 100 mm per 12 h, at the nearest
point and as 18 and 36 km disc means; wind tail: the same above 15 and 20 m/s. The input row is the native O320 x of the
files (rule 41), height-corrected with the eefo orography column. Global point set: no box-edge strip.

    python3 xset_score_stations.py --pred DIR --label GL512_500k --cache CACHE --out DIR

The input orography can differ between model cycles (48r1 to 11 Nov 2024, 49r1 after): the evaluator runs once per cycle
with that cycle's cache column (h_o320_eefo_48r1, h_o320_eefo_49r1) and the per-date lines are concatenated (exact: the
lines are per date). Writes DIR/obs_points_<cycle>/ and DIR/obs_points/scores_per_date.csv (what xset_table.py reads).
"""
import argparse
import logging
from pathlib import Path

REGIONS = {
    "all": None,
    "north_30N": [30.0, 90.01, -180.0, 180.0],
    "tropics_30S_30N": [-30.0, 30.0, -180.0, 180.0],
    "south_30S": [-90.0, -30.0, -180.0, 180.0],
    "europe": [35.0, 75.0, -12.5, 42.5],
    "north_america": [15.0, 72.0, -170.0, -50.0],
    "east_asia": [5.0, 55.0, 95.0, 150.0],
    # CorrDiff baseline (17 Nov 2026): the E2 box [10, 70] N x [-80, 40] E minus 0.5 deg, so every station of the region
    # and its 36 km disc fall on points that CorrDiff predicts (it writes NaN outside the box; obs_points drops those)
    "e2_box_in": [10.5, 69.5, -79.5, 39.5],
}


def config(cache, input_height):
    return {"cache": str(cache), "vars": ["2t", "10ff", "msl", "tp"], "box_edge_deg": 0.0,
            "thresholds": [20.0, 50.0, 100.0], "wind_thresholds": [15.0, 20.0], "radii": [0.0, 18.0, 36.0],
            "input_height_column": input_height, "regions": REGIONS, "height_classes": {}, "seeps": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dates-csv", default=str(Path(__file__).parent / "lists" / "dates_xset1.csv"),
                    help="date list with a cycle column: the input row's height column is chosen per model cycle")
    ap.add_argument("--input-height", default="h_o320_eefo_{cycle}",
                    help="cache column of the input orography; {cycle} is replaced by the date's cycle (48r1, 49r1)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    import csv
    import re

    import pandas as pd
    from eval.evaluators import obs_points
    have = {re.match(r"predictions_(\d{8})_step", f.name).group(1) for f in Path(a.pred).glob("predictions_*_step*.nc")}
    cycles = {}
    for r in csv.DictReader(open(a.dates_csv)):
        if r["date"] in have:
            cycles.setdefault(r["cycle"] if "{cycle}" in a.input_height else "all", []).append(r["date"])
    missing = have - {d for v in cycles.values() for d in v}
    if missing:
        raise SystemExit(f"prediction dates not in {a.dates_csv}: {sorted(missing)}")
    out = Path(a.out)
    parts = []
    for cyc, dates in sorted(cycles.items()):
        cfg = config(a.cache, a.input_height.format(cycle=cyc))
        cfg["dates"] = dates
        o = obs_points.run(a.pred, {}, cfg, output_dir=out / f"obs_points_{cyc}", run_label=a.label, overwrite=a.overwrite)
        df = pd.read_csv(Path(o) / "scores_per_date.csv", dtype={"date": str})
        df["cycle"] = cyc
        parts.append(df)
        print(f"STATIONS {a.label} cycle {cyc}: {len(dates)} dates, {len(df)} lines, input height {cfg['input_height_column']}", flush=True)
    (out / "obs_points").mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(out / "obs_points" / "scores_per_date.csv", index=False)
    print("STATIONS_DONE", a.label, out / "obs_points" / "scores_per_date.csv", flush=True)


if __name__ == "__main__":
    main()
