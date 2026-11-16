#!/usr/bin/env python3
"""Station scores of one row of the extremes set XSET-1: eval.cli's obs_points evaluator with the set's configuration.

Fair CRPS and spread of 2t (raw and height-corrected), 10ff, msl and 12 h tp at every STVL station of the cache; rain
tail: fair twCRPS, fair Brier, observed count and forecast frequency above 20, 50 and 100 mm per 12 h, at the nearest
point and as 18 and 36 km disc means; wind tail: the same above 15 and 20 m/s. The input row is the native O320 x of the
files (rule 41), height-corrected with the eefo orography column. Global point set: no box-edge strip.

    python3 xset_score_stations.py --pred DIR --label GL512_500k --cache CACHE --out DIR [--input-height h_o320_eefo]

Writes DIR/obs_points/ (scores_per_date.csv is what xset_table.py reads).
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
    ap.add_argument("--input-height", default="h_o320_eefo")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    from eval.evaluators import obs_points
    out = obs_points.run(a.pred, {}, config(a.cache, a.input_height), output_dir=Path(a.out) / "obs_points",
                         run_label=a.label, overwrite=a.overwrite)
    print("STATIONS_DONE", a.label, out, flush=True)


if __name__ == "__main__":
    main()
