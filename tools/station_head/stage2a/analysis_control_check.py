"""Put the analysis control on the same footing as the forecast controls.

The analysis at the nearest point is deterministic, so its fair CRPS is exactly its
absolute error, while the forecast controls are ten-member ensembles whose fair CRPS
is always lower than the absolute error of any one of their members. Comparing the
two directly therefore flatters the ensemble, and a reader who sees the analysis with
the larger number could wrongly conclude that the analysis is the worse product.

This script prints, on the same rows, four numbers per target: the mean absolute
error of the analysis at the nearest point, the mean absolute error of a single
member of the nearest-point forecast, the mean absolute error of the ensemble mean of
that forecast, and the fair CRPS of that ensemble. The comparison to make is the
analysis against the single member and against the ensemble mean; the fair CRPS is
printed only so that the table in the run folder can be read beside it.

    python analysis_control_check.py --data-dir <assembled dataset directory>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def fair_crps_rows(members: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = members.shape[1]
    skill = np.abs(members - y[:, None]).mean(axis=1)
    if m > 1:
        pair = np.abs(members[:, :, None] - members[:, None, :]).sum(axis=(1, 2))
        return skill - pair / (2.0 * m * (m - 1))
    return skill


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cases", default=None, help="comma-separated case ids; default is all")
    ap.add_argument("--seen-only", action="store_true", help="drop the held-out stations")
    ap.add_argument("--stable-only", action="store_true")
    args = ap.parse_args()

    paths = sorted(Path(args.data_dir).glob("*.npz"))
    if args.cases:
        want = set(c.strip() for c in args.cases.split(","))
        paths = [p for p in paths if p.stem in want]
    acc = {k: [] for k in ["analysis_mae", "member_mae", "mean_mae", "fair_crps"]}
    n_total = 0
    for p in paths:
        z = np.load(p, allow_pickle=False)
        y = z["y"].astype(np.float64)
        members = z["ctrl_near"].astype(np.float64)
        an = z["analysis"].astype(np.float64)
        keep = np.isfinite(an)
        if args.seen_only:
            keep &= ~z["holdout"]
        if args.stable_only:
            keep &= z["stable"]
        z.close()
        if not keep.any():
            continue
        y, members, an = y[keep], members[keep], an[keep]
        acc["analysis_mae"].append(np.abs(an - y))
        acc["member_mae"].append(np.abs(members - y[:, None]).mean(axis=1))
        acc["mean_mae"].append(np.abs(members.mean(axis=1) - y))
        acc["fair_crps"].append(fair_crps_rows(members, y))
        n_total += int(keep.sum())

    print("cases %d, rows %d, seen_only=%s stable_only=%s"
          % (len(paths), n_total, args.seen_only, args.stable_only))
    for k in ["analysis_mae", "member_mae", "mean_mae", "fair_crps"]:
        v = np.concatenate(acc[k]) if acc[k] else np.array([np.nan])
        print("  %-14s %.4f" % (k, float(np.mean(v))))


if __name__ == "__main__":
    main()
