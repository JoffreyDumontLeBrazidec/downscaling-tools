"""Paired shape differences between two arms, with date-clustered errors.

    python -m eval.evaluators.shape.paired --a <results_dir_A> --b <results_dir_B> \
        [--label-a RW50k --label-b R47k] --out paired.csv

A pair is one (date, member) cell present in both arms (same dates, members, leads
and seeds), exactly as aggregate_p1.py / shape_diff_RW50k.py defined it. The paired
difference is the mean of A - B over the cells. Two standard errors are reported:
``se_date`` resamples whole dates (members of a date together, 4000 resamples), the
one to use; ``se_naive`` is the published formula (standard deviation of the
differences over sqrt(number of pairs)), which treats the members of a date as
independent and is kept only to compare with the tables of 2026-09-21/22.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from eval.shared.date_bootstrap import DEFAULT_N_BOOT, DEFAULT_SEED, boot_mean

from .runner import DEFAULT_WINDOWS, read_rows

STATS = ("elong_frac_gt3", "ell_ratio_median", "A_open_ocean")
COLS = ["step", "var", "window", "field", "band", "statistic", "A_mean", "B_mean",
        "paired_diff", "se_date", "ci_lo", "ci_hi", "se_naive", "diff_over_se_date",
        "n_pairs", "n_dates", "diff_over_2se_date"]


def _index(rows, field, band):
    return {(r["step"], r["var"], r["date"], int(r["member"])): r for r in rows
            if r["field"] == field and r["band"] == band}


def paired(rows_a, rows_b, *, field="draw", band="mid", windows=None, n_boot=DEFAULT_N_BOOT,
           seed=DEFAULT_SEED, stats=STATS):
    windows = dict(DEFAULT_WINDOWS, **(windows or {}))
    A, B = _index(rows_a, field, band), _index(rows_b, field, band)
    common = set(A) & set(B)
    win = lambda d: windows.get(d[:6], d[:6])  # noqa: E731
    out = []
    for step in sorted({c[0] for c in common}):
        for var in sorted({c[1] for c in common}):
            wins = sorted({win(c[2]) for c in common if c[0] == step and c[1] == var})
            for window in wins + (["both"] if len(wins) > 1 else []):
                cells = sorted(c for c in common if c[0] == step and c[1] == var
                               and (window == "both" or win(c[2]) == window))
                if not cells:
                    continue
                dates = [c[2] for c in cells]
                for st in stats:
                    a = np.array([float(A[c][st]) if A[c][st] != "" else np.nan for c in cells])
                    b = np.array([float(B[c][st]) if B[c][st] != "" else np.nan for c in cells])
                    d = a - b
                    ok = np.isfinite(d)
                    bd = boot_mean(dates, d, n_boot, seed)
                    se_n = float(d[ok].std(ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else np.nan
                    se = bd["se"]
                    out.append({"step": step, "var": var, "window": window, "field": field,
                                "band": band, "statistic": st, "A_mean": float(np.nanmean(a)),
                                "B_mean": float(np.nanmean(b)), "paired_diff": bd["value"],
                                "se_date": se, "ci_lo": bd["ci_lo"], "ci_hi": bd["ci_hi"],
                                "se_naive": se_n,
                                "diff_over_se_date": (bd["value"] / se) if se else np.nan,
                                "n_pairs": int(ok.sum()), "n_dates": bd["n_dates"],
                                "diff_over_2se_date": int(bool(se) and abs(bd["value"]) > 2 * se)})
    return out


def write(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for r in rows:
            w.writerow([r[c] if isinstance(r[c], (str, int)) else
                        ("" if r[c] is None or not np.isfinite(r[c]) else f"{r[c]:.6g}") for c in COLS])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--a", required=True, help="results dir (or rows.csv) of arm A")
    p.add_argument("--b", required=True, help="results dir (or rows.csv) of arm B")
    p.add_argument("--out", required=True)
    p.add_argument("--field", default="draw")
    p.add_argument("--band", default="mid")
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    a = p.parse_args(argv)

    def rows_of(x):
        x = Path(x)
        return read_rows(x / "rows.csv" if x.is_dir() else x)

    rows = paired(rows_of(a.a), rows_of(a.b), field=a.field, band=a.band, n_boot=a.n_boot)
    write(rows, a.out)
    print(f"wrote {a.out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
