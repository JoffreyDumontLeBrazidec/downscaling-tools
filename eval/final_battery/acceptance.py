"""Acceptance check of a battery run against the committed 17 Nov reference tables (row GLB450e2_m).

    python -m eval.final_battery.acceptance <OUT> <DOCS_SUPPORT>

<OUT> = a battery output folder made on GLB450e2_m (and, for the 2d/skt part, GLB450e2_m10 at +12/+24 h);
<DOCS_SUPPORT> = docs epics/fine-scale-o320-o1280/in-progress/support. Every reference CSV that has a counterpart in
<OUT> is compared cell by cell (numeric columns, rows matched on the non-numeric columns); prints the largest
relative difference per file and ACCEPT / REJECT (tolerance 1e-3 relative, 1e-6 absolute: the CSVs are written with
4-6 significant digits). Files absent on either side are listed, not failed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PAIRS = (  # (reference path under support, path under OUT)
    *[(f"20261117_realism_t2m/tables/{n}", f"terrain/tables/{n}") for n in (
        "T1_bias_rmse_binned.csv", "T1e_season.csv", "T2_lapse_slopes.csv", "T3_hydrostatic.csv", "T3_hydrostatic_summary.csv",
        "T5_o320_imprint.csv", "T6_tails.csv", "T7_large_scale.csv", "T8_spread.csv")],
    *[(f"20261117_realism_t2m/tables/z/{n}", f"terrain/z/tables/{n}") for n in (
        "Z1_spread_by_scale.csv", "Z2_msl_move_by_lst.csv", "Z3_sp_diagnosed.csv", "Z4_2t_spread_coast_dz.csv")],
    ("20261117_realism_t2m/tables/z/Z6_sp_residual_baseline.csv", "terrain/z6/tables/Z6_sp_residual_baseline.csv"),
    *[(f"20261117_realism_t2m/tables/d2/{n}", f"terrain/d2/tables/{n}") for n in (
        "D1_2d_skt_binned.csv", "D2_lapse_2d_skt.csv", "D4_2d_vs_2t.csv", "D4_skt_minus_2t.csv", "DH_fine_amplitude.csv")],
    ("20261117_realism_consistency/results/scalars.csv", "consistency/edge1/scalars.csv"),
    ("20261117_realism_consistency/results/ratios.csv", "consistency/edge1/ratios.csv"),
    ("20261117_realism_consistency/results/stationary.csv", "consistency/edge1/stationary.csv"),
    ("20261117_realism_consistency/results_edge5/ratios.csv", "consistency/edge5/ratios.csv"),
    *[(f"20261117_realism_wind/out{L}/small/{n}", f"wind/out{L:03d}/small/{n}") for L in (24, 72)
      for n in ("band_table.csv", "class_dates.csv", "spread_dates.csv", "tile_summary.csv")],
    *[(f"20261117_realism_wind/out{L}/cases.csv.gz", f"wind/out{L:03d}/cases.csv") for L in (24, 72)],
)


def compare(ref: Path, new: Path, rtol=1e-3, atol=1e-6) -> tuple[float, str]:
    import pandas as pd
    def rd(p):
        with open(p, "rb") as fh:
            gz = fh.read(2) == b"\x1f\x8b"
        return pd.read_csv(p, compression="gzip" if gz else None)
    a, b = rd(ref), rd(new)
    if ref.name == "ratios.csv":  # three colliding 'se' headers
        cols = ["key", "lead", "mt", "mt_se", "it", "it_se", "mi", "mi_se", "n_dates"]
        a.columns, b.columns = cols[: len(a.columns)], cols[: len(b.columns)]
    num = [c for c in a.columns if c in b.columns and pd.api.types.is_float_dtype(a[c])]
    keys = [c for c in a.columns if c in b.columns and c not in num]
    for df in (a, b):
        for c in keys:
            df[c] = df[c].astype(object).where(df[c].notna(), "").astype(str)
    if len(a) != len(b):
        return float("inf"), f"rows {len(a)} vs {len(b)}"
    if keys:
        a = a.sort_values(keys, kind="mergesort").reset_index(drop=True)
        b = b.sort_values(keys, kind="mergesort").reset_index(drop=True)
        if not (a[keys].values == b[keys].values).all():
            return float("inf"), "key columns differ"
    worst, where = 0.0, ""
    for c in num:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        both_nan = np.isnan(x) & np.isnan(y)
        d = np.abs(x - y) / (np.abs(x) * rtol + atol) * rtol
        d = np.where(both_nan, 0.0, d)
        d = np.where(np.isnan(d), np.inf, d)
        if d.size and d.max() > worst:
            worst, where = float(d.max()), c
    return worst, where


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    out, sup = Path(argv[0]), Path(argv[1])
    bad = 0
    for r, n in PAIRS:
        rp, np_ = sup / r, out / n
        if not rp.exists() or not np_.exists():
            print(f"ABSENT  {n}  (ref {'ok' if rp.exists() else 'missing'}, new {'ok' if np_.exists() else 'missing'})")
            continue
        w, where = compare(rp, np_)
        ok = w <= 1e-3
        bad += not ok
        print(f"{'SAME   ' if ok else 'DIFFER '} {n}  max rel diff {w:.2e} {where}")
    print("ACCEPT" if bad == 0 else f"REJECT ({bad} files differ)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
