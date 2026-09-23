"""shape evaluator: the feature-shape instrument of 2026-09-16/17, ported unchanged.

The statistics (flow-relative anisotropy index, two-point correlation ellipse in
4x4 degree open-ocean windows, morphology of the connected features above the 90th
percentile), the seven-Gaussian band pass (mid band 40-109 km, band 1-2 21-40 km),
the box (10-40 N, 100-58 W), the open-ocean mask and the 0.09 degree regrid are the
verbatim code of ``shape_stats.py`` / ``shape_fullrung.py`` (see instrument.py and
fullrung.py, which record the source and md5 of each). This module only replaces
the command-line loop: it walks the prediction files of a directory, writes the
same per-member CSV rows (one part file per prediction file, identical columns),
and adds what the original lacked: means pooled over dates and members with a
date-clustered bootstrap standard error (dates resampled with replacement, members
of a date kept together).

Paired differences between two arms are made by ``python -m eval.evaluators.shape.paired``
from two results directories.

eval_config keys (all optional): steps (default [24, 120]), dates, vars ("10u,10v"),
bands ("mid,b12"), members, n_boot, seed, geom_cache, mask_cache, probe_file,
stats_file, windows ({"YYYYMM": label}).
"""
from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from eval.discovery.predictions import find_predictions
from eval.shared.date_bootstrap import DEFAULT_N_BOOT, DEFAULT_SEED, boot_mean

LOG = logging.getLogger(__name__)

DEFAULT_WINDOWS = {"202509": "humberto", "202308": "idalia"}
SUMMARY_STATS = ("elong_frac_gt3", "ell_ratio_median", "A_open_ocean", "A_all_interior",
                 "elong_median", "ell_major_km_median", "ell_minor_km_median", "n_components")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _configure(eval_config: dict):
    from . import fullrung as F
    from . import instrument as S

    if eval_config.get("geom_cache"):
        S.GEOMCACHE = str(eval_config["geom_cache"])
    if eval_config.get("mask_cache"):
        S.DONOR_MASKCACHE = str(eval_config["mask_cache"])
    if eval_config.get("stats_file"):
        S.STATS = str(eval_config["stats_file"])
    if eval_config.get("probe_file"):
        F.PROBE = str(eval_config["probe_file"])
    return F, S


def read_rows(path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def summarise(rows: list[dict], windows: dict, n_boot: int, seed: int) -> list[dict]:
    """Pooled mean over (date, member) cells per (step, var, window, field, band,
    statistic), with the date-clustered bootstrap error."""
    out = []
    win_of = lambda d: windows.get(str(d)[:6], str(d)[:6])  # noqa: E731
    keys = sorted({(r["step"], r["var"], r["field"], r["band"]) for r in rows})
    wins = sorted({win_of(r["date"]) for r in rows})
    for step, var, field, band in keys:
        sub = [r for r in rows if (r["step"], r["var"], r["field"], r["band"]) == (step, var, field, band)]
        for window in wins + (["both"] if len(wins) > 1 else []):
            ws = [r for r in sub if window == "both" or win_of(r["date"]) == window]
            if not ws:
                continue
            dates = [r["date"] for r in ws]
            for st in SUMMARY_STATS:
                vals = np.array([float(r[st]) if r[st] != "" else np.nan for r in ws])
                b = boot_mean(dates, vals, n_boot, seed)
                out.append({"step": step, "var": var, "window": window, "field": field,
                            "band": band, "statistic": st, "mean": b["value"], "se_date": b["se"],
                            "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"], "n_cells": b["n"],
                            "n_dates": b["n_dates"],
                            "n_members": len({r["member"] for r in ws})})
    return out


def run(predictions_dir, lane_config: dict, eval_config: dict, *, output_dir=None,
        overwrite: bool = False, run_label: str = "", **kwargs) -> Path:
    t0 = time.time()
    predictions_dir = Path(predictions_dir).expanduser()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "shape"
    parts = output_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    F, S = _configure(eval_config)

    steps = [int(s) for s in (eval_config.get("steps") or [24, 120])]
    dates = {str(d) for d in (eval_config.get("dates") or [])}
    var_names = str(eval_config.get("vars", "10u,10v")).split(",")
    bands = tuple(str(eval_config.get("bands", "mid,b12")).split(","))
    members = eval_config.get("members")
    n_boot = int(eval_config.get("n_boot", DEFAULT_N_BOOT))
    seed = int(eval_config.get("seed", DEFAULT_SEED))
    windows = dict(DEFAULT_WINDOWS, **(eval_config.get("windows") or {}))
    arm = run_label or eval_config.get("run_label") or predictions_dir.name

    files = [p for p in find_predictions(predictions_dir)
             if p.step in steps and (not dates or p.date in dates)]
    if not files:
        raise FileNotFoundError(f"shape: no prediction files for steps {steps} under {predictions_dir}")
    stdev = {str(k): float(v)
             for k, v in np.load(S.STATS, allow_pickle=True).item()["stdev"].items()}
    LOG.info("shape: %d files, arm %s, git %s", len(files), arm, _git_commit())

    ap = None
    ap_grid = None
    all_rows: list[dict] = []
    per_file = []
    for pf in files:
        path = str(pf.path)
        dirname = Path(os.path.realpath(path)).parent.parent.name
        sel, blat, blon, selnote = F.box_selection(path)
        if ap is None:
            ap = S.Apparatus(blat, blon)
            ap_grid = (blat, blon)
        else:
            assert np.array_equal(ap_grid[0], blat) and np.array_equal(ap_grid[1], blon), \
                f"{path}: box grid differs from the first file's"
        tf = time.time()
        rows: list[dict] = []
        nk = F.run_file(ap, arm, dirname, pf.date, pf.step, path, var_names, sel, stdev, rows,
                        members=members, bands=bands)
        out = parts / f"p1_{arm}_{pf.date}_s{int(pf.step):03d}.csv"
        F.write_csv(str(out), rows)
        dt = time.time() - tf
        per_file.append({"file": path, "resolved": os.path.realpath(path), "dirname": dirname,
                         "date": pf.date, "step": pf.step, "n_members": nk, "seconds": dt,
                         "box_selection": selnote, "part": str(out)})
        # re-read the part so that the pooled numbers are made from exactly what was written
        all_rows += read_rows(out)
        LOG.info("shape: %s done in %.1fs", Path(path).name, dt)

    with open(output_dir / "rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=F.CSV_COLS)
        w.writeheader()
        w.writerows(all_rows)
    summary = summarise(all_rows, windows, n_boot, seed)
    (output_dir / "summary.json").write_text(json.dumps(
        {"arm": arm, "n_boot": n_boot, "seed": seed, "rows": summary}, indent=1) + "\n")
    meta = {"evaluator": "shape", "git_commit": _git_commit(), "arm": arm,
            "predictions_dir": str(predictions_dir), "steps": steps, "vars": var_names,
            "bands": list(bands), "files": per_file, "geom_cache": S.GEOMCACHE,
            "mask_cache": ap.maskcache if ap else None, "probe_file": F.PROBE, "stats_file": S.STATS,
            "n_boot": n_boot, "seed": seed, "seconds": time.time() - t0}
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return output_dir
