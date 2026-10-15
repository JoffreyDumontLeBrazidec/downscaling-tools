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
stats_file, windows ({"YYYYMM": label}), and, for a box other than the epic's (boxes.py):

- box: a name (tc_atlantic, the default; north_atlantic 35-55 N, 45-20 W; e2) or
  [lat_min, lat_max, lon_min, lon_max]. Any box is cut out of each file by its own
  coordinates, so Europe-box (E2) files and global files both work;
- cache_dir: where the per-box geometry and mask caches live (default
  <EVAL_ROOT>/_cache/shape);
- regrid_deg: the regular grid of statistics 2 and 3 (default 0.09, the instrument's);
- knn_file, forcings, inter_mat_dir: the global O1280 inputs that build a new box's
  open-ocean mask once (Atos paths by default; copy the mask cache to other clusters).

Other grids (o96->o320). The band-pass kernels (SIGMAS, 4.1-18.4 km), the gradient half-width
(GRAD_H_KM, 8 km), the regrid (0.09 deg) and the correlation windows (4 deg) are O1280 sizes
(9 km spacing). On O320 (32 km spacing) the three narrowest kernels hold the point alone, so
the b12 band (S_0 - S_2) is empty and the mid band becomes a grid-scale high-pass. Two keys put
the instrument on another grid:

- kernel_scale: multiplies SIGMAS and GRAD_H_KM (default 1). kernel_scale 4 keeps the
  instrument in grid units on O320 (O320 spacing / O1280 spacing = 3.96): the b12 band then
  spans about 84-160 km and the mid band about 160-436 km. Pass regrid_deg 0.36 with it;
- window_deg: the side of the correlation windows (default 4); 8 on O320 holds 22 x 22
  regrid pixels, as 4 deg holds 44 x 44 on O1280 at 0.09.

The geometry cache name carries the kernel scale, so two scales never share a cache.

Off the epic's box, all dates pool into one window "all" unless ``windows`` is given, and
files whose members share one input (matched-seed draws) are read member by member,
since the full-rung reader asserts that inputs differ.

From the command line: ``eval.cli evaluate --only shape --opt shape.box=north_atlantic``.
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


_MODULE_DEFAULTS: dict = {}


def kernel_scale(eval_config: dict) -> float:
    value = float(eval_config.get("kernel_scale") or 1.0)
    if value <= 0:
        raise ValueError(f"shape.kernel_scale must be positive, got {value}")
    return value


def band_km(S) -> dict[str, list[float]]:
    """Approximate wavelength span of each band (the epic's 21-40 / 40-109 km at scale 1)."""
    k = float(S.SIGMAS[0]) / 4.12
    return {"b12": [round(21.0 * k, 1), round(40.0 * k, 1)], "mid": [round(40.0 * k, 1), round(109.0 * k, 1)]}


def _configure(eval_config: dict):
    """Set the instrument's module constants for this run, starting from their defaults
    each time, so that two runs in one process (a model and its reference) never inherit
    each other's box or caches."""
    from . import fullrung as F
    from . import instrument as S

    if not _MODULE_DEFAULTS:
        _MODULE_DEFAULTS.update(
            {("S", k): getattr(S, k) for k in ("GEOMCACHE", "DONOR_MASKCACHE", "LOCAL_MASKCACHE",
                                               "STATS", "KNN", "FORC", "IM", "REGRID_DEG",
                                               "SIGMAS", "GRAD_H_KM", "WINDOW_DEG",
                                               "build_gradient_operator")})
        _MODULE_DEFAULTS[("F", "PROBE")] = F.PROBE
    for (mod, k), v in _MODULE_DEFAULTS.items():
        setattr(S if mod == "S" else F, k, v)

    if eval_config.get("regrid_deg"):
        S.REGRID_DEG = float(eval_config["regrid_deg"])
    scale = kernel_scale(eval_config)
    if scale != 1.0:
        S.SIGMAS = [float(v) * scale for v in _MODULE_DEFAULTS[("S", "SIGMAS")]]
        S.GRAD_H_KM = float(_MODULE_DEFAULTS[("S", "GRAD_H_KM")]) * scale
        # the verbatim instrument binds h_km=GRAD_H_KM as a default argument at import, so the
        # scaled half-width is passed explicitly (the geometry cache records it as grad_h)
        original = _MODULE_DEFAULTS[("S", "build_gradient_operator")]
        S.build_gradient_operator = (
            lambda lat, lon, idx, h_km=None, kk=S.GRAD_K, _f=original:
            _f(lat, lon, idx, h_km=S.GRAD_H_KM if h_km is None else h_km, kk=kk))
    if eval_config.get("window_deg"):
        S.WINDOW_DEG = float(eval_config["window_deg"])
    for key, attr in (("knn_file", "KNN"), ("forcings", "FORC"), ("inter_mat_dir", "IM")):
        if eval_config.get(key):
            setattr(S, attr, str(eval_config[key]))
    if eval_config.get("geom_cache"):
        S.GEOMCACHE = str(eval_config["geom_cache"])
    if eval_config.get("mask_cache"):
        S.DONOR_MASKCACHE = str(eval_config["mask_cache"])
        S.LOCAL_MASKCACHE = str(eval_config["mask_cache"])
    if eval_config.get("stats_file"):
        S.STATS = str(eval_config["stats_file"])
    if eval_config.get("probe_file"):
        F.PROBE = str(eval_config["probe_file"])
    return F, S


def members_share_input(path, var_names, sel, members=None) -> list[int]:
    """The member indices when every member of the file has the same ``x_interp`` on the
    box (matched-seed draws), else [] (full-rung file: one input per member)."""
    import netCDF4 as nc

    with nc.Dataset(str(path)) as d:
        nm = d.dimensions["ensemble_member"].size
        ks = list(range(nm)) if members is None else [int(k) for k in members]
        if len(ks) < 2:
            return []
        j = [str(x) for x in d["weather_state"][:]].index(var_names[0])
        first = np.asarray(d["x_interp"][0, ks[0], :, j])[sel]
        for k in ks[1:]:
            if not np.array_equal(np.asarray(d["x_interp"][0, k, :, j])[sel], first):
                return []
    return ks


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
    from . import boxes as B
    box = B.resolve_box(eval_config.get("box"))
    F, S = _configure(eval_config)
    use_probe = B.is_legacy(box) and os.path.exists(F.PROBE)

    steps = [int(s) for s in (eval_config.get("steps") or [24, 120])]
    dates = {str(d) for d in (eval_config.get("dates") or [])}
    var_names = str(eval_config.get("vars", "10u,10v")).split(",")
    bands = tuple(str(eval_config.get("bands", "mid,b12")).split(","))
    members = eval_config.get("members")
    n_boot = int(eval_config.get("n_boot", DEFAULT_N_BOOT))
    seed = int(eval_config.get("seed", DEFAULT_SEED))
    if B.is_legacy(box) or eval_config.get("windows"):
        windows = dict(DEFAULT_WINDOWS, **(eval_config.get("windows") or {}))
    else:
        windows = None   # one window "all", filled from the files' dates below
    arm = run_label or eval_config.get("run_label") or predictions_dir.name

    files = [p for p in find_predictions(predictions_dir)
             if p.step in steps and (not dates or p.date in dates)]
    if not files:
        raise FileNotFoundError(f"shape: no prediction files for steps {steps} under {predictions_dir}")
    if windows is None:
        windows = {str(p.date)[:6]: "all" for p in files}
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
        if use_probe:
            sel, blat, blon, selnote = F.box_selection(path)
        else:
            sel, blat, blon, selnote = B.select_box(path, box)
        if ap is None:
            if not use_probe:
                # one cache per box and grid; explicit geom_cache / mask_cache still win
                cp = B.cache_paths(box, eval_config.get("cache_dir"), n_points=blat.size)
                if not eval_config.get("geom_cache"):
                    S.GEOMCACHE = cp["geom_cache"]
                    scale = kernel_scale(eval_config)
                    if scale != 1.0:   # the gradient operator depends on GRAD_H_KM
                        S.GEOMCACHE = S.GEOMCACHE.replace("_geometry.npz", f"_k{scale:g}_geometry.npz")
                if not eval_config.get("mask_cache"):
                    S.DONOR_MASKCACHE = S.LOCAL_MASKCACHE = cp["mask_cache"]
            ap = S.Apparatus(blat, blon)
            ap_grid = (blat, blon)
            B.check_masks(ap, box)
        else:
            assert np.array_equal(ap_grid[0], blat) and np.array_equal(ap_grid[1], blon), \
                f"{path}: box grid differs from the first file's"
        tf = time.time()
        rows: list[dict] = []
        shared = members_share_input(path, var_names, sel, members)
        if shared:
            # matched-seed draws: one input for every member, which read_fullrung refuses
            # when given several members at once; one member per call reads the same rows
            nk = sum(F.run_file(ap, arm, dirname, pf.date, pf.step, path, var_names, sel, stdev,
                                rows, members=[k], bands=bands) for k in shared)
        else:
            nk = F.run_file(ap, arm, dirname, pf.date, pf.step, path, var_names, sel, stdev, rows,
                            members=members, bands=bands)
        out = parts / f"p1_{arm}_{pf.date}_s{int(pf.step):03d}.csv"
        F.write_csv(str(out), rows)
        dt = time.time() - tf
        per_file.append({"file": path, "resolved": os.path.realpath(path), "dirname": dirname,
                         "date": pf.date, "step": pf.step, "n_members": nk, "seconds": dt,
                         "box_selection": selnote, "members_share_input": bool(shared),
                         "part": str(out)})
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
            "bands": list(bands), "box": box.tag, "box_bounds": box.as_list(),
            "box_points": int(ap.n) if ap else None, "regrid_deg": S.REGRID_DEG,
            "kernel_scale": kernel_scale(eval_config), "sigmas_km": list(S.SIGMAS),
            "grad_h_km": S.GRAD_H_KM, "window_deg": S.WINDOW_DEG, "band_km": band_km(S),
            "windows": windows, "files": per_file, "geom_cache": S.GEOMCACHE,
            "mask_cache": ap.maskcache if ap else None, "probe_file": F.PROBE, "stats_file": S.STATS,
            "n_boot": n_boot, "seed": seed, "seconds": time.time() - t0}
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return output_dir
