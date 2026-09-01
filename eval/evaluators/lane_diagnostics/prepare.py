"""Turn the stored measurements into exactly the arrays each figure asks for.

Nothing here opens a high-resolution field.  Every input is either a small JSON
summary written by ``compute`` or one of the JSON artefacts produced by an
earlier measuring session.  Keeping this separate from the figure code means the
figures stay readable and the reductions stay testable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import figures as F

PRECIP_STATS = ["max_mm", "p999_mm", "p99_mm", "wet_frac", "mean_mm"]


# ---------------------------------------------------------------------------
# precipitation, from the precip_scores per-slice per-member table
# ---------------------------------------------------------------------------

def load_precip_rows(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def precip_peaks(scores: dict) -> dict:
    """Per-slice, per-member largest six-hour precipitation for the three fields."""
    model, driver, truth = [], [], []
    for r in scores["rows"]:
        truth.append(r["truth"]["max_mm"])
        for m in r["members"]:
            model.append(m["model"]["max_mm"])
            if "baseline" in m:
                driver.append(m["baseline"]["max_mm"])
    return {"model": np.asarray(model), "driver": np.asarray(driver),
            "truth": np.asarray(truth)}


def precip_ranks(scores: dict, side: str = "model") -> dict:
    """Where the target falls among the ten members, for several statistics.

    Rank 1 means the target is below every member and rank 11 means it is above
    every one of them.  A calibrated ensemble puts the target anywhere with equal
    probability, so the histogram would be flat.
    """
    out: dict = {}
    for stat in PRECIP_STATS:
        ranks, n_mem = [], 0
        for r in scores["rows"]:
            t = r["truth"].get(stat)
            mem = [m.get(side, {}).get(stat) for m in r["members"]]
            mem = [v for v in mem if v is not None]
            if t is None or not mem:
                continue
            n_mem = len(mem)
            ranks.append(int(np.sum(np.asarray(mem) < t)) + 1)
        ranks = np.asarray(ranks)
        hist = np.bincount(ranks, minlength=n_mem + 2)[1:]
        out[stat] = {"histogram": [int(v) for v in hist],
                     "mean_rank": float(ranks.mean()),
                     "n": int(ranks.size), "n_members": n_mem}
    return out


def precip_spread(scores: dict) -> dict:
    """Per slice: the ten model peaks and the ten driver peaks, and the target's."""
    out = {k: [] for k in ("truth", "model_min", "model_max", "model_med", "model_sd",
                           "driver_min", "driver_max", "driver_med", "driver_sd")}
    for r in scores["rows"]:
        mm = np.asarray([m["model"]["max_mm"] for m in r["members"]], dtype=float)
        bl = np.asarray([m["baseline"]["max_mm"] for m in r["members"]
                         if "baseline" in m], dtype=float)
        out["truth"].append(r["truth"]["max_mm"])
        for key, v in (("model", mm), ("driver", bl)):
            out[f"{key}_min"].append(float(v.min()))
            out[f"{key}_max"].append(float(v.max()))
            out[f"{key}_med"].append(float(np.median(v)))
            out[f"{key}_sd"].append(float(v.std()))
    return out


def precip_per_step(scores: dict) -> dict:
    ps = scores["per_step"]
    steps = sorted(int(s) for s in ps["model_rmse_mm"])
    def series(key):
        return [float(ps[key][str(s)]) for s in steps]
    return {
        "steps": steps,
        "model_rmse": series("model_rmse_mm"),
        "baseline_rmse": series("baseline_rmse_mm"),
        "model_corr": series("model_corr"),
        "baseline_corr": series("baseline_corr"),
        "n_slices": int(scores["meta"]["n_slices"]),
    }


def precip_campaign_quantiles(scores: dict) -> dict:
    """Median over slices of each distribution statistic, for the three fields."""
    keys = [("mean_mm", "grid mean"), ("p99_mm", "99th percentile"),
            ("p999_mm", "99.9th percentile"), ("max_mm", "per-slice peak")]
    out = {"labels": [lbl for _, lbl in keys], "model": [], "driver": [], "truth": []}
    n_member_slices = 0
    for stat, _ in keys:
        mv, bv, tv = [], [], []
        for r in scores["rows"]:
            tv.append(r["truth"][stat])
            for m in r["members"]:
                mv.append(m["model"][stat])
                if "baseline" in m:
                    bv.append(m["baseline"][stat])
        n_member_slices = len(mv)
        out["model"].append(float(np.median(mv)))
        out["driver"].append(float(np.median(bv)))
        out["truth"].append(float(np.median(tv)))
    out["n_member_slices"] = n_member_slices
    out["n_slices"] = len(scores["rows"])
    return out


def precip_delivered_increment(scores: dict, stat: str) -> float:
    """Mean over member-slices of the model's own increment over its driver.

    A mean, not a median, because the systematic increment it is compared against
    is itself a mean over paired times, and the two have to be the same kind of
    average. It matters here: the driver's peak is strongly skewed, so its mean
    and its median are far apart.
    """
    diffs = []
    for r in scores["rows"]:
        for m in r["members"]:
            if "baseline" in m:
                diffs.append(m["model"][stat] - m["baseline"][stat])
    return float(np.mean(diffs))


# ---------------------------------------------------------------------------
# the paired training-dataset scan
# ---------------------------------------------------------------------------

def load_scan(path: str | Path) -> list[dict]:
    """Read the paired tropical-belt scan, dropping repeated sample indices."""
    seen, rows = set(), []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("i") in seen:
            continue
        seen.add(rec.get("i"))
        rows.append(rec)
    return rows


def scan_bins(rows: list[dict], thresholds=(1000.0, 990.0, 980.0, 970.0, 960.0)) -> list[dict]:
    out = []
    for thr in thresholds:
        sel = [r for r in rows if r["lo"]["msl_min_hpa"] < thr]
        if not sel:
            continue
        dp = np.asarray([r["hi"]["msl_min_hpa"] - r["lo"]["msl_min_hpa"] for r in sel])
        wg = np.asarray([r["hi"]["wind_max_ms"] - r["lo"]["wind_max_ms"] for r in sel])
        dlo, dhi = F._bootstrap_ci(dp)
        wlo, whi = F._bootstrap_ci(wg)
        out.append({"label": f"< {thr:.0f} hPa", "n": len(sel),
                    "deepening": float(dp.mean()), "deepening_lo": dlo, "deepening_hi": dhi,
                    "wind_gain": float(wg.mean()), "wind_gain_lo": wlo, "wind_gain_hi": whi})
    return out


def scan_increment(rows: list[dict], key: str, msl_below: float | None = None) -> float:
    sel = rows if msl_below is None else [r for r in rows if r["lo"]["msl_min_hpa"] < msl_below]
    return float(np.mean([r["hi"][key] - r["lo"][key] for r in sel]))


def scan_n(rows: list[dict], msl_below: float | None = None) -> int:
    if msl_below is None:
        return len(rows)
    return len([r for r in rows if r["lo"]["msl_min_hpa"] < msl_below])


# ---------------------------------------------------------------------------
# what the model delivers on the cyclone quantities
# ---------------------------------------------------------------------------

def cyclone_delivered(cap: dict, driver_below_hpa: float = 990.0) -> tuple[float, int]:
    """Mean deepening the model adds, on the same selection as the scan bin.

    Restricted to cases where the model and the target are describing the same
    low and to the trained lead range, and selected on the DRIVER's own box
    minimum so the selection does not peek at the target.
    """
    mask = F.colocated_short_lead(cap) & (cap["interp"] < driver_below_hpa)
    return float(-cap["closed"][mask].mean()), int(mask.sum())


def wind_delivered(wind_rows: list[dict], cap: dict, driver_below_hpa: float = 990.0):
    """Mean wind the model adds over its driver, on the cyclone selection.

    The wind table and the pressure table share the (date, step, member) key, so
    the pressure-based selection can be carried over verbatim.
    """
    key_to_row = {(r["date"], int(r["step"]), int(r["member"])): r for r in wind_rows}
    mask = F.colocated_short_lead(cap) & (cap["interp"] < driver_below_hpa)
    gains = []
    for k in np.flatnonzero(mask):
        key = (str(cap["date"][k]), int(cap["step"][k]), int(cap["member"][k]))
        r = key_to_row.get(key)
        if r is not None:
            gains.append(r["model"] - r["interp"])
    return float(np.mean(gains)), len(gains)
