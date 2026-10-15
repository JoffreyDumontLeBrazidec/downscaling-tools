"""Exact rain-tail statistics over every member (``precip_dist`` with ``tail_stats: true``).

The histogram PDF of ``tp_histogram_comparison`` reads one member and bins the values, so its
quantiles are bin-limited. A rain read needs the tail itself, pooled over all members: for
each series (``pred`` = y_pred, ``truth`` = y, ``input`` = x_interp) and each scope (``all``
pooled over every file, or one date ``YYYYMMDD``) and lead step, ``tail_stats.csv`` holds

    n             number of finite values (grid points x members)
    wet_frac      share of values above ``wet_threshold_mm`` (default 0.1 mm)
    mean_mm       mean
    p99, p99_9, p99_99   quantiles of all values (numpy linear interpolation)
    max_mm        maximum
    neg_share     share of raw values below zero (raw tp is not clipped)
    min_mm        minimum (most negative raw value)

Values are in mm (the files hold metres: x 1000). Points are not area weighted (the
octahedral grids are close to equal area). A series whose values are all NaN (the Idalia
bundles carry no tp truth) is skipped, as is ``input`` when x_interp is absent.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from eval.discovery.predictions import find_predictions

LOG = logging.getLogger(__name__)

MM = 1000.0
SERIES = (("pred", "y_pred"), ("truth", "y"), ("input", "x_interp"))
QUANTILES = (("p99", 99.0), ("p99_9", 99.9), ("p99_99", 99.99))
COLS = ["scope", "step", "series", "variable", "n", "wet_frac", "mean_mm",
        "p99", "p99_9", "p99_99", "max_mm", "neg_share", "min_mm", "n_members"]


def _decode(var) -> list[str]:
    out = []
    for v in np.asarray(var[:]).reshape(-1):
        out.append(v.decode() if isinstance(v, bytes) else str(v))
    return out


def _read_all_members(ds, name: str, j: int) -> np.ndarray | None:
    """Every member of one weather state, flattened, as float32 (None if the variable is absent)."""
    if name not in ds.variables:
        return None
    v = ds[name]
    key = tuple(0 if d == "sample" else (j if d == "weather_state" else slice(None))
                for d in v.dimensions)
    return np.asarray(v[key], dtype=np.float32).reshape(-1)


def stats_of(values_mm: np.ndarray, wet_threshold_mm: float) -> dict | None:
    v = values_mm[np.isfinite(values_mm)]
    if v.size == 0:
        return None
    q = np.percentile(v, [p for _, p in QUANTILES])
    out = {"n": int(v.size), "wet_frac": float(np.mean(v > wet_threshold_mm)),
           "mean_mm": float(np.mean(v, dtype=np.float64)), "max_mm": float(v.max()),
           "neg_share": float(np.mean(v < 0.0)), "min_mm": float(v.min())}
    out.update({k: float(x) for (k, _), x in zip(QUANTILES, q)})
    return out


def compute_tail_stats(predictions_dir, output_dir, *, variable: str = "tp",
                       wet_threshold_mm: float = 0.1, steps=None, dates=None,
                       per_date: bool = True) -> Path:
    """Write ``tail_stats.csv`` (and return its path)."""
    import netCDF4

    files = [p for p in find_predictions(Path(predictions_dir))
             if (not steps or p.step in {int(s) for s in steps})
             and (not dates or p.date in {str(d) for d in dates})]
    if not files:
        raise FileNotFoundError(f"tail_stats: no prediction files in {predictions_dir}")
    pooled: dict[tuple[int, str], list[np.ndarray]] = {}
    rows: list[dict] = []
    n_members = 0
    for pf in files:
        with netCDF4.Dataset(str(pf.path)) as ds:
            ds.set_auto_mask(False)
            states = _decode(ds["weather_state"])
            if variable not in states:
                LOG.warning("tail_stats: %s has no %s; skipped", pf.path.name, variable)
                continue
            j = states.index(variable)
            n_members = len(ds.dimensions["ensemble_member"]) if "ensemble_member" in ds.dimensions else 1
            for series, name in SERIES:
                vals = _read_all_members(ds, name, j)
                if vals is None:
                    continue
                vals = vals * np.float32(MM)
                if per_date:
                    st = stats_of(vals, wet_threshold_mm)
                    if st is not None:
                        rows.append({"scope": pf.date, "step": int(pf.step), "series": series,
                                     "variable": variable, "n_members": n_members, **st})
                pooled.setdefault((int(pf.step), series), []).append(vals)
    for (step, series), parts in sorted(pooled.items()):
        st = stats_of(np.concatenate(parts), wet_threshold_mm)
        if st is not None:
            rows.insert(0, {"scope": "all", "step": step, "series": series,
                            "variable": variable, "n_members": n_members, **st})
    rows.sort(key=lambda r: (r["scope"] != "all", r["scope"], r["step"], r["series"]))
    out = Path(output_dir) / "tail_stats.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    LOG.info("tail_stats: wrote %s (%d rows)", out, len(rows))
    return out
