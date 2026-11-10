"""Per-date fair CRPS straight from the prediction files (fallback of the paired test).

Used only when ``probabilistic``'s ``scores_by_lead.csv`` lacks the per-date value of a
(variable, domain) for the row or the reference. Same definitions as
``eval/evaluators/probabilistic/core/scoring.py`` so the two agree (tested):

- fair CRPS for M members at one point:
  ``mean_i |x_i - y|  -  (1 / (2 M (M - 1))) * sum_{i,j} |x_i - x_j|``
- a point counts when the truth and every member are finite;
- the domain value is the area-weighted mean over the domain's valid points, the weight
  being ``area_weight`` when the file has it, else cos(latitude) (``_area_weights``);
- domains: global, n.hem (lat >= 20), tropics (-20 < lat < 20), s.hem (lat <= -20),
  europe (35-75 N, 25 W-45 E), as ``_domain_mask``;
- the truth is member 0 of ``y`` (one truth per file, as the evaluator does);
- ``ws`` (alias ``10ff``) is hypot(10u, 10v);
- tp (and any variable in ``clip_vars``) is clipped at 0 in the members and the truth
  before scoring.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np

PRED_RE = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")
DEFAULT_CLIP = ("tp",)


def domain_mask(name: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    key = name.lower()
    lon = ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0
    if key in {"global", "all"}:
        return np.ones(lat.shape, dtype=bool)
    if key in {"n.hem", "nhem", "north_hemisphere"}:
        return lat >= 20.0
    if key == "tropics":
        return (lat > -20.0) & (lat < 20.0)
    if key in {"s.hem", "shem", "south_hemisphere"}:
        return lat <= -20.0
    if key == "europe":
        return (lat >= 35.0) & (lat <= 75.0) & (lon >= -25.0) & (lon <= 45.0)
    raise ValueError(f"unknown domain {name!r}")


def area_weights(lat: np.ndarray, area_weight: np.ndarray | None = None) -> np.ndarray:
    w = np.asarray(area_weight if area_weight is not None else np.cos(np.deg2rad(lat)),
                   dtype=np.float64).reshape(-1)
    s = float(np.sum(w))
    return w / s if s > 0 else np.full(w.shape, 1.0 / w.size)


def fair_crps_points(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Pointwise fair CRPS; members (M, P), truth (P,). NaN where any input is non-finite."""
    f = np.asarray(members, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    m = f.shape[0]
    out = np.full(y.shape, np.nan)
    if m < 2:
        return out
    ok = np.isfinite(y) & np.all(np.isfinite(f), axis=0)
    if not ok.any():
        return out
    fo, yo = f[:, ok], y[ok]
    skill = np.mean(np.abs(fo - yo[None, :]), axis=0)
    # sum_{i,j} |x_i - x_j| = 2 * sum_k (2k - M - 1) x_(k), k = 1..M on the sorted members
    s = np.sort(fo, axis=0)
    k = np.arange(1, m + 1, dtype=np.float64)[:, None]
    pair_sum = 2.0 * np.sum((2.0 * k - m - 1.0) * s, axis=0)
    out[ok] = skill - pair_sum / (2.0 * m * (m - 1))
    return out


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    ok = np.isfinite(values) & np.isfinite(weights)
    if not ok.any():
        return math.nan
    tot = float(weights[ok].sum())
    return float(np.sum(values[ok] * weights[ok]) / tot) if tot > 0 else math.nan


def _decode(v) -> str:
    if isinstance(v, bytes):
        return v.decode()
    return str(v)


def _field(ds, var_name: str, field: str, ws_index: dict[str, int]) -> np.ndarray | None:
    """(member, point) array of one weather state (or ws) of one variable of the file."""
    da = ds[var_name]
    if "sample" in da.dims:
        da = da.isel(sample=0)
    mem_dim = "ensemble_member" if "ensemble_member" in da.dims else None
    pt_dim = [d for d in da.dims if d not in ("weather_state", mem_dim)][0]

    def one(name):
        a = da.isel(weather_state=ws_index[name])
        a = a.transpose(mem_dim, pt_dim) if mem_dim else a.expand_dims("m").transpose("m", pt_dim)
        return np.asarray(a.values, dtype=np.float64)

    if field in ws_index:
        return one(field)
    if field in ("ws", "10ff") and "10u" in ws_index and "10v" in ws_index:
        return np.hypot(one("10u"), one("10v"))
    return None


def per_date_fcrps(pred_dir: str | Path, variables: Iterable[str], domains: Iterable[str], *,
                   clip_vars: Iterable[str] = DEFAULT_CLIP, steps: Iterable[int] | None = None
                   ) -> list[dict]:
    """[{date, step, variable, domain, value}] per prediction file of ``pred_dir``."""
    import xarray as xr

    clip = set(clip_vars)
    step_set = {int(s) for s in steps} if steps else None
    out: list[dict] = []
    for path in sorted(Path(pred_dir).glob("predictions_*.nc")):
        m = PRED_RE.search(path.name)
        if not m:
            continue
        date, step = m.group(1), int(m.group(2))
        if step_set is not None and step not in step_set:
            continue
        with xr.open_dataset(path, cache=False, decode_timedelta=False) as ds:
            ws_index = {_decode(v): i for i, v in enumerate(ds["weather_state"].values.tolist())}
            n = int(ds.sizes["grid_point_hres"])
            lat = np.asarray(ds["lat_hres"].values, dtype=np.float64).reshape(-1)[:n]
            lon = np.asarray(ds["lon_hres"].values, dtype=np.float64).reshape(-1)[:n]
            aw = None
            if "area_weight" in ds.variables and int(ds["area_weight"].size) == n:
                aw = np.asarray(ds["area_weight"].values, dtype=np.float64).reshape(-1)
            w = area_weights(lat, aw)
            masks = {d: domain_mask(d, lat, lon) for d in domains}
            for var in variables:
                pred = _field(ds, "y_pred", var, ws_index)
                truth = _field(ds, "y", var, ws_index)
                if pred is None or truth is None:
                    continue
                truth = truth[0]
                if var in clip:
                    pred = np.maximum(pred, 0.0)
                    truth = np.maximum(truth, 0.0)
                pts = fair_crps_points(pred, truth)
                for d, mask in masks.items():
                    val = weighted_mean(pts, np.where(mask, w, 0.0))
                    out.append({"date": date, "step": step, "variable": var, "domain": d,
                                "value": val})
    return out
