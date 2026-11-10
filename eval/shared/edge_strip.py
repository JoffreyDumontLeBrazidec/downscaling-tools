"""The box-edge strip of a regional prediction, and the scan for the slot-head edge artefact.

A slot-head model on a regional box (the lossless class of anemoi-core with ``write: slot_head``) writes the
points of the half cells at the box edge from tokens it never sees inside the box. On XCR100 (exact_only read)
this gives 800-840 mm of tp in the first 0.25 degrees on 30/30 dates, 2t error 8x the inner box (docs ablation
note s21, 6 Nov 2026). Every score of a regional box therefore leaves out ``DEFAULT_BOX_EDGE_DEG`` (1 degree)
at the box edge; a global point set has no edge and keeps every point.

``scan_edge_artefact`` reads a few prediction files and says whether the artefact is present, so that
``eval.cli`` refuses the evaluators that do not leave the strip out.

How evaluators apply the strip (one mask, ``box_edge_mask``):

* ``resolve_box_edge_deg(eval_config)`` gives the strip width: the evaluator block's ``box_edge_deg`` when set
  (``box_edge_deg: 0`` is the explicit per-evaluator opt-out), else the width ``eval.cli`` set for the evaluator
  being run (``configured``), else ``DEFAULT_BOX_EDGE_DEG``.
* ``box_edge_index(lat, lon, edge_deg)`` gives the indices of the points to keep, or ``slice(None)`` when every
  point is kept (a global grid, or width 0), so ``array[..., idx, ...]`` is free on global grids.
* ``strip_box_edge(ds, edge_deg)`` returns an xarray dataset without the strip points (every variable on the
  ``lat_hres`` dimension is subset; the input grid ``x`` is not). A regional box minus its strip is a smaller
  regional box, so everything downstream (area weights, neighbours, box selections) works unchanged.

A global point set (``is_global``) has no edge and is never touched.
"""
from __future__ import annotations

import contextlib
import contextvars
import os
from pathlib import Path

import numpy as np

DEFAULT_BOX_EDGE_DEG = 1.0
EDGE_DEG_ENV = "EVAL_BOX_EDGE_DEG"   # the active width, inherited by evaluator subprocesses
# scan: a point within SCAN_EDGE_DEG of the edge whose tp (mm, as stored x 1000) exceeds HOT_MM in some member
SCAN_EDGE_DEG = 0.25
HOT_MM = 300.0
# artefact = such a point on at least this share of the scanned files, while the inner box stays below HOT_MM there
PERSIST_SHARE = 0.5


def wrap_lon(lon) -> np.ndarray:
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def is_global(lat, lon) -> bool:
    """True when the point set covers the sphere (no box edges)."""
    return bool(np.ptp(np.asarray(lat, dtype=np.float64)) > 170.0 and np.ptp(wrap_lon(lon)) > 350.0)


def edge_distance_deg(lat, lon) -> np.ndarray:
    """Distance in degrees (latitude or longitude, the smaller) from each point to the edge of its box.

    The longitude span is measured on the -180..180 wrap unless the box crosses the dateline, when the
    0..360 wrap is tighter. A global set gets +inf everywhere."""
    lat = np.asarray(lat, dtype=np.float64)
    if is_global(lat, lon):
        return np.full(lat.shape, np.inf)
    lon180 = wrap_lon(lon)
    lon360 = lon180 % 360.0
    lo = lon180 if np.ptp(lon180) <= np.ptp(lon360) else lon360
    return np.minimum.reduce([lat - lat.min(), lat.max() - lat, lo - lo.min(), lo.max() - lo])


def box_edge_mask(lat, lon, edge_deg: float = DEFAULT_BOX_EDGE_DEG) -> np.ndarray:
    """Points at least ``edge_deg`` degrees inside a regional box (True = keep). Global sets keep every point."""
    lat = np.asarray(lat, dtype=np.float64)
    if edge_deg <= 0.0:
        return np.ones(lat.shape, dtype=bool)
    return edge_distance_deg(lat, lon) >= float(edge_deg)


_ACTIVE_EDGE_DEG: contextvars.ContextVar[float | None] = contextvars.ContextVar("box_edge_deg", default=None)


def resolve_box_edge_deg(eval_config: dict | None = None) -> float:
    """The strip width for an evaluator: its ``box_edge_deg`` knob, else the active one, else 1 degree."""
    if isinstance(eval_config, dict) and eval_config.get("box_edge_deg") is not None:
        return float(eval_config["box_edge_deg"])
    active = _ACTIVE_EDGE_DEG.get()
    if active is None and os.environ.get(EDGE_DEG_ENV, "").strip():
        active = float(os.environ[EDGE_DEG_ENV])   # set by configured() in a parent process
    return DEFAULT_BOX_EDGE_DEG if active is None else float(active)


@contextlib.contextmanager
def configured(eval_config: dict | None = None):
    """Make ``resolve_box_edge_deg(None)`` return this evaluator's width while its code runs (``eval.cli``).

    The width is also exported as ``EVAL_BOX_EDGE_DEG`` for the evaluators that run their work in a subprocess."""
    width = resolve_box_edge_deg(eval_config)
    token = _ACTIVE_EDGE_DEG.set(width)
    previous = os.environ.get(EDGE_DEG_ENV)
    os.environ[EDGE_DEG_ENV] = repr(width)
    try:
        yield
    finally:
        _ACTIVE_EDGE_DEG.reset(token)
        if previous is None:
            os.environ.pop(EDGE_DEG_ENV, None)
        else:
            os.environ[EDGE_DEG_ENV] = previous


def subprocess_env(eval_config: dict | None = None, base: dict | None = None) -> dict:
    """An environment for an evaluator subprocess that carries this evaluator's strip width."""
    env = dict(os.environ if base is None else base)
    env[EDGE_DEG_ENV] = repr(resolve_box_edge_deg(eval_config))
    return env


def box_edge_index(lat, lon, edge_deg: float | None = None):
    """Indices of the points outside the box-edge strip, or ``slice(None)`` when every point is kept."""
    edge_deg = resolve_box_edge_deg() if edge_deg is None else float(edge_deg)
    keep = box_edge_mask(lat, lon, edge_deg)
    if keep.all():
        return slice(None)
    if not keep.any():
        raise ValueError(f"the {edge_deg:g}-degree box-edge strip covers every one of the {keep.size} points")
    return np.flatnonzero(keep)


def strip_box_edge(ds, edge_deg: float | None = None, *, lat_var: str = "lat_hres", lon_var: str = "lon_hres"):
    """An xarray dataset without the points of the box-edge strip (unchanged on a global grid or without lat/lon)."""
    if lat_var not in ds.variables or lon_var not in ds.variables:
        return ds
    lat = np.asarray(ds[lat_var].values, dtype=np.float64).reshape(-1)
    lon = np.asarray(ds[lon_var].values, dtype=np.float64).reshape(-1)
    idx = box_edge_index(lat, lon, edge_deg)
    if isinstance(idx, slice):
        return ds
    dim = ds[lat_var].dims[-1]
    out = ds.isel({dim: idx})
    out.attrs = {**ds.attrs, "box_edge_strip_removed_deg": float(resolve_box_edge_deg() if edge_deg is None
                                                                  else edge_deg)}
    return out


def inner_bounds(lat, lon, edge_deg: float | None = None):
    """The lat/lon rectangle outside the box-edge strip of a regional point set, or None (global / width 0).

    Returns ``(lat_lo, lat_hi, lon_lo, lon_hi, wrap)`` with ``wrap`` 180 or 360, the longitude convention in
    which the box is contiguous. Use ``inside_bounds`` to test other points (e.g. a regridded map) against it."""
    edge_deg = resolve_box_edge_deg() if edge_deg is None else float(edge_deg)
    lat = np.asarray(lat, dtype=np.float64).reshape(-1)
    if edge_deg <= 0.0 or is_global(lat, lon):
        return None
    lon180 = wrap_lon(np.asarray(lon, dtype=np.float64).reshape(-1))
    lon360 = lon180 % 360.0
    wrap, lo = (180, lon180) if np.ptp(lon180) <= np.ptp(lon360) else (360, lon360)
    return (float(lat.min()) + edge_deg, float(lat.max()) - edge_deg,
            float(lo.min()) + edge_deg, float(lo.max()) - edge_deg, wrap)


def inside_bounds(bounds, lat, lon) -> np.ndarray:
    """True for the points inside ``inner_bounds`` (every point when ``bounds`` is None)."""
    lat = np.asarray(lat, dtype=np.float64)
    if bounds is None:
        return np.ones(lat.shape, dtype=bool)
    lat_lo, lat_hi, lon_lo, lon_hi, wrap = bounds
    lo = wrap_lon(lon)
    if wrap == 360:
        lo = lo % 360.0
    return (lat >= lat_lo) & (lat <= lat_hi) & (lo >= lon_lo) & (lo <= lon_hi)


def _tp_members(ds) -> np.ndarray | None:
    names = [str(s) for s in ds["weather_state"][:]] if "weather_state" in ds.variables else []
    if "tp" not in names or "y_pred" not in ds.variables:
        return None
    y = np.asarray(ds["y_pred"][:], dtype=np.float64)
    y = y.reshape(-1, y.shape[-2], y.shape[-1])  # (members, points, vars)
    return np.clip(y[:, :, names.index("tp")], 0.0, None) * 1000.0


def scan_edge_artefact(predictions_dir: Path, *, max_files: int = 6) -> dict:
    """Look for the slot-head edge artefact in up to ``max_files`` prediction files of a directory.

    Returns ``{"checked": n_files, "regional": bool, "artefact": bool, "hot_share": float, "edge_max_mm": [...],
    "inner_max_mm": [...], "detail": str}``. Files without tp or with a global grid are not counted."""
    import netCDF4

    files = sorted(Path(predictions_dir).glob("predictions_*.nc"))
    if len(files) > max_files:
        files = [files[i] for i in np.linspace(0, len(files) - 1, max_files).round().astype(int)]
    out = {"checked": 0, "regional": False, "artefact": False, "hot_share": 0.0,
           "edge_max_mm": [], "inner_max_mm": [], "detail": ""}
    hot = 0
    for f in files:
        try:
            with netCDF4.Dataset(f) as ds:
                ds.set_auto_mask(False)
                if "lat_hres" not in ds.variables:
                    continue
                lat = np.asarray(ds["lat_hres"][:], dtype=np.float64)
                lon = np.asarray(ds["lon_hres"][:], dtype=np.float64)
                tp = _tp_members(ds)
        except (OSError, KeyError, IndexError):
            continue
        if tp is None or is_global(lat, lon):
            continue
        out["regional"] = True
        de = edge_distance_deg(lat, lon)
        e_max = float(tp[:, de < SCAN_EDGE_DEG].max()) if (de < SCAN_EDGE_DEG).any() else 0.0
        i_max = float(tp[:, de >= DEFAULT_BOX_EDGE_DEG].max()) if (de >= DEFAULT_BOX_EDGE_DEG).any() else 0.0
        out["edge_max_mm"].append(round(e_max, 1))
        out["inner_max_mm"].append(round(i_max, 1))
        out["checked"] += 1
        hot += int(e_max > HOT_MM)
    if out["checked"]:
        out["hot_share"] = hot / out["checked"]
        out["artefact"] = out["hot_share"] >= PERSIST_SHARE
        out["detail"] = (f"tp above {HOT_MM:g} mm within {SCAN_EDGE_DEG:g} deg of the box edge on {hot}/{out['checked']} "
                         f"scanned files (edge max {out['edge_max_mm']} mm, inner max {out['inner_max_mm']} mm)")
    return out
