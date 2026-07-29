"""Compute per-instance TC extremes from prediction NetCDFs and reduce them.

Two modes, both emitted into the same per_instance.json so the dashboard reads one contract:

  GRID  -- the pinned input-selected budget (many cases x all members). This is the ladder
           default and the only mode whose reproduction of the full protocol is validated.
  DRAWS -- N independent re-runs of ONE instance (`draw_*` sub-directories, as produced by the
           ladder's seed-draw mode). Model noise is unseeded, so repeated predict = independent
           samples of the conditional PDF. Ten draws rank checkpoints at rho +0.95 for ~10
           forecasts, but they under-deepen by a near-constant ~6.6 hPa: RANK-ONLY, never a value.

Reads one contiguous latitude band per member (memory-safe at o1280/o2560) and masks to the event
box in numpy, then stores the full ``(case, member)`` table so any reduction can be recomputed
without touching the predictions again.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import netCDF4
import xarray as xr

from eval._backends.tc.events import EVENTS
from eval._backends.tc.grid import point_mask
from eval._backends.tc.loading_predictions import (
    discover_prediction_files,
    prediction_point_coordinates,
)

LOG = logging.getLogger(__name__)

# curve -> netCDF variable. `y` is the target (ENFO on the eefo->enfo lanes) and `x_interp` the
# interpolated input (EEFO): both travel inside the same file, so they cannot drift in support.
CURVES = {"model": "y_pred", "enfo": "y", "eefo": "x_interp"}


def _band_extremes(path, var, lo, hi, sub_mask, i_msl, i_u, i_v):
    """Per-member (eye = min MSLP hPa, peak = max |10m wind|) inside the event box."""
    nc = netCDF4.Dataset(path)
    try:
        v = nc.variables[var]
        n_members = v.shape[1]
        eye = np.empty(n_members)
        peak = np.empty(n_members)
        for m in range(n_members):
            msl = np.asarray(v[0, m, lo:hi, i_msl], dtype=np.float64)[sub_mask] / 100.0
            u = np.asarray(v[0, m, lo:hi, i_u], dtype=np.float64)[sub_mask]
            w = np.asarray(v[0, m, lo:hi, i_v], dtype=np.float64)[sub_mask]
            eye[m] = msl.min()
            peak[m] = np.hypot(u, w).max()
    finally:
        nc.close()
    return eye, peak


def _resolve_predictions(root: Path) -> Path:
    found = sorted(root.rglob("predictions_*.nc"))
    if not found:
        raise FileNotFoundError(f"tc_proxy: no prediction files under {root}")
    return found[0].parent


def _table(pdir: Path, events, require_full_grid: bool):
    """Per-(case, member) eye/peak tables for one prediction directory."""
    pred_files = discover_prediction_files(pdir)
    dates = sorted({d for _, d, _ in pred_files})
    steps = sorted({s for _, _, s in pred_files})
    grid = {(d, s): p for p, d, s in pred_files}
    if require_full_grid:
        missing = [(d, s) for d in dates for s in steps if (d, s) not in grid]
        if missing:
            # a ragged budget makes "deepest over the budget" incomparable between rungs
            raise ValueError(f"tc_proxy: incomplete date x step grid, missing {missing}")
    cases = [(d, s) for d in dates for s in steps if (d, s) in grid]

    ds0 = xr.open_dataset(pred_files[0][0], decode_timedelta=False)
    weather_states = ds0["weather_state"].values.tolist()
    for required in ("msl", "10u", "10v"):
        if required not in weather_states:
            raise ValueError(f"tc_proxy: weather state {required!r} absent; have {weather_states}")
    i_msl = weather_states.index("msl")
    i_u = weather_states.index("10u")
    i_v = weather_states.index("10v")
    lon, lat = prediction_point_coordinates(ds0)
    ds0.close()

    out = {"cases": [f"{d}_{s:03d}" for d, s in cases], "grid_points": int(len(lon)),
           "dates": [int(d) for d in dates], "steps": [int(s) for s in steps], "events": {}}
    for event in events:
        if event not in EVENTS:
            LOG.warning("tc_proxy: unknown event %r, skipping", event)
            continue
        mask = point_mask(lon, lat, EVENTS[event].bbox)
        idx = np.where(mask)[0]
        if idx.size == 0:
            LOG.warning("tc_proxy: event %r box has no grid points on this lane", event)
            continue
        lo, hi = int(idx.min()), int(idx.max()) + 1
        sub_mask = mask[lo:hi]
        tables, n_members = {}, None
        for curve, var in CURVES.items():
            eye_rows, peak_rows = [], []
            for d, s in cases:
                eye, peak = _band_extremes(grid[(d, s)], var, lo, hi, sub_mask, i_msl, i_u, i_v)
                if n_members is None:
                    n_members = int(eye.size)
                elif int(eye.size) != n_members:
                    raise ValueError(
                        f"tc_proxy: member count drifts within the budget "
                        f"({eye.size} vs {n_members}) at {d}+{s:03d}h — not comparable")
                eye_rows.append(eye.tolist())
                peak_rows.append(peak.tolist())
            tables[curve] = {"eye": eye_rows, "peak": peak_rows}
        out["events"][event] = {"bbox_points": int(mask.sum()), "n_members": n_members,
                                "tables": tables}
    return out


def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    **kwargs,
) -> Path:
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "tc_proxy"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "per_instance.json"
    if out_path.exists() and not overwrite:
        LOG.info("tc_proxy per-instance table already exists, skipping: %s", out_path)
        return output_dir

    events = eval_config.get("events") or (lane_config.get("tc", {}) or {}).get("events") or []
    if not events:
        LOG.warning("tc_proxy: no TC events configured for this lane "
                    "(set tc_proxy.events or tc.events)")
        out_path.write_text(json.dumps({"events": {}, "draws": []}))
        return output_dir

    report: dict = {"grid": None, "draws": [], "events": {}}

    # --- draw mode: sibling draw_* directories, each one an independent re-run ---------------
    draw_roots = sorted(
        d for d in (list(predictions_dir.glob("draw_*")) + list(predictions_dir.parent.glob("draw_*")))
        if d.is_dir())
    seen: set = set()
    for d in draw_roots:
        if d in seen:
            continue
        seen.add(d)
        try:
            report["draws"].append(_table(_resolve_predictions(d), events, require_full_grid=False))
        except (FileNotFoundError, ValueError) as exc:
            LOG.warning("tc_proxy: draw %s unusable (%s)", d.name, exc)
    if report["draws"]:
        LOG.info("tc_proxy: %d independent draws found", len(report["draws"]))

    # --- grid mode: the pinned budget ---------------------------------------------------------
    try:
        pdir = _resolve_predictions(predictions_dir)
        if not any(pdir == _resolve_predictions(d) for d in seen):
            report["grid"] = _table(pdir, events, require_full_grid=True)
            report["events"] = report["grid"]["events"]
            for event, payload in report["events"].items():
                LOG.info("tc_proxy %s: %d cases x %d members over %d box points", event,
                         len(report["grid"]["cases"]), payload["n_members"],
                         payload["bbox_points"])
    except FileNotFoundError:
        if not report["draws"]:
            raise

    out_path.write_text(json.dumps(report))
    return output_dir
