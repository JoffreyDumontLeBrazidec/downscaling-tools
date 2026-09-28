"""Per-leadtime compute kernel.

Computes, for each leadtime (step) found in a predictions directory:

  * Per-variable, per-region surface nMSE (full-field and residual).
  * Skill score vs the interpolated-input baseline.
  * Power spectra (full-field and residual) via healpy, if available.

All heavy helpers are imported directly from _surface_compute to stay
mathematically consistent with the existing surface evaluator.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from eval._backends.scoreboard._surface_compute import (
    SURFACE_VARIABLES,
    _area_weights,
    _to_member_point_weather,
    _weather_state_index,
)


def _to_member_point_weather_lres(da: xr.DataArray) -> xr.DataArray:
    """Like _to_member_point_weather but for the lres (x) variable.

    x lives on grid_point_lres, not grid_point_hres, so we can't reuse the
    hres helper which peeks at ds["lon_hres"] to infer spatial dims.
    Returns shape (member, grid_point_lres, weather_state).
    """
    if "sample" in da.dims and da.sizes["sample"] == 1:
        da = da.isel(sample=0, drop=True)
    # Normalise the member dimension name
    non_spatial = {"weather_state", "grid_point_lres"}
    member_dims = [d for d in da.dims if d not in non_spatial]
    if not member_dims:
        da = da.expand_dims({"member": [0]})
        member_dim = "member"
    elif len(member_dims) == 1:
        member_dim = member_dims[0]
    else:
        da = da.stack(member=member_dims)
        member_dim = "member"
    if member_dim != "member":
        da = da.rename({member_dim: "member"})
    return da.transpose("member", "grid_point_lres", "weather_state")
from eval.discovery.predictions import find_predictions

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Region definitions [lat_min, lat_max, lon_min, lon_max]
# ---------------------------------------------------------------------------
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "global":   (-90.0,  90.0, -180.0, 180.0),
    "tropics":  (-20.0,  20.0, -180.0, 180.0),
    "nh_extra": ( 20.0,  90.0, -180.0, 180.0),
    "sh_extra": (-90.0, -20.0, -180.0, 180.0),
}

SPECTRA_VARS = ["10u", "10v", "2t", "msl", "t_850", "z_500"]


# ---------------------------------------------------------------------------
# Region mask helpers
# ---------------------------------------------------------------------------
def _lon_pm180(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def _region_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    box: tuple[float, float, float, float],
) -> np.ndarray:
    lat_min, lat_max, lon_min, lon_max = box
    lon_n = _lon_pm180(lon)
    lat_ok = (lat >= lat_min) & (lat <= lat_max)
    if abs(lon_max - lon_min) >= 359.9:
        return lat_ok
    lo = _lon_pm180(np.array([lon_min]))[0]
    hi = _lon_pm180(np.array([lon_max]))[0]
    lon_ok = (lon_n >= lo) & (lon_n <= hi) if lo <= hi else (lon_n >= lo) | (lon_n <= hi)
    return lat_ok & lon_ok


def _build_region_masks(
    lat: np.ndarray,
    lon: np.ndarray,
) -> dict[str, np.ndarray]:
    return {name: _region_mask(lat, lon, box) for name, box in REGIONS.items()}


# ---------------------------------------------------------------------------
# Low-res → high-res nearest-neighbour mapping
# ---------------------------------------------------------------------------
def _build_nn_map(lat_lres: np.ndarray, lon_lres: np.ndarray,
                  lat_hres: np.ndarray, lon_hres: np.ndarray) -> np.ndarray:
    """Return index array mapping each hres point to its nearest lres point."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy is required for lres→hres remapping") from exc
    lon_l = _lon_pm180(lon_lres)
    lon_h = _lon_pm180(lon_hres)
    pts_lres = np.stack([np.deg2rad(lat_lres), np.deg2rad(lon_l)], axis=1)
    pts_hres = np.stack([np.deg2rad(lat_hres), np.deg2rad(lon_h)], axis=1)
    _, idx = cKDTree(pts_lres).query(pts_hres, k=1, workers=-1)
    return idx.astype(np.intp)


# ---------------------------------------------------------------------------
# Per-region weighted statistics
# ---------------------------------------------------------------------------
def _weighted_mse(
    pred: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> float:
    w = weights * mask
    w_sum = w.sum()
    if w_sum == 0.0:
        return float("nan")
    return float(((pred - truth) ** 2 * w).sum() / w_sum)


def _weighted_var(field: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    w = weights * mask
    w_sum = w.sum()
    if w_sum == 0.0:
        return float("nan")
    mu = (field * w).sum() / w_sum
    return float(((field - mu) ** 2 * w).sum() / w_sum)


def _region_stats_for_var(
    y_pred_members: np.ndarray,  # (n_members, n_hres)
    y_members: np.ndarray,       # (n_members, n_hres)
    x_interp_members: np.ndarray,  # (n_members, n_hres)  x on hres grid
    weights: np.ndarray,         # (n_hres,) normalised
    masks: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compute per-region stats, averaged across members."""
    results: dict[str, dict[str, float]] = {}
    for rname, mask in masks.items():
        mse_full_vals, mse_res_vals, mse_base_vals, var_truth_vals, var_truth_res_vals = [], [], [], [], []
        for i in range(y_pred_members.shape[0]):
            yp = y_pred_members[i]
            y = y_members[i]
            xi = x_interp_members[i]
            mse_full_vals.append(_weighted_mse(yp, y, weights, mask))
            mse_res_vals.append(_weighted_mse(yp - xi, y - xi, weights, mask))
            mse_base_vals.append(_weighted_mse(xi, y, weights, mask))
            var_truth_vals.append(_weighted_var(y, weights, mask))
            var_truth_res_vals.append(_weighted_var(y - xi, weights, mask))

        mse_full = float(np.nanmean(mse_full_vals))
        mse_res = float(np.nanmean(mse_res_vals))
        mse_base = float(np.nanmean(mse_base_vals))
        var_truth = float(np.nanmean(var_truth_vals))
        var_truth_res = float(np.nanmean(var_truth_res_vals))

        nmse_full = mse_full / var_truth if var_truth > 0 else float("nan")
        nmse_res = mse_res / var_truth_res if var_truth_res > 0 else float("nan")
        skill = 1.0 - mse_full / mse_base if mse_base > 0 else float("nan")

        results[rname] = {
            "mse_full": mse_full,
            "mse_residual": mse_res,
            "mse_baseline": mse_base,
            "var_truth": var_truth,
            "var_truth_residual": var_truth_res,
            "nmse_full": nmse_full,
            "nmse_residual": nmse_res,
            "skill_vs_interp": skill,
        }
    return results


# ---------------------------------------------------------------------------
# Spectra helpers
# ---------------------------------------------------------------------------
def _try_import_cl_from_unstructured():
    for path in [
        "eval._backends.spectra.calibrate_fast_spectra_proxy",
        "eval.spectra.calibrate_fast_spectra_proxy",
        "eval._backends.scoreboard.spectra_proxy",
    ]:
        try:
            import importlib
            mod = importlib.import_module(path)
            return getattr(mod, "cl_from_unstructured", None)
        except (ImportError, AttributeError):
            continue
    # fallback: look in the spectra evaluator subtree
    try:
        from eval.evaluators.spectra import proxy_runner as _pr
        return getattr(_pr, "cl_from_unstructured", None)
    except (ImportError, AttributeError):
        pass
    return None


def _compute_spectra_for_var(
    y_pred_members: np.ndarray,   # (n_members, n_hres)
    y_members: np.ndarray,
    x_interp_members: np.ndarray,
    lat_hres: np.ndarray,
    lon_hres: np.ndarray,
    cl_fn,
    nside: int,
    lmax: int,
) -> dict[str, list[np.ndarray]]:
    curves: dict[str, list[np.ndarray]] = {
        "pred_cl": [], "truth_cl": [], "residual_pred_cl": [], "residual_truth_cl": [],
    }
    for i in range(y_pred_members.shape[0]):
        yp = y_pred_members[i]
        y = y_members[i]
        xi = x_interp_members[i]
        curves["pred_cl"].append(cl_fn(lat_hres, lon_hres, yp, nside=nside, lmax=lmax))
        curves["truth_cl"].append(cl_fn(lat_hres, lon_hres, y, nside=nside, lmax=lmax))
        curves["residual_pred_cl"].append(cl_fn(lat_hres, lon_hres, yp - xi, nside=nside, lmax=lmax))
        curves["residual_truth_cl"].append(cl_fn(lat_hres, lon_hres, y - xi, nside=nside, lmax=lmax))
    return curves


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def compute_leadtime_scores(
    predictions_dir: Path,
    *,
    surface_vars: list[str] | None = None,
    spectra_vars: list[str] | None = None,
    nside: int = 128,
    lmax: int = 319,
    skip_spectra: bool = False,
) -> dict[str, Any]:
    """Compute per-leadtime surface scores and spectra for all predictions.

    Returns a dict with the schema written to ``leadtime_scores.json``.
    """
    predictions_dir = Path(predictions_dir)
    pred_files = find_predictions(predictions_dir)
    if not pred_files:
        raise ValueError(f"No predictions_*.nc found in {predictions_dir}")

    surface_vars = surface_vars or list(SURFACE_VARIABLES.keys())
    spectra_vars = spectra_vars or SPECTRA_VARS

    cl_fn = None if skip_spectra else _try_import_cl_from_unstructured()
    if not skip_spectra and cl_fn is None:
        LOG.warning(
            "Could not import cl_from_unstructured; spectra will be skipped. "
            "Pass skip_spectra=True to suppress this warning."
        )

    # Group by step
    steps_seen: set[int] = {pf.step for pf in pred_files}
    steps = sorted(steps_seen)

    # Accumulators: step → var → region → list[per-file-mean dict]
    acc_surface: dict[int, dict[str, dict[str, list[dict[str, float]]]]] = {
        s: {v: {r: [] for r in REGIONS} for v in surface_vars} for s in steps
    }
    # step → var → curve_key → list[np.ndarray]
    acc_spectra: dict[int, dict[str, dict[str, list[np.ndarray]]]] = {
        s: {} for s in steps
    }

    # Cache: (lat_lres_hash, lat_hres_hash) → nn_idx
    _nn_cache: dict[tuple[int, int], np.ndarray] = {}

    for pf in pred_files:
        LOG.info("Processing %s (step=%dh)...", pf.path.name, pf.step)
        with xr.open_dataset(pf.path, cache=False) as ds:
            ws_index = _weather_state_index(ds)
            n_hres = int(ds["lat_hres"].size)
            weights = _area_weights(ds, n_hres)  # normalised cos(lat) weights

            lat_hres = np.asarray(ds["lat_hres"].values, dtype=np.float64).reshape(-1)
            lon_hres = np.asarray(ds["lon_hres"].values, dtype=np.float64).reshape(-1)
            lat_lres = np.asarray(ds["lat_lres"].values, dtype=np.float64).reshape(-1)
            lon_lres = np.asarray(ds["lon_lres"].values, dtype=np.float64).reshape(-1)

            masks = _build_region_masks(lat_hres, lon_hres)

            # Build or retrieve NN map (lres → hres)
            cache_key = (id(lat_lres.data), id(lat_hres.data))  # cheap proxy
            hashed_key = (int(lat_lres.size), int(lat_hres.size))
            if hashed_key not in _nn_cache:
                LOG.debug("Building lres→hres NN map (%d → %d pts)", lat_lres.size, lat_hres.size)
                _nn_cache[hashed_key] = _build_nn_map(lat_lres, lon_lres, lat_hres, lon_hres)
            nn_idx = _nn_cache[hashed_key]

            y_pred_da = _to_member_point_weather(ds["y_pred"], ds, label="y_pred")
            y_da = _to_member_point_weather(ds["y"], ds, label="y")

            # x_interp: prefer dedicated variable (hres grid), else NN-map x (lres grid)
            has_x_interp = "x_interp" in ds.variables
            if has_x_interp:
                x_interp_da = _to_member_point_weather(ds["x_interp"], ds, label="x_interp")
            else:
                x_da = _to_member_point_weather_lres(ds["x"])

            n_members = int(y_pred_da.sizes["member"])

            for var in surface_vars:
                if var not in ws_index:
                    continue
                idx = ws_index[var]
                yp_arr = np.asarray(y_pred_da.isel(weather_state=idx).values, dtype=np.float64)
                y_arr = np.asarray(y_da.isel(weather_state=idx).values, dtype=np.float64)

                if has_x_interp and var in ws_index:
                    xi_arr = np.asarray(x_interp_da.isel(weather_state=idx).values, dtype=np.float64)
                else:
                    x_arr = np.asarray(x_da.isel(weather_state=idx).values, dtype=np.float64)
                    xi_arr = x_arr[:, nn_idx]

                region_stats = _region_stats_for_var(yp_arr, y_arr, xi_arr, weights, masks)
                for rname, stats in region_stats.items():
                    acc_surface[pf.step][var][rname].append(stats)

            if cl_fn is not None:
                for var in spectra_vars:
                    if var not in ws_index:
                        continue
                    idx = ws_index[var]
                    yp_arr = np.asarray(y_pred_da.isel(weather_state=idx).values, dtype=np.float64)
                    y_arr = np.asarray(y_da.isel(weather_state=idx).values, dtype=np.float64)

                    if has_x_interp and var in ws_index:
                        xi_arr = np.asarray(x_interp_da.isel(weather_state=idx).values, dtype=np.float64)
                    else:
                        x_arr = np.asarray(x_da.isel(weather_state=idx).values, dtype=np.float64)
                        xi_arr = x_arr[:, nn_idx]

                    curves = _compute_spectra_for_var(
                        yp_arr, y_arr, xi_arr, lat_hres, lon_hres, cl_fn, nside, lmax
                    )
                    if var not in acc_spectra[pf.step]:
                        acc_spectra[pf.step][var] = {k: [] for k in curves}
                    for k, v in curves.items():
                        acc_spectra[pf.step][var][k].extend(v)

    # Aggregate surface scores
    by_step: dict[str, Any] = {}
    for step in steps:
        by_step[str(step)] = {}
        for rname in REGIONS:
            by_step[str(step)][rname] = {}
            for var in surface_vars:
                entries = acc_surface[step][var][rname]
                if not entries:
                    continue
                agg: dict[str, float] = {}
                for key in entries[0]:
                    vals = [e[key] for e in entries if not np.isnan(e[key])]
                    agg[key] = float(np.mean(vals)) if vals else float("nan")
                by_step[str(step)][rname][var] = agg

    # Aggregate spectra (mean Cl across members/dates)
    spectra_out: dict[str, Any] = {}
    for step in steps:
        spectra_out[str(step)] = {}
        for var, curves in acc_spectra[step].items():
            spectra_out[str(step)][var] = {}
            for k, arrs in curves.items():
                if arrs:
                    spectra_out[str(step)][var][k] = np.nanmean(
                        np.stack(arrs, axis=0), axis=0
                    ).tolist()

    return {
        "steps": steps,
        "regions": list(REGIONS.keys()),
        "surface_vars": surface_vars,
        "spectra_vars": spectra_vars if cl_fn is not None else [],
        "n_prediction_files": len(pred_files),
        "nside": nside,
        "lmax": lmax,
        "by_step": by_step,
        "spectra": spectra_out,
    }
