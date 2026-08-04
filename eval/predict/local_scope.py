"""Local prediction-output scoping helpers.

This module intentionally crops only exported high-resolution prediction arrays.
It is the safe first workflow surface for local/global parity checks; true model
internal graph cutting should reuse the same scope contract but must happen
before model/datamodule construction.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from .types import EnsemblePrediction


def load_local_scope(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Parse and validate a local-scope config mapping."""

    if raw in (None, ""):
        return {"mode": "global"}
    if isinstance(raw, str):
        scope = json.loads(raw)
    elif isinstance(raw, dict):
        scope = dict(raw)
    else:
        raise TypeError(f"local scope must be JSON string or dict, got {type(raw).__name__}")
    if not isinstance(scope, dict):
        raise ValueError(f"local scope must decode to a mapping, got {type(scope).__name__}")

    mode = str(scope.get("mode", "global") or "global")
    scope["mode"] = mode
    if mode == "global":
        return scope
    if mode == "bbox":
        required = ("lat_min", "lat_max", "lon_min", "lon_max")
    elif mode == "radius_km":
        required = ("center_lat", "center_lon", "radius_km")
    else:
        raise ValueError(
            f"Unsupported local scope mode {mode!r}; expected 'global', 'bbox', or 'radius_km'."
        )
    missing = [key for key in required if key not in scope]
    if missing:
        raise ValueError(f"local scope mode {mode!r} missing required key(s): {missing}")
    return scope


def local_scope_enabled(raw: str | dict[str, Any] | None) -> bool:
    """Return whether a scope requests anything other than global output."""

    return load_local_scope(raw).get("mode") != "global"


def _norm_lon(lon) -> np.ndarray:
    return (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _lon_mask(lon: np.ndarray, lon_min: float, lon_max: float) -> np.ndarray:
    lon_n = _norm_lon(lon)
    if abs(float(lon_max) - float(lon_min)) >= 359.999:
        return np.ones(lon_n.shape, dtype=bool)
    lo = float(_norm_lon([lon_min])[0])
    hi = float(_norm_lon([lon_max])[0])
    if lo <= hi:
        return (lon_n >= lo) & (lon_n <= hi)
    return (lon_n >= lo) | (lon_n <= hi)


def _haversine_km(lat: np.ndarray, lon: np.ndarray, center_lat: float, center_lon: float) -> np.ndarray:
    radius_earth_km = 6371.0088
    lat = np.asarray(lat, dtype=np.float64)
    lon = _norm_lon(lon)
    phi1 = np.radians(lat)
    phi2 = math.radians(float(center_lat))
    dphi = np.radians(lat - float(center_lat))
    dlon = np.radians(_norm_lon(lon - float(center_lon)))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * math.cos(phi2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_earth_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def hres_mask_for_scope(lon_hres, lat_hres, raw_scope: str | dict[str, Any] | None) -> np.ndarray:
    """Build a boolean hres-node mask for a local scope."""

    scope = load_local_scope(raw_scope)
    lon = np.asarray(lon_hres, dtype=np.float64).reshape(-1)
    lat = np.asarray(lat_hres, dtype=np.float64).reshape(-1)
    if lon.shape != lat.shape:
        raise ValueError(f"lon_hres/lat_hres shape mismatch: {lon.shape} != {lat.shape}")

    mode = scope["mode"]
    if mode == "global":
        return np.ones(lon.shape, dtype=bool)
    if mode == "bbox":
        lat_lo, lat_hi = sorted((float(scope["lat_min"]), float(scope["lat_max"])))
        return (lat >= lat_lo) & (lat <= lat_hi) & _lon_mask(lon, scope["lon_min"], scope["lon_max"])
    if mode == "radius_km":
        dist = _haversine_km(lat, lon, float(scope["center_lat"]), float(scope["center_lon"]))
        return dist <= float(scope["radius_km"])
    raise AssertionError(f"unreachable local scope mode: {mode}")


def _subset_hres_array(array: np.ndarray | None, mask: np.ndarray, *, name: str) -> np.ndarray | None:
    if array is None:
        return None
    arr = np.asarray(array)
    if arr.ndim < 2:
        raise ValueError(f"{name} must have a hres grid axis at -2, got shape {arr.shape}")
    if arr.shape[-2] != mask.size:
        raise ValueError(
            f"{name} hres axis length {arr.shape[-2]} does not match local-scope mask length {mask.size}"
        )
    return arr[..., mask, :]


def apply_local_output_scope(ensemble: EnsemblePrediction, raw_scope: str | dict[str, Any] | None) -> EnsemblePrediction:
    """Return an ensemble with hres outputs cropped to the requested local scope.

    Low-resolution conditioning arrays are intentionally left on their full grid;
    high-resolution arrays (``y_pred``, ``y``, and ``x_interp``) plus hres
    coordinates are cropped together so existing evaluators can run on the local
    support. This is not a model-internal graph cut.
    """

    scope = load_local_scope(raw_scope)
    if scope["mode"] == "global":
        return ensemble
    if ensemble.lon_hres is None or ensemble.lat_hres is None:
        raise ValueError("Cannot apply local scope without lon_hres/lat_hres coordinates")

    mask = hres_mask_for_scope(ensemble.lon_hres, ensemble.lat_hres, scope)
    count = int(mask.sum())
    if count == 0:
        raise ValueError(f"Local scope {scope!r} selected zero hres grid points")
    if count == int(mask.size):
        return ensemble

    return EnsemblePrediction(
        init_date=ensemble.init_date,
        lead_step_hours=ensemble.lead_step_hours,
        member_ids=list(ensemble.member_ids),
        source_bundle_paths=list(ensemble.source_bundle_paths),
        members_missing_target=list(ensemble.members_missing_target),
        weather_states=list(ensemble.weather_states),
        lon_lres=ensemble.lon_lres,
        lat_lres=ensemble.lat_lres,
        lon_hres=np.asarray(ensemble.lon_hres)[mask],
        lat_hres=np.asarray(ensemble.lat_hres)[mask],
        x_stack=ensemble.x_stack,
        y_stack=_subset_hres_array(ensemble.y_stack, mask, name="y"),
        y_pred_stack=_subset_hres_array(ensemble.y_pred_stack, mask, name="y_pred"),
        x_interp_stack=_subset_hres_array(ensemble.x_interp_stack, mask, name="x_interp"),
        used_missing_target_unsafe=ensemble.used_missing_target_unsafe,
    )
