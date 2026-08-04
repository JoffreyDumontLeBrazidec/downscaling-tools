"""Local/global prediction parity checks."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


_PRED_RE = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")


def _norm_lon(lon) -> np.ndarray:
    return (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _prediction_files(root: Path) -> dict[str, Path]:
    files = {}
    for path in sorted(root.glob("predictions_*.nc")):
        if _PRED_RE.match(path.name):
            files[path.name] = path
    return files


def _coord_key(lat: float, lon: float, tolerance: float) -> tuple[int, int]:
    scale = 1.0 / float(tolerance)
    return (int(round(float(lat) * scale)), int(round(float(_norm_lon([lon])[0]) * scale)))


def _global_indices_for_local(local: xr.Dataset, global_ds: xr.Dataset, tolerance: float) -> list[int]:
    local_lon = _norm_lon(local["lon_hres"].values).reshape(-1)
    local_lat = np.asarray(local["lat_hres"].values, dtype=np.float64).reshape(-1)
    global_lon = _norm_lon(global_ds["lon_hres"].values).reshape(-1)
    global_lat = np.asarray(global_ds["lat_hres"].values, dtype=np.float64).reshape(-1)

    index = {
        _coord_key(lat, lon, tolerance): i
        for i, (lat, lon) in enumerate(zip(global_lat, global_lon, strict=False))
    }
    out: list[int] = []
    missing: list[tuple[float, float]] = []
    for lat, lon in zip(local_lat, local_lon, strict=False):
        key = _coord_key(lat, lon, tolerance)
        if key in index:
            out.append(index[key])
        else:
            missing.append((float(lat), float(lon)))
    if missing:
        preview = ", ".join(f"({lat:.6f},{lon:.6f})" for lat, lon in missing[:5])
        raise ValueError(f"{len(missing)} local hres point(s) not found in global grid within tolerance: {preview}")
    return out


def _array_metrics(local_values: np.ndarray, global_values: np.ndarray) -> dict[str, float | int]:
    if local_values.shape != global_values.shape:
        raise ValueError(f"shape mismatch: local {local_values.shape} vs global {global_values.shape}")
    diff = np.asarray(local_values, dtype=np.float64) - np.asarray(global_values, dtype=np.float64)
    finite = np.isfinite(diff)
    if not np.any(finite):
        return {"n": int(diff.size), "finite_n": 0, "max_abs": math.nan, "mean_abs": math.nan, "rmse": math.nan}
    selected = diff[finite]
    return {
        "n": int(diff.size),
        "finite_n": int(selected.size),
        "max_abs": float(np.max(np.abs(selected))),
        "mean_abs": float(np.mean(np.abs(selected))),
        "rmse": float(np.sqrt(np.mean(selected * selected))),
    }


def _weather_states(ds: xr.Dataset) -> list[str]:
    return [str(value) for value in ds["weather_state"].values.tolist()]


def _tc_extremes(ds: xr.Dataset, *, var: str = "y_pred") -> dict[str, float]:
    states = _weather_states(ds)
    required = ("msl", "10u", "10v")
    missing = [name for name in required if name not in states]
    if missing:
        raise ValueError(f"Cannot compute TC extremes for {var}: missing weather states {missing}")
    values = np.asarray(ds[var].values, dtype=np.float64)
    i_msl = states.index("msl")
    i_u10 = states.index("10u")
    i_v10 = states.index("10v")
    msl_hpa = values[..., i_msl] / 100.0
    wind = np.sqrt(values[..., i_u10] ** 2 + values[..., i_v10] ** 2)
    return {
        "min_msl_hpa": float(np.nanmin(msl_hpa)),
        "max_wind_ms": float(np.nanmax(wind)),
    }


def compute_parity(
    *,
    local_predictions_dir: str | Path,
    global_predictions_dir: str | Path,
    output_dir: str | Path,
    coordinate_tolerance: float = 1e-6,
    variables: list[str] | None = None,
) -> dict[str, Any]:
    """Compare local prediction files against cropped global prediction files."""

    local_predictions_dir = Path(local_predictions_dir).expanduser().resolve()
    global_predictions_dir = Path(global_predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    local_files = _prediction_files(local_predictions_dir)
    global_files = _prediction_files(global_predictions_dir)
    if not local_files:
        raise ValueError(f"No prediction files found in {local_predictions_dir}")

    selected_variables = variables or ["y_pred", "y", "x_interp"]
    file_rows: list[dict[str, Any]] = []
    aggregate: dict[str, list[dict[str, float | int]]] = {name: [] for name in selected_variables}
    tc_rows: list[dict[str, Any]] = []

    for name, local_path in local_files.items():
        global_path = global_files.get(name)
        if global_path is None:
            raise ValueError(f"Missing matching global prediction file for {name} in {global_predictions_dir}")
        with xr.open_dataset(local_path) as local_ds, xr.open_dataset(global_path) as global_ds:
            indices = _global_indices_for_local(local_ds, global_ds, float(coordinate_tolerance))
            global_crop = global_ds.isel(grid_point_hres=indices)
            row: dict[str, Any] = {
                "file": name,
                "local_hres_points": int(local_ds.sizes.get("grid_point_hres", 0)),
                "global_hres_points": int(global_ds.sizes.get("grid_point_hres", 0)),
                "variables": {},
            }
            for var in selected_variables:
                if var not in local_ds or var not in global_crop:
                    continue
                metrics = _array_metrics(local_ds[var].values, global_crop[var].values)
                row["variables"][var] = metrics
                aggregate[var].append(metrics)
            if "y_pred" in local_ds and "y_pred" in global_crop:
                local_tc = _tc_extremes(local_ds, var="y_pred")
                global_tc = _tc_extremes(global_crop, var="y_pred")
                tc_row = {
                    "file": name,
                    "local": local_tc,
                    "global_cropped": global_tc,
                    "abs_diff": {
                        "min_msl_hpa": abs(local_tc["min_msl_hpa"] - global_tc["min_msl_hpa"]),
                        "max_wind_ms": abs(local_tc["max_wind_ms"] - global_tc["max_wind_ms"]),
                    },
                }
                row["tc_extremes"] = tc_row
                tc_rows.append(tc_row)
            file_rows.append(row)

    headline: dict[str, float | int] = {"files": int(len(file_rows))}
    for var, rows in aggregate.items():
        if not rows:
            continue
        headline[f"{var}_max_abs"] = float(max(float(row["max_abs"]) for row in rows))
        headline[f"{var}_rmse_max"] = float(max(float(row["rmse"]) for row in rows))
    if tc_rows:
        headline["tc_min_msl_hpa_abs_diff_max"] = float(max(row["abs_diff"]["min_msl_hpa"] for row in tc_rows))
        headline["tc_max_wind_ms_abs_diff_max"] = float(max(row["abs_diff"]["max_wind_ms"] for row in tc_rows))

    payload = {
        "local_predictions_dir": str(local_predictions_dir),
        "global_predictions_dir": str(global_predictions_dir),
        "coordinate_tolerance": float(coordinate_tolerance),
        "headline": headline,
        "files": file_rows,
    }
    out_path = output_dir / "local_global_parity.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload
