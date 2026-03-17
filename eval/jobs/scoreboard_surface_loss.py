#!/usr/bin/env python3
"""Compute area-weighted and variable-weighted surface MSE for scoreboard evaluation."""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

# Surface variables and weights from anemoi-config/training/scalers/downscaling.yaml.
SURFACE_VARIABLES: OrderedDict[str, float] = OrderedDict(
    [
        ("10u", 2.5),
        ("10v", 2.5),
        ("2d", 2.0),
        ("2t", 2.0),
        ("msl", 2.0),
        ("skt", 0.5),
        ("sp", 1.5),
        ("tcw", 1.0),
    ]
)
TOTAL_WEIGHT = float(sum(SURFACE_VARIABLES.values()))  # 14.0


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        raise ValueError("Cannot normalize empty area weights")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Area weights contain non-finite values")
    if np.any(weights < 0):
        raise ValueError("Area weights must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Area weights sum must be > 0")
    return weights / total


def _decode_weather_state(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _infer_spatial_dims(ds: xr.Dataset, da: xr.DataArray) -> tuple[str, ...]:
    if "grid_point_hres" in da.dims:
        return ("grid_point_hres",)

    lon_dims = tuple(ds["lon_hres"].dims)
    lat_dims = tuple(ds["lat_hres"].dims)
    if lon_dims == lat_dims and lon_dims and all(dim in da.dims for dim in lon_dims):
        return lon_dims

    if ds["lon_hres"].ndim == 1 and ds["lat_hres"].ndim == 1:
        candidate = tuple(dict.fromkeys((*lat_dims, *lon_dims)))
        if candidate and all(dim in da.dims for dim in candidate):
            return candidate

    raise ValueError(f"Could not infer spatial dimensions from {da.dims}")


def _to_member_point_weather(da: xr.DataArray, ds: xr.Dataset, *, label: str) -> xr.DataArray:
    if "weather_state" not in da.dims:
        raise ValueError(f"{label} is missing 'weather_state' dimension: {da.dims}")

    if "sample" in da.dims and da.sizes["sample"] == 1:
        da = da.isel(sample=0, drop=True)

    spatial_dims = _infer_spatial_dims(ds, da)
    member_dims = [dim for dim in da.dims if dim not in (*spatial_dims, "weather_state")]
    if not member_dims:
        da = da.expand_dims({"member": [0]})
        member_dim = "member"
    elif len(member_dims) == 1:
        member_dim = member_dims[0]
    else:
        da = da.stack(member=member_dims)
        member_dim = "member"

    if spatial_dims != ("grid_point_hres",):
        da = da.stack(grid_point_hres=spatial_dims)
    if member_dim != "member":
        da = da.rename({member_dim: "member"})

    return da.transpose("member", "grid_point_hres", "weather_state")


def _area_weights(ds: xr.Dataset, n_points: int) -> np.ndarray:
    if "area_weight" in ds.variables:
        candidate = np.asarray(ds["area_weight"].values, dtype=np.float64).reshape(-1)
        if candidate.size == n_points:
            return _normalize_weights(candidate)

    if "lat_hres" in ds.variables:
        lat_da = ds["lat_hres"]
        lat_vals = np.asarray(lat_da.values, dtype=np.float64).reshape(-1)
        if lat_vals.size == n_points:
            return _normalize_weights(np.cos(np.deg2rad(lat_vals)))
        if lat_da.ndim == 1 and "lon_hres" in ds.variables and ds["lon_hres"].ndim == 1:
            lat_grid, _ = xr.broadcast(lat_da, ds["lon_hres"])
            lat_grid_vals = np.asarray(lat_grid.values, dtype=np.float64).reshape(-1)
            if lat_grid_vals.size == n_points:
                return _normalize_weights(np.cos(np.deg2rad(lat_grid_vals)))

    print("Warning: could not infer area weights, using uniform weights")
    return np.full(n_points, 1.0 / n_points, dtype=np.float64)


def _weather_state_index(ds: xr.Dataset) -> dict[str, int]:
    if "weather_state" not in ds.variables and "weather_state" not in ds.coords:
        raise ValueError("Dataset is missing weather_state coordinate")
    weather_states = [_decode_weather_state(v) for v in ds["weather_state"].values.tolist()]
    return {name: idx for idx, name in enumerate(weather_states)}


def _member_area_weighted_mse(
    y_pred_member_point: np.ndarray,
    y_member_point: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    sq_error = np.square(y_pred_member_point - y_member_point)
    valid = np.isfinite(sq_error)

    weighted_valid = valid * weights[np.newaxis, :]
    weight_sums = weighted_valid.sum(axis=1)
    if np.any(weight_sums <= 0.0):
        raise ValueError("No valid weighted points available for one or more members")

    numerators = np.where(valid, sq_error, 0.0) * weights[np.newaxis, :]
    return numerators.sum(axis=1) / weight_sums


def process_predictions_dir(predictions_dir: Path) -> dict[str, Any]:
    pred_files = sorted(predictions_dir.glob("predictions_*.nc"))
    if not pred_files:
        raise ValueError(f"No prediction files found in {predictions_dir}")

    print(f"Found {len(pred_files)} prediction files")
    var_mse_values: dict[str, list[float]] = {var: [] for var in SURFACE_VARIABLES}

    for pred_file in pred_files:
        print(f"Processing {pred_file.name}...")
        with xr.open_dataset(pred_file) as ds:
            for required_var in ("y_pred", "y", "weather_state"):
                if required_var not in ds:
                    raise ValueError(f"{pred_file}: missing required variable '{required_var}'")

            y_pred = _to_member_point_weather(ds["y_pred"], ds, label="y_pred")
            y_true = _to_member_point_weather(ds["y"], ds, label="y")

            if y_pred.sizes["grid_point_hres"] != y_true.sizes["grid_point_hres"]:
                raise ValueError(
                    f"{pred_file}: mismatched grid_point_hres sizes "
                    f"({y_pred.sizes['grid_point_hres']} vs {y_true.sizes['grid_point_hres']})"
                )
            if y_pred.sizes["weather_state"] != y_true.sizes["weather_state"]:
                raise ValueError(
                    f"{pred_file}: mismatched weather_state sizes "
                    f"({y_pred.sizes['weather_state']} vs {y_true.sizes['weather_state']})"
                )

            if y_true.sizes["member"] == 1 and y_pred.sizes["member"] > 1:
                y_true = y_true.isel(member=0, drop=True).expand_dims(member=y_pred["member"].values)
            elif y_true.sizes["member"] != y_pred.sizes["member"]:
                raise ValueError(
                    f"{pred_file}: mismatched member sizes "
                    f"({y_pred.sizes['member']} vs {y_true.sizes['member']})"
                )

            n_points = int(y_pred.sizes["grid_point_hres"])
            weights = _area_weights(ds, n_points)
            ws_index = _weather_state_index(ds)
            missing_vars = [var for var in SURFACE_VARIABLES if var not in ws_index]
            if missing_vars:
                raise ValueError(
                    f"{pred_file}: missing required surface weather_state entries: {missing_vars}"
                )

            y_pred_vals = np.asarray(y_pred.values, dtype=np.float64)
            y_true_vals = np.asarray(y_true.values, dtype=np.float64)

            for var_name in SURFACE_VARIABLES:
                idx = ws_index[var_name]
                per_member_mse = _member_area_weighted_mse(
                    y_pred_vals[:, :, idx],
                    y_true_vals[:, :, idx],
                    weights,
                )
                var_mse_values[var_name].extend(per_member_mse.tolist())

    per_var_results: dict[str, dict[str, Any]] = {}
    weighted_surface_mse = 0.0
    n_member_samples_per_variable: int | None = None

    for var_name, var_weight in SURFACE_VARIABLES.items():
        values = np.asarray(var_mse_values[var_name], dtype=np.float64)
        if values.size == 0:
            raise ValueError(f"No MSE values collected for required variable '{var_name}'")
        mean_mse = float(np.mean(values))
        n_samples = int(values.size)
        normalized_weight = float(var_weight / TOTAL_WEIGHT)

        per_var_results[var_name] = {
            "mean_mse": mean_mse,
            "weight": float(var_weight),
            "normalized_weight": normalized_weight,
            "n_member_samples": n_samples,
        }
        weighted_surface_mse += mean_mse * normalized_weight

        if n_member_samples_per_variable is None:
            n_member_samples_per_variable = n_samples
        elif n_member_samples_per_variable != n_samples:
            raise ValueError(
                "Inconsistent member sample counts across variables; cannot build aggregate summary"
            )

    return {
        "weighted_surface_mse": float(weighted_surface_mse),
        "total_weight": TOTAL_WEIGHT,
        "n_prediction_files": len(pred_files),
        "n_member_samples_per_variable": int(n_member_samples_per_variable or 0),
        "variables": per_var_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute area-weighted and variable-weighted surface MSE for scoreboard"
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        required=True,
        help="Directory containing predictions_*.nc files",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        required=True,
        help="Output JSON file path (e.g., surface_loss_summary.json)",
    )
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    out_json = Path(args.out_json)
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    print(f"Computing surface loss from: {predictions_dir}")
    results = process_predictions_dir(predictions_dir)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"\nResults written to: {out_json}")
    print(f"Weighted surface MSE: {results['weighted_surface_mse']:.6e}")
    print("\nPer-variable breakdown:")
    for var, data in sorted(results["variables"].items()):
        print(
            f"  {var:6s}: MSE={data['mean_mse']:.6e}, "
            f"weight={data['normalized_weight']:.3f}, "
            f"samples={data['n_member_samples']}"
        )


if __name__ == "__main__":
    main()
