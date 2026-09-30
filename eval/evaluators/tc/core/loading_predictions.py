"""Prediction NetCDF I/O for TC evaluation."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from .data_types import BoundingBox, CurveVectors, SupportMode, curve_support_signature
from .grid import normalize_lon, point_mask, structured_grid_from_points, interp_structured, nearest_point_indices


def discover_prediction_files(pred_dir: Path) -> list[tuple[Path, int, int]]:
    files = sorted(pred_dir.glob("predictions_*.nc"))
    rx = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")
    out: list[tuple[Path, int, int]] = []
    for path in files:
        match = rx.match(path.name)
        if not match:
            continue
        out.append((path, int(match.group(1)), int(match.group(2))))
    return out


def select_prediction_files_for_event(
    pred_files: Iterable[tuple[Path, int, int]],
    event,
) -> list[tuple[Path, int, int]]:
    """Filter prediction files to those matching an event's year/month/dates."""
    selected: list[tuple[Path, int, int]] = []
    allowed_days = {int(day) for day in event.dates}
    for path, ymd, step in pred_files:
        ymd_s = f"{ymd:08d}"
        if ymd_s[:4] != event.year or ymd_s[4:6] != event.month:
            continue
        if int(ymd_s[6:8]) in allowed_days:
            selected.append((path, ymd, step))
    return selected


def event_days_steps(pred_files: Iterable[tuple[Path, int, int]]) -> tuple[list[int], list[int]]:
    pred_files = list(pred_files)
    days = sorted({int(f"{ymd:08d}"[6:8]) for _, ymd, _ in pred_files})
    steps = sorted({step for _, _, step in pred_files})
    return days, steps


def forecast_dates_for_event(event, days: Iterable[int] | None = None) -> list[str]:
    if days is None:
        return [f"{event.year}{event.month}{day}" for day in event.dates]
    return [f"{event.year}{event.month}{int(day):02d}" for day in sorted(set(days))]


def analysis_dates_for_event(event, days: Iterable[int] | None = None) -> list[str]:
    if days is None:
        return list(event.analysis_dates)
    return [f"{event.year}{event.month}{int(day):02d}" for day in sorted(set(days))]


def prediction_point_coordinates(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Extract flattened lon/lat arrays from a prediction dataset."""
    lon_da = ds["lon_hres"]
    lat_da = ds["lat_hres"]
    if lon_da.ndim == 1 and lat_da.ndim == 1 and lon_da.dims == lat_da.dims:
        lon = normalize_lon(np.asarray(lon_da.values, dtype=np.float64)).reshape(-1)
        lat = np.asarray(lat_da.values, dtype=np.float64).reshape(-1)
    elif lon_da.ndim == 1 and lat_da.ndim == 1:
        lon_axis = normalize_lon(np.asarray(lon_da.values, dtype=np.float64))
        lat_axis = np.asarray(lat_da.values, dtype=np.float64)
        lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)
        lon = lon_grid.reshape(-1)
        lat = lat_grid.reshape(-1)
    elif lon_da.ndim == 2 and lat_da.ndim == 2 and lon_da.dims == lat_da.dims:
        # Multi-ds predictions store lon/lat per-member; take first member since
        # coordinates are identical across members.
        lon = normalize_lon(np.asarray(lon_da.values[0], dtype=np.float64)).reshape(-1)
        lat = np.asarray(lat_da.values[0], dtype=np.float64).reshape(-1)
    else:
        raise ValueError(
            f"Unsupported lon_hres/lat_hres coordinate layout: "
            f"{lon_da.dims}/{lat_da.dims}"
        )
    if lon.shape != lat.shape:
        raise ValueError("Prediction lon_hres/lat_hres must have the same flattened size")
    return lon, lat


def load_prediction_curves(
    pred_files: Iterable[tuple[Path, int, int]],
    *,
    bbox: BoundingBox,
    support_mode: SupportMode,
    target_lon: np.ndarray | None = None,
    target_lat: np.ndarray | None = None,
    prediction_var: str = "y_pred",
) -> CurveVectors:
    """Load prediction curves, applying bbox as spatial mask/crop."""
    pred_files = list(pred_files)
    if not pred_files:
        raise ValueError("No prediction files provided")

    if support_mode == "native":
        return _load_prediction_curve_native(pred_files, bbox=bbox, prediction_var=prediction_var)
    if support_mode == "regridded":
        if target_lon is None or target_lat is None:
            raise ValueError("target_lon/target_lat are required for regridded prediction loading")
        return _load_prediction_curve_regridded(
            pred_files,
            target_lon=target_lon,
            target_lat=target_lat,
            prediction_var=prediction_var,
        )
    raise ValueError(f"Unsupported support_mode={support_mode!r}")


# --- Internal helpers ---


def _prediction_values_by_point(ds: xr.Dataset, *, prediction_var: str = "y_pred") -> np.ndarray:
    if prediction_var not in ds:
        raise KeyError(f"Prediction dataset is missing variable {prediction_var!r}")
    y_pred = ds[prediction_var]
    if "sample" in y_pred.dims:
        y_pred = y_pred.isel(sample=0, drop=True)
    spatial_dims = _prediction_spatial_dims(ds, y_pred)
    member_dims = [dim for dim in y_pred.dims if dim not in (*spatial_dims, "weather_state")]
    if len(member_dims) > 1:
        raise ValueError(f"Unsupported prediction dimensions: {y_pred.dims}")
    if not member_dims:
        y_pred = y_pred.expand_dims({"ensemble_member": [0]})
        member_dim = "ensemble_member"
    else:
        member_dim = member_dims[0]
    if spatial_dims != ("grid_point_hres",):
        y_pred = y_pred.stack(grid_point_hres=spatial_dims)
    y_pred = y_pred.transpose(member_dim, "grid_point_hres", "weather_state")
    return np.asarray(y_pred.values, dtype=np.float64)


def _prediction_spatial_dims(ds: xr.Dataset, y_pred: xr.DataArray) -> tuple[str, ...]:
    _MEMBER_LIKE = {"ensemble_member", "member", "realization"}
    lon_dims = tuple(d for d in ds["lon_hres"].dims if d not in _MEMBER_LIKE)
    lat_dims = tuple(d for d in ds["lat_hres"].dims if d not in _MEMBER_LIKE)
    if lon_dims == lat_dims and lon_dims:
        if all(dim in y_pred.dims for dim in lon_dims):
            return lon_dims
    if ds["lon_hres"].ndim == 1 and ds["lat_hres"].ndim == 1:
        spatial_dims = tuple(dict.fromkeys((*lat_dims, *lon_dims)))
        if spatial_dims and all(dim in y_pred.dims for dim in spatial_dims):
            return spatial_dims
    if "grid_point_hres" in y_pred.dims:
        return ("grid_point_hres",)
    raise ValueError(f"Could not infer prediction spatial dims from {y_pred.dims}")


def _prediction_structured_grid(ds: xr.Dataset):
    lon, lat = prediction_point_coordinates(ds)
    return structured_grid_from_points(lon, lat, required=False)


def _load_prediction_curve_native(
    pred_files: list[tuple[Path, int, int]],
    *,
    bbox: BoundingBox,
    prediction_var: str = "y_pred",
) -> CurveVectors:
    msl_vals: list[np.ndarray] = []
    wind_vals: list[np.ndarray] = []
    support_signatures: list[str] = []
    for path, _, _ in pred_files:
        with xr.open_dataset(path) as ds:
            weather_states = ds["weather_state"].values.tolist()
            i_msl = weather_states.index("msl")
            i_u10 = weather_states.index("10u")
            i_v10 = weather_states.index("10v")
            lon, lat = prediction_point_coordinates(ds)
            mask = point_mask(lon, lat, bbox)
            if not np.any(mask):
                continue
            support_signatures.append(curve_support_signature("native", lon[mask], lat[mask]))
            y_pred = _prediction_values_by_point(ds, prediction_var=prediction_var)
            msl_vals.append((y_pred[:, mask, i_msl] / 100.0).reshape(-1))
            u10 = y_pred[:, mask, i_u10]
            v10 = y_pred[:, mask, i_v10]
            wind_vals.append(np.sqrt(u10 * u10 + v10 * v10).reshape(-1))

    if not msl_vals or not wind_vals:
        raise RuntimeError("No native prediction values extracted")

    signatures = set(support_signatures)
    if len(signatures) != 1:
        raise ValueError(
            "Native prediction files do not share one spatial support: "
            f"{sorted(signatures)}"
        )
    return CurveVectors(
        msl=np.concatenate(msl_vals),
        wind=np.concatenate(wind_vals),
        support_mode="native",
        support_signature=support_signatures[0],
    )


def _load_prediction_curve_regridded(
    pred_files: list[tuple[Path, int, int]],
    *,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    prediction_var: str = "y_pred",
) -> CurveVectors:
    msl_vals: list[np.ndarray] = []
    wind_vals: list[np.ndarray] = []
    for path, _, _ in pred_files:
        with xr.open_dataset(path) as ds:
            weather_states = ds["weather_state"].values.tolist()
            i_msl = weather_states.index("msl")
            i_u10 = weather_states.index("10u")
            i_v10 = weather_states.index("10v")
            y_pred = _prediction_values_by_point(ds, prediction_var=prediction_var)
            source_lon, source_lat = prediction_point_coordinates(ds)
            source_grid = _prediction_structured_grid(ds)
            if source_grid is not None:
                target_grid = structured_grid_from_points(target_lon, target_lat)
                msl = interp_structured(y_pred[:, :, i_msl], src_grid=source_grid, target_grid=target_grid)
                u10 = interp_structured(y_pred[:, :, i_u10], src_grid=source_grid, target_grid=target_grid)
                v10 = interp_structured(y_pred[:, :, i_v10], src_grid=source_grid, target_grid=target_grid)
            else:
                target_indices = nearest_point_indices(
                    src_lon=source_lon,
                    src_lat=source_lat,
                    target_lon=target_lon,
                    target_lat=target_lat,
                )
                msl = y_pred[:, target_indices, i_msl]
                u10 = y_pred[:, target_indices, i_u10]
                v10 = y_pred[:, target_indices, i_v10]

            msl_vals.append((msl / 100.0).reshape(-1))
            wind_vals.append(np.sqrt(u10 * u10 + v10 * v10).reshape(-1))

    return CurveVectors(
        msl=np.concatenate(msl_vals),
        wind=np.concatenate(wind_vals),
        support_mode="regridded",
        support_signature=curve_support_signature("regridded", target_lon, target_lat),
    )
