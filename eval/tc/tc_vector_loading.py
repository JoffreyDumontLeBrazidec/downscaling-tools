from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import earthkit.data as ekd
import numpy as np
import xarray as xr

try:
    from .tc_events import TCEvent
except ImportError:  # allow running as a script
    from tc_events import TCEvent


SupportMode = Literal["native", "regridded"]
FORECAST_STEP_COUNT = 5


@dataclass(frozen=True)
class CurveVectors:
    msl: np.ndarray
    wind: np.ndarray


@dataclass(frozen=True)
class StructuredGrid:
    lat_axis: np.ndarray
    lon_axis: np.ndarray
    point_indices: np.ndarray


def _import_metview():
    import metview as mv  # type: ignore

    return mv


def normalize_lon(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def step_to_index(step: int) -> int:
    idx = (int(step) // 24) - 1
    if idx < 0 or idx > 4:
        raise ValueError(f"Unsupported step={step}; expected one of 24,48,72,96,120")
    return idx


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
    cfg: TCEvent,
) -> list[tuple[Path, int, int]]:
    selected: list[tuple[Path, int, int]] = []
    allowed_days = {int(day) for day in cfg.dates}
    for path, ymd, step in pred_files:
        ymd_s = f"{ymd:08d}"
        if ymd_s[:4] != cfg.year or ymd_s[4:6] != cfg.month:
            continue
        if int(ymd_s[6:8]) in allowed_days:
            selected.append((path, ymd, step))
    return selected


def event_days_steps(pred_files: Iterable[tuple[Path, int, int]]) -> tuple[list[int], list[int]]:
    pred_files = list(pred_files)
    days = sorted({int(f"{ymd:08d}"[6:8]) for _, ymd, _ in pred_files})
    steps = sorted({step for _, _, step in pred_files})
    return days, steps


def forecast_dates_for_event(cfg: TCEvent, days: Iterable[int] | None = None) -> list[str]:
    if days is None:
        return [f"{cfg.year}{cfg.month}{day}" for day in cfg.dates]
    return [f"{cfg.year}{cfg.month}{int(day):02d}" for day in sorted(set(days))]


def analysis_dates_for_event(cfg: TCEvent, days: Iterable[int] | None = None) -> list[str]:
    if days is None:
        return list(cfg.analysis_dates)
    return [f"{cfg.year}{cfg.month}{int(day):02d}" for day in sorted(set(days))]


def _analysis_row_indices(frame_count: int, step_indices: list[int] | None) -> slice | list[int]:
    offset = 1 if frame_count > FORECAST_STEP_COUNT else 0
    if step_indices is None:
        return slice(offset, None)
    return [offset + idx for idx in step_indices]


def regridded_target_points_from_grib(
    cfg: TCEvent,
    *,
    dir_data_base: str,
    sample_analysis_date: str,
) -> tuple[np.ndarray, np.ndarray]:
    mv = _import_metview()
    sample_path = Path(dir_data_base) / cfg.name / f"surface_an_{cfg.analysis}_{sample_analysis_date}.grib"
    field = mv.read(
        data=mv.read(str(sample_path)),
        grid=[cfg.regrid_resolution, cfg.regrid_resolution],
        area=[cfg.area_south, cfg.area_west, cfg.area_north, cfg.area_east],
        param="msl",
    )
    ds = field.to_dataset()
    latitudes = np.asarray(ds["latitude"].values, dtype=np.float64)
    longitudes = normalize_lon(np.asarray(ds["longitude"].values, dtype=np.float64))
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    return lon_grid.reshape(-1), lat_grid.reshape(-1)


def load_grib_event_curves(
    cfg: TCEvent,
    *,
    dir_data_base: str,
    ml_exps: Iterable[str] = (),
    extra_reference_expids: Iterable[str] = (),
    support_mode: SupportMode = "regridded",
    forecast_dates: Iterable[str] | None = None,
    analysis_dates: Iterable[str] | None = None,
    steps: Iterable[int] | None = None,
    max_pf_members: int | None = None,
) -> dict[str, CurveVectors]:
    forecast_dates = list(forecast_dates or forecast_dates_for_event(cfg))
    analysis_dates = list(analysis_dates or analysis_dates_for_event(cfg))
    step_indices = None if steps is None else [step_to_index(step) for step in steps]

    expids = list(dict.fromkeys([*list(ml_exps), *list(cfg.reference_expids), *list(extra_reference_expids)]))
    event_dir = Path(dir_data_base) / cfg.name

    if support_mode == "native":
        curves: dict[str, CurveVectors] = {}
        curves[cfg.analysis] = _load_native_curve(
            [
                str(event_dir / f"surface_an_{cfg.analysis}_{date}.grib")
                for date in analysis_dates
            ],
            is_analysis=True,
            step_indices=step_indices,
        )
        for expid in expids:
            curves[expid] = _load_native_curve(
                [
                    str(event_dir / f"surface_pf_{expid}_{date}.grib")
                    for date in forecast_dates
                ],
                is_analysis=False,
                step_indices=step_indices,
                max_pf_members=max_pf_members,
            )
        return curves

    if support_mode == "regridded":
        curves = {}
        curves[cfg.analysis] = _load_regridded_curve(
            [
                str(event_dir / f"surface_an_{cfg.analysis}_{date}.grib")
                for date in analysis_dates
            ],
            cfg=cfg,
            is_analysis=True,
            step_indices=step_indices,
        )
        for expid in expids:
            curves[expid] = _load_regridded_curve(
                [
                    str(event_dir / f"surface_pf_{expid}_{date}.grib")
                    for date in forecast_dates
                ],
                cfg=cfg,
                is_analysis=False,
                step_indices=step_indices,
                max_pf_members=max_pf_members,
            )
        return curves

    raise ValueError(f"Unsupported support_mode={support_mode!r}")


def load_prediction_event_curve(
    pred_files: Iterable[tuple[Path, int, int]],
    *,
    cfg: TCEvent,
    support_mode: SupportMode,
    target_lon: np.ndarray | None = None,
    target_lat: np.ndarray | None = None,
) -> CurveVectors:
    pred_files = list(pred_files)
    if not pred_files:
        raise ValueError(f"No prediction files provided for event={cfg.name}")

    if support_mode == "native":
        return _load_prediction_curve_native(pred_files, cfg=cfg)
    if support_mode == "regridded":
        if target_lon is None or target_lat is None:
            raise ValueError("target_lon/target_lat are required for regridded prediction loading")
        return _load_prediction_curve_regridded(
            pred_files,
            target_lon=target_lon,
            target_lat=target_lat,
        )
    raise ValueError(f"Unsupported support_mode={support_mode!r}")


def _load_native_curve(
    files: list[str],
    *,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None = None,
) -> CurveVectors:
    ds = ekd.from_source("file", files).to_xarray(engine="cfgrib")
    if "longitude" in ds.coords:
        ds = ds.assign_coords(longitude=normalize_lon(ds["longitude"].values.astype(np.float64)))
    return CurveVectors(
        msl=_extract_native_values(
            ds["msl"],
            is_analysis=is_analysis,
            step_indices=step_indices,
            max_pf_members=max_pf_members,
            scale=0.01,
        ),
        wind=_extract_native_wind(
            ds,
            is_analysis=is_analysis,
            step_indices=step_indices,
            max_pf_members=max_pf_members,
        ),
    )


def _extract_native_values(
    da: xr.DataArray,
    *,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None,
    scale: float = 1.0,
) -> np.ndarray:
    if "number" in da.dims and int(da.sizes["number"]) > 1 and max_pf_members is not None:
        da = da.isel(number=slice(0, max_pf_members))

    if is_analysis:
        if "forecast_reference_time" in da.dims:
            if step_indices is None:
                da = da.isel(forecast_reference_time=slice(1, None))
            else:
                da = da.isel(forecast_reference_time=[1 + idx for idx in step_indices])
    elif step_indices is not None and "step" in da.dims:
        da = da.isel(step=step_indices)

    return (np.asarray(da.values, dtype=np.float64) * scale).reshape(-1)


def _extract_native_wind(
    ds: xr.Dataset,
    *,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None,
) -> np.ndarray:
    u10 = _extract_native_values(
        ds["u10"],
        is_analysis=is_analysis,
        step_indices=step_indices,
        max_pf_members=max_pf_members,
    )
    v10 = _extract_native_values(
        ds["v10"],
        is_analysis=is_analysis,
        step_indices=step_indices,
        max_pf_members=max_pf_members,
    )
    return np.sqrt(u10 * u10 + v10 * v10)


def _load_regridded_curve(
    files: list[str],
    *,
    cfg: TCEvent,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None = None,
) -> CurveVectors:
    mv = _import_metview()
    holders = [mv.read(path) for path in files]
    return CurveVectors(
        msl=_extract_regridded_values(
            holders,
            cfg=cfg,
            parameter="msl",
            is_analysis=is_analysis,
            step_indices=step_indices,
            max_pf_members=max_pf_members,
        ),
        wind=_extract_regridded_wind(
            holders,
            cfg=cfg,
            is_analysis=is_analysis,
            step_indices=step_indices,
            max_pf_members=max_pf_members,
        ),
    )


def _read_regridded_variable(data, *, cfg: TCEvent, parameter: str) -> np.ndarray:
    mv = _import_metview()
    field = mv.read(
        data=data,
        grid=[cfg.regrid_resolution, cfg.regrid_resolution],
        area=[cfg.area_south, cfg.area_west, cfg.area_north, cfg.area_east],
        param=parameter,
    )
    dataset = field.to_dataset()
    if parameter == "msl":
        return np.asarray(dataset["msl"].values, dtype=np.float64) / 100.0
    if parameter == "10u":
        return np.asarray(dataset["u10"].values, dtype=np.float64)
    if parameter == "10v":
        return np.asarray(dataset["v10"].values, dtype=np.float64)
    raise ValueError(f"Unsupported parameter={parameter!r}")


def _extract_regridded_values(
    holders: list,
    *,
    cfg: TCEvent,
    parameter: str,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for holder in holders:
        arr = _read_regridded_variable(holder, cfg=cfg, parameter=parameter)
        if is_analysis:
            arr = arr[_analysis_row_indices(arr.shape[0], step_indices), :, :]
        else:
            if max_pf_members is not None:
                arr = arr[:max_pf_members, :, :, :]
            if step_indices is not None:
                arr = arr[:, step_indices, :, :]
        chunks.append(arr.reshape(-1))
    return np.concatenate(chunks)


def _extract_regridded_wind(
    holders: list,
    *,
    cfg: TCEvent,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for holder in holders:
        u10 = _read_regridded_variable(holder, cfg=cfg, parameter="10u")
        v10 = _read_regridded_variable(holder, cfg=cfg, parameter="10v")
        arr = np.sqrt(u10 * u10 + v10 * v10)
        if is_analysis:
            arr = arr[_analysis_row_indices(arr.shape[0], step_indices), :, :]
        else:
            if max_pf_members is not None:
                arr = arr[:max_pf_members, :, :, :]
            if step_indices is not None:
                arr = arr[:, step_indices, :, :]
        chunks.append(arr.reshape(-1))
    return np.concatenate(chunks)


def _load_prediction_curve_native(
    pred_files: list[tuple[Path, int, int]],
    *,
    cfg: TCEvent,
) -> CurveVectors:
    msl_vals: list[np.ndarray] = []
    wind_vals: list[np.ndarray] = []
    for path, _, _ in pred_files:
        with xr.open_dataset(path) as ds:
            weather_states = ds["weather_state"].values.tolist()
            i_msl = weather_states.index("msl")
            i_u10 = weather_states.index("10u")
            i_v10 = weather_states.index("10v")
            lon, lat = _prediction_point_coordinates(ds)
            mask = _event_point_mask(lon, lat, cfg)
            if not np.any(mask):
                continue
            y_pred = _prediction_values_by_point(ds)
            msl_vals.append((y_pred[:, mask, i_msl] / 100.0).reshape(-1))
            u10 = y_pred[:, mask, i_u10]
            v10 = y_pred[:, mask, i_v10]
            wind_vals.append(np.sqrt(u10 * u10 + v10 * v10).reshape(-1))

    if not msl_vals or not wind_vals:
        raise RuntimeError(f"No native prediction values extracted for event={cfg.name}")

    return CurveVectors(
        msl=np.concatenate(msl_vals),
        wind=np.concatenate(wind_vals),
    )


def _load_prediction_curve_regridded(
    pred_files: list[tuple[Path, int, int]],
    *,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> CurveVectors:
    msl_vals: list[np.ndarray] = []
    wind_vals: list[np.ndarray] = []
    for path, _, _ in pred_files:
        with xr.open_dataset(path) as ds:
            weather_states = ds["weather_state"].values.tolist()
            i_msl = weather_states.index("msl")
            i_u10 = weather_states.index("10u")
            i_v10 = weather_states.index("10v")
            y_pred = _prediction_values_by_point(ds)
            source_lon, source_lat = _prediction_point_coordinates(ds)
            source_grid = _prediction_structured_grid(ds)
            if source_grid is not None:
                target_grid = _structured_grid_from_points(target_lon, target_lat)
                msl = _interp_structured_prediction_values(
                    y_pred[:, :, i_msl],
                    src_grid=source_grid,
                    target_grid=target_grid,
                )
                u10 = _interp_structured_prediction_values(
                    y_pred[:, :, i_u10],
                    src_grid=source_grid,
                    target_grid=target_grid,
                )
                v10 = _interp_structured_prediction_values(
                    y_pred[:, :, i_v10],
                    src_grid=source_grid,
                    target_grid=target_grid,
                )
            else:
                target_indices = _nearest_point_indices(
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
    )


def _event_point_mask(lon: np.ndarray, lat: np.ndarray, cfg: TCEvent) -> np.ndarray:
    lat_mask = (lat >= cfg.area_south) & (lat <= cfg.area_north)
    west = normalize_lon(np.asarray([cfg.area_west], dtype=np.float64))[0]
    east = normalize_lon(np.asarray([cfg.area_east], dtype=np.float64))[0]
    if east >= west:
        lon_mask = (lon >= west) & (lon <= east)
    else:
        lon_mask = (lon >= west) | (lon <= east)
    return lat_mask & lon_mask


def _prediction_values_by_point(ds: xr.Dataset) -> np.ndarray:
    y_pred = ds["y_pred"]
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


def _prediction_point_coordinates(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
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
        lon = normalize_lon(np.asarray(lon_da.values, dtype=np.float64)).reshape(-1)
        lat = np.asarray(lat_da.values, dtype=np.float64).reshape(-1)
    else:
        raise ValueError(
            f"Unsupported lon_hres/lat_hres coordinate layout: "
            f"{lon_da.dims}/{lat_da.dims}"
        )
    if lon.shape != lat.shape:
        raise ValueError("Prediction lon_hres/lat_hres must have the same flattened size")
    return lon, lat


def _prediction_structured_grid(ds: xr.Dataset) -> StructuredGrid | None:
    lon, lat = _prediction_point_coordinates(ds)
    return _structured_grid_from_points(lon, lat, required=False)


def _prediction_spatial_dims(ds: xr.Dataset, y_pred: xr.DataArray) -> tuple[str, ...]:
    lon_dims = tuple(ds["lon_hres"].dims)
    lat_dims = tuple(ds["lat_hres"].dims)
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


def _structured_grid_from_points(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    required: bool = True,
) -> StructuredGrid | None:
    lon = normalize_lon(np.asarray(lon, dtype=np.float64)).reshape(-1)
    lat = np.asarray(lat, dtype=np.float64).reshape(-1)
    if lon.shape != lat.shape:
        raise ValueError("lon/lat point arrays must have identical flattened size")

    lon_axis = np.unique(lon)
    lat_axis = np.unique(lat)
    if lon_axis.size * lat_axis.size != lon.size:
        if required:
            raise ValueError("Point set does not describe a structured lat/lon grid")
        return None

    lon_index = np.searchsorted(lon_axis, lon)
    lat_index = np.searchsorted(lat_axis, lat)
    point_indices = np.full((lat_axis.size, lon_axis.size), -1, dtype=np.int64)
    for flat_index, (iy, ix) in enumerate(zip(lat_index, lon_index)):
        if point_indices[iy, ix] != -1:
            if required:
                raise ValueError("Duplicate source points prevent structured-grid reconstruction")
            return None
        point_indices[iy, ix] = flat_index

    if np.any(point_indices < 0):
        if required:
            raise ValueError("Missing source points prevent structured-grid reconstruction")
        return None

    return StructuredGrid(
        lat_axis=np.asarray(lat_axis, dtype=np.float64),
        lon_axis=np.asarray(lon_axis, dtype=np.float64),
        point_indices=point_indices,
    )


def _interp_structured_prediction_values(
    values_by_point: np.ndarray,
    *,
    src_grid: StructuredGrid,
    target_grid: StructuredGrid,
) -> np.ndarray:
    source = np.asarray(values_by_point, dtype=np.float64)[:, src_grid.point_indices]
    data = xr.DataArray(
        source,
        dims=("member", "lat", "lon"),
        coords={"lat": src_grid.lat_axis, "lon": src_grid.lon_axis},
    )
    interpolated = data.interp(
        lat=target_grid.lat_axis,
        lon=target_grid.lon_axis,
        method="linear",
    )
    return np.asarray(interpolated.values, dtype=np.float64)


def _nearest_point_indices(
    *,
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(np.column_stack([src_lon, src_lat]))
        _, idx = tree.query(np.column_stack([target_lon, target_lat]), k=1)
        return np.asarray(idx, dtype=np.int64)
    except Exception:
        idx = np.empty(target_lon.shape[0], dtype=np.int64)
        for i, (lon, lat) in enumerate(zip(target_lon, target_lat)):
            idx[i] = int(np.argmin((src_lon - lon) ** 2 + (src_lat - lat) ** 2))
        return idx
