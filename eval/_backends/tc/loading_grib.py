"""GRIB file I/O for TC evaluation."""
from __future__ import annotations

import glob as _glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import earthkit.data as ekd
import numpy as np
import xarray as xr

from .data_types import BoundingBox, CurveVectors, FORECAST_STEP_COUNT, SupportMode, step_to_index
from .grid import normalize_lon, point_mask


def _import_metview():
    import metview as mv  # type: ignore

    return mv


def _analysis_row_indices(frame_count: int, step_indices: list[int] | None) -> slice | list[int]:
    offset = 1 if frame_count > FORECAST_STEP_COUNT else 0
    if step_indices is None:
        return slice(offset, None)
    return [offset + idx for idx in step_indices]


def _is_per_date_an(path: str) -> bool:
    """Detect whether a GRIB file is a per-date AN file (type=an, multiple intraday times)."""
    import eccodes

    with open(path, "rb") as f:
        gid = eccodes.codes_grib_new_from_file(f)
        if gid is None:
            return False
        try:
            grib_type = eccodes.codes_get_string(gid, "type")
            return grib_type == "an"
        finally:
            eccodes.codes_release(gid)


def _verification_dates(init_date: str, n_steps: int = FORECAST_STEP_COUNT) -> list[str]:
    """From init YYYYMMDD, return [init+1d, ..., init+{n_steps}d]."""
    dt = datetime.strptime(init_date, "%Y%m%d")
    return [(dt + timedelta(days=d + 1)).strftime("%Y%m%d") for d in range(n_steps)]


def _expand_analysis_files(
    event_dir: Path, analysis_expid: str, analysis_dates: list[str], step_indices: list[int] | None,
) -> list[str]:
    """Expand init-date analysis_dates into per-verification-date file paths.

    For per-date AN files, each init date produces FORECAST_STEP_COUNT verification
    date paths (init+1d through init+5d).  If step_indices is given, only the
    corresponding verification dates are included.
    """
    paths: list[str] = []
    for init_date in analysis_dates:
        vdates = _verification_dates(init_date)
        if step_indices is not None:
            vdates = [vdates[i] for i in step_indices]
        for vd in vdates:
            paths.append(str(event_dir / f"surface_an_{analysis_expid}_{vd}.grib"))
    return paths


def regridded_target_points(
    bbox: BoundingBox,
    regrid_resolution: float,
    sample_grib_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the regular target grid for regridded mode from a sample GRIB."""
    mv = _import_metview()
    field = mv.read(
        data=mv.read(sample_grib_path),
        grid=[regrid_resolution, regrid_resolution],
        area=[bbox.south, bbox.west, bbox.north, bbox.east],
        param="msl",
    )
    ds = field.to_dataset()
    latitudes = np.asarray(ds["latitude"].values, dtype=np.float64)
    longitudes = normalize_lon(np.asarray(ds["longitude"].values, dtype=np.float64))
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    return lon_grid.reshape(-1), lat_grid.reshape(-1)


def load_grib_curves(
    *,
    dir_data_base: str,
    event_name: str,
    analysis_expid: str,
    analysis_dates: list[str],
    forecast_dates: list[str],
    reference_expids: Iterable[str] = (),
    support_mode: SupportMode = "regridded",
    bbox: BoundingBox | None = None,
    regrid_resolution: float = 0.25,
    steps: Iterable[int] | None = None,
    max_pf_members: int | None = None,
) -> dict[str, CurveVectors]:
    """Load analysis + reference curves from GRIB files.

    Returns dict keyed by expid (e.g. {"OPER_O320_0001": ..., "ENFO_O320_0001": ...}).

    Supports two analysis file formats:
    - Rolling-window (legacy): one file per init date, 6 dates × 1 time per file.
    - Per-date AN: one file per calendar date, 1 date × 4 intraday times (type=an).
      Detected automatically; init dates are expanded to verification dates.
    """
    step_indices = None if steps is None else [step_to_index(step) for step in steps]
    expids = list(dict.fromkeys(reference_expids))
    event_dir = Path(dir_data_base) / event_name

    # Detect analysis format from first file
    first_an = str(event_dir / f"surface_an_{analysis_expid}_{analysis_dates[0]}.grib")
    per_date_an = _is_per_date_an(first_an)

    if per_date_an:
        an_files = _expand_analysis_files(event_dir, analysis_expid, analysis_dates, step_indices)
        # step_indices already applied during expansion; disable downstream re-indexing
        an_step_indices = None
    else:
        an_files = [str(event_dir / f"surface_an_{analysis_expid}_{date}.grib") for date in analysis_dates]
        an_step_indices = step_indices

    if support_mode == "native":
        if bbox is None:
            raise ValueError("bbox is required for native support mode")
        curves: dict[str, CurveVectors] = {}
        if per_date_an:
            # Keep one 00Z analysis frame from EACH verification-date file.
            # Loading all files then isel(time=0) silently retained only the first date.
            analysis_curves = [
                _load_native_curve(
                    [path], bbox=bbox, is_analysis=True, per_date_an=True,
                    step_indices=an_step_indices,
                )
                for path in an_files
            ]
            curves[analysis_expid] = CurveVectors(
                msl=np.concatenate([curve.msl for curve in analysis_curves]),
                wind=np.concatenate([curve.wind for curve in analysis_curves]),
            )
        else:
            curves[analysis_expid] = _load_native_curve(
                an_files, bbox=bbox, is_analysis=True, per_date_an=False,
                step_indices=an_step_indices,
            )
        for expid in expids:
            curves[expid] = _load_native_curve(
                [str(event_dir / f"surface_pf_{expid}_{date}.grib") for date in forecast_dates],
                bbox=bbox,
                is_analysis=False,
                step_indices=step_indices,
                max_pf_members=max_pf_members,
            )
        return curves

    if support_mode == "regridded":
        if bbox is None:
            raise ValueError("bbox is required for regridded support mode")
        curves = {}
        curves[analysis_expid] = _load_regridded_curve(
            an_files,
            bbox=bbox,
            regrid_resolution=regrid_resolution,
            is_analysis=True,
            per_date_an=per_date_an,
            step_indices=an_step_indices,
        )
        for expid in expids:
            curves[expid] = _load_regridded_curve(
                [str(event_dir / f"surface_pf_{expid}_{date}.grib") for date in forecast_dates],
                bbox=bbox,
                regrid_resolution=regrid_resolution,
                is_analysis=False,
                step_indices=step_indices,
                max_pf_members=max_pf_members,
            )
        return curves

    raise ValueError(f"Unsupported support_mode={support_mode!r}")


def load_grib_curve_from_paths(
    grib_paths: list[str] | str,
    *,
    support_mode: SupportMode = "native",
    bbox: BoundingBox | None = None,
    regrid_resolution: float = 0.25,
) -> CurveVectors:
    """Load MSL + 10m wind from arbitrary GRIB files (IEKM, ENFO, or any custom reference).

    grib_paths may be a single string with glob wildcards or a list of paths.
    """
    if isinstance(grib_paths, str):
        if "*" in grib_paths or "?" in grib_paths:
            files = sorted(_glob.glob(grib_paths))
        else:
            files = [grib_paths]
    else:
        files = list(grib_paths)
    if not files:
        raise FileNotFoundError(f"No GRIB files matched: {grib_paths}")

    if support_mode == "native":
        if bbox is None:
            raise ValueError("bbox is required for native support mode")
        return _load_native_curve(files, bbox=bbox, is_analysis=False, step_indices=None)

    if support_mode == "regridded":
        if bbox is None:
            raise ValueError("bbox is required for regridded support mode")
        return _load_regridded_curve(
            files,
            bbox=bbox,
            regrid_resolution=regrid_resolution,
            is_analysis=False,
            step_indices=None,
        )

    raise ValueError(f"Unsupported support_mode={support_mode!r}")


# --- Internal helpers ---


def _crop_native_dataset(ds: xr.Dataset, bbox: BoundingBox) -> xr.Dataset:
    """Restrict a native GRIB dataset to the exact event box used by every curve."""
    if "longitude" not in ds.coords or "latitude" not in ds.coords:
        raise ValueError("Native GRIB dataset is missing longitude/latitude coordinates")
    lon = normalize_lon(np.asarray(ds["longitude"].values, dtype=np.float64))
    lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    if lon.ndim != 1 or lat.ndim != 1 or ds["longitude"].dims != ds["latitude"].dims:
        raise ValueError("Native GRIB longitude/latitude coordinates must share one spatial dimension")
    mask = point_mask(lon, lat, bbox)
    if not np.any(mask):
        raise ValueError(f"Native GRIB has no points inside event bbox {bbox}")
    spatial_dim = ds["longitude"].dims[0]
    return ds.assign_coords(longitude=(spatial_dim, lon)).isel({spatial_dim: mask})


def _load_native_curve(
    files: list[str],
    *,
    bbox: BoundingBox,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None = None,
    per_date_an: bool = False,
) -> CurveVectors:
    ds = ekd.from_source("file", files).to_xarray(engine="cfgrib")
    ds = _crop_native_dataset(ds, bbox)
    # For per-date AN files, select 00Z only so each date contributes one frame.
    if per_date_an and "time" in ds.coords:
        ds = ds.isel(time=0)
    return CurveVectors(
        msl=_extract_native_values(
            ds["msl"],
            is_analysis=is_analysis,
            per_date_an=per_date_an,
            step_indices=step_indices,
            max_pf_members=max_pf_members,
            scale=0.01,
        ),
        wind=_extract_native_wind(
            ds,
            is_analysis=is_analysis,
            per_date_an=per_date_an,
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
    per_date_an: bool = False,
) -> np.ndarray:
    if "number" in da.dims and int(da.sizes["number"]) > 1 and max_pf_members is not None:
        da = da.isel(number=slice(0, max_pf_members))

    if is_analysis and not per_date_an:
        # Legacy rolling-window format: skip init-date frame
        if "forecast_reference_time" in da.dims:
            if step_indices is None:
                da = da.isel(forecast_reference_time=slice(1, None))
            else:
                da = da.isel(forecast_reference_time=[1 + idx for idx in step_indices])
    elif not is_analysis and step_indices is not None and "step" in da.dims:
        da = da.isel(step=step_indices)

    return (np.asarray(da.values, dtype=np.float64) * scale).reshape(-1)


def _extract_native_wind(
    ds: xr.Dataset,
    *,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None,
    per_date_an: bool = False,
) -> np.ndarray:
    u10 = _extract_native_values(
        ds["u10"],
        is_analysis=is_analysis,
        per_date_an=per_date_an,
        step_indices=step_indices,
        max_pf_members=max_pf_members,
    )
    v10 = _extract_native_values(
        ds["v10"],
        is_analysis=is_analysis,
        per_date_an=per_date_an,
        step_indices=step_indices,
        max_pf_members=max_pf_members,
    )
    return np.sqrt(u10 * u10 + v10 * v10)


def _load_regridded_curve(
    files: list[str],
    *,
    bbox: BoundingBox,
    regrid_resolution: float,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None = None,
    per_date_an: bool = False,
) -> CurveVectors:
    mv = _import_metview()
    holders = [mv.read(path) for path in files]
    return CurveVectors(
        msl=_extract_regridded_values(
            holders,
            bbox=bbox,
            regrid_resolution=regrid_resolution,
            parameter="msl",
            is_analysis=is_analysis,
            step_indices=step_indices,
            max_pf_members=max_pf_members,
            per_date_an=per_date_an,
        ),
        wind=_extract_regridded_wind(
            holders,
            bbox=bbox,
            regrid_resolution=regrid_resolution,
            is_analysis=is_analysis,
            step_indices=step_indices,
            max_pf_members=max_pf_members,
            per_date_an=per_date_an,
        ),
    )


def _read_regridded_variable(
    data, *, bbox: BoundingBox, regrid_resolution: float, parameter: str,
    time_filter: int | None = None,
) -> np.ndarray:
    mv = _import_metview()
    read_kwargs = dict(
        data=data,
        grid=[regrid_resolution, regrid_resolution],
        area=[bbox.south, bbox.west, bbox.north, bbox.east],
        param=parameter,
    )
    if time_filter is not None:
        read_kwargs["time"] = time_filter
    field = mv.read(**read_kwargs)
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
    bbox: BoundingBox,
    regrid_resolution: float,
    parameter: str,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None,
    per_date_an: bool = False,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    # For per-date AN files: filter to 00Z so each file yields one frame.
    # step_indices already applied during file expansion, so no row indexing needed.
    time_filter = 0 if (is_analysis and per_date_an) else None
    for holder in holders:
        arr = _read_regridded_variable(
            holder, bbox=bbox, regrid_resolution=regrid_resolution,
            parameter=parameter, time_filter=time_filter,
        )
        if is_analysis and not per_date_an:
            arr = arr[_analysis_row_indices(arr.shape[0], step_indices), :, :]
        elif not is_analysis:
            if max_pf_members is not None:
                arr = arr[:max_pf_members, :, :, :]
            if step_indices is not None:
                arr = arr[:, step_indices, :, :]
        chunks.append(arr.reshape(-1))
    return np.concatenate(chunks)


def _extract_regridded_wind(
    holders: list,
    *,
    bbox: BoundingBox,
    regrid_resolution: float,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None,
    per_date_an: bool = False,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    time_filter = 0 if (is_analysis and per_date_an) else None
    for holder in holders:
        u10 = _read_regridded_variable(
            holder, bbox=bbox, regrid_resolution=regrid_resolution,
            parameter="10u", time_filter=time_filter,
        )
        v10 = _read_regridded_variable(
            holder, bbox=bbox, regrid_resolution=regrid_resolution,
            parameter="10v", time_filter=time_filter,
        )
        arr = np.sqrt(u10 * u10 + v10 * v10)
        if is_analysis and not per_date_an:
            arr = arr[_analysis_row_indices(arr.shape[0], step_indices), :, :]
        elif not is_analysis:
            if max_pf_members is not None:
                arr = arr[:max_pf_members, :, :, :]
            if step_indices is not None:
                arr = arr[:, step_indices, :, :]
        chunks.append(arr.reshape(-1))
    return np.concatenate(chunks)
