from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import xarray as xr

DEFAULT_CONSTANT_FORCINGS_NPZ = "/home/ecm5702/hpcperm/data/o320-forcings.npz"
FALLBACK_CONSTANT_FORCINGS_NPZ = (
    "/home/ecm5702/hpcperm/data/o48-forcings.npz",
    "/home/ecm5702/hpcperm/data/o1280-forcings.npz",
    "/home/ecm5702/hpcperm/data/forcings_for_anemoi_inference/forcings_o320.npz",
    "/home/ecm5702/hpcperm/data/forcings_for_anemoi_inference/forcings_o1280.npz",
)

SFC_TO_CFGRIB = {
    "10u": "u10",
    "10v": "v10",
    "2d": "d2m",
    "2t": "t2m",
    "cp": "cp",
    "msl": "msl",
    "skt": "skt",
    "sp": "sp",
    "tcw": "tcw",
    "tp": "tp",
}

OPTIONAL_ZERO_LRES_SFC_VARS = frozenset(
    {
        "cp",
        "hcc",
        "lcc",
        "mcc",
        "ssrd",
        "strd",
        "tcc",
        "tp",
    }
)


def split_level_channel(name: str) -> tuple[str, int | None]:
    match = re.match(r"^(.*)_([0-9]+)$", name)
    if match is None:
        return name, None
    return match.group(1), int(match.group(2))


def _cleanup_empty_cfgrib_indexes(grib_path: str | Path) -> None:
    path = Path(grib_path)
    for idx_path in path.parent.glob(f"{path.name}.*.idx"):
        try:
            if idx_path.is_file() and idx_path.stat().st_size == 0:
                idx_path.unlink()
        except FileNotFoundError:
            continue


def _open_cfgrib_dataset(grib_path: str | Path) -> xr.Dataset:
    _cleanup_empty_cfgrib_indexes(grib_path)
    import earthkit.data as ekd  # pylint: disable=import-outside-toplevel

    return ekd.from_source("file", str(grib_path)).to_xarray(engine="cfgrib")


def _infer_target_gribs_from_hres(
    hres_grib: str | Path,
) -> tuple[Path | None, Path | None]:
    """Try to infer colocated target GRIB paths from hres source file name.

    Example:
      enfo_o320_0001_date20230829_time0000_step24to120_sfc.grib
    -> enfo_o320_0001_date20230829_time0000_mem1to10_step24to120_sfc_y.grib
       enfo_o320_0001_date20230829_time0000_mem1to10_step24to120_pl_y.grib

      enfo_o96_0001_date20250926_time0000_step24to120_sfc.grib
    -> iekm_o96_iekm_date20250926_time0000_step24to120_sfc_y.grib
       iekm_o96_iekm_date20250926_time0000_step24to120_pl_y.grib
    """
    hres_path = Path(hres_grib)
    name = hres_path.name
    candidate_pairs: list[tuple[str, str]] = []

    m = re.match(
        r"^(enfo_o320_[^_]+_date\d{8}_time\d{4})_(step[^_]+)_sfc\.grib$",
        name,
    )
    if m is not None:
        prefix = m.group(1)
        step_token = m.group(2)
        candidate_pairs.append(
            (
                f"{prefix}_mem1to10_{step_token}_sfc_y.grib",
                f"{prefix}_mem1to10_{step_token}_pl_y.grib",
            )
        )

    m = re.match(
        r"^enfo_o96_[^_]+_date(\d{8})_time(\d{4})_(step[^_]+)_sfc\.grib$",
        name,
    )
    if m is not None:
        date_token = m.group(1)
        time_token = m.group(2)
        step_token = m.group(3)
        candidate_pairs.append(
            (
                f"iekm_o96_iekm_date{date_token}_time{time_token}_{step_token}_sfc_y.grib",
                f"iekm_o96_iekm_date{date_token}_time{time_token}_{step_token}_pl_y.grib",
            )
        )

    if not candidate_pairs:
        return None, None

    search_roots: list[Path] = [hres_path.parent]
    if hres_path.parent != hres_path.parent.parent:
        search_roots.append(hres_path.parent.parent)

    def _resolve(filename: str) -> Path | None:
        for root in search_roots:
            candidate = root / filename
            if candidate.exists():
                return candidate
        return None

    for sfc_name, pl_name in candidate_pairs:
        sfc_path = _resolve(sfc_name)
        pl_path = _resolve(pl_name)
        if sfc_path is not None or pl_path is not None:
            return sfc_path, pl_path
    return None, None


def parse_valid_time(value, fallback) -> datetime:
    raw = value if value is not None else fallback
    if raw is None or str(raw).strip().lower() == "unknown":
        raise ValueError(
            "No valid time available. Provide `valid_time_override` or ensure GRIB contains `valid_time`."
        )
    return datetime.fromisoformat(str(raw).replace("Z", "").split(".")[0])


def _open_bundle_dataset(path: str | Path) -> xr.Dataset:
    bundle_path = Path(path)
    if bundle_path.suffix == ".zarr" or bundle_path.is_dir():
        return xr.open_zarr(bundle_path, consolidated=False)
    return xr.open_dataset(bundle_path)


def _interpolate_level_points(
    field: xr.DataArray,
    available_levels: Sequence[int],
    target_level: int,
) -> np.ndarray:
    levels_np = np.asarray(available_levels, dtype=np.int32)
    order = np.argsort(levels_np)
    sorted_levels = levels_np[order]
    if target_level < int(sorted_levels[0]) or target_level > int(sorted_levels[-1]):
        raise KeyError(f"Missing level {target_level}")
    if "level" not in field.dims:
        raise KeyError("Missing 'level' dimension for pressure-level interpolation")
    values = np.asarray(field.transpose("level", ...).values, dtype=np.float32).reshape(len(levels_np), -1)
    values = values[order, :]
    interp = np.empty(values.shape[1], dtype=np.float32)
    for idx in range(values.shape[1]):
        interp[idx] = np.interp(float(target_level), sorted_levels, values[:, idx])
    return interp


def open_bundle_dataset(path: str | Path) -> xr.Dataset:
    return _open_bundle_dataset(path)


def _borrow_or_open_bundle_dataset(
    bundle_nc: str | Path | xr.Dataset,
) -> tuple[xr.Dataset, bool]:
    if isinstance(bundle_nc, xr.Dataset):
        return bundle_nc, False
    return open_bundle_dataset(bundle_nc), True


def _get_pl_level_coord(ds_pl: xr.Dataset) -> str:
    for name in ds_pl.coords:
        if "isobaric" in name.lower() or name.lower() == "level":
            return name
    raise KeyError("No pressure-level coordinate found in PL GRIB dataset")


def _extract_sfc_field(ds_sfc: xr.Dataset, channel_name: str) -> np.ndarray:
    key = SFC_TO_CFGRIB.get(channel_name, channel_name)
    if key not in ds_sfc:
        raise KeyError(f"Missing SFC variable for channel {channel_name} (expected {key})")
    return np.asarray(ds_sfc[key].values, dtype=np.float32).squeeze()


def _extract_pl_field(ds_pl: xr.Dataset, base: str, level: int) -> np.ndarray:
    if base not in ds_pl:
        raise KeyError(f"Missing PL variable: {base}")
    level_coord = _get_pl_level_coord(ds_pl)
    levels = np.asarray(ds_pl[level_coord].values).astype(int).tolist()
    if int(level) not in levels:
        raise KeyError(f"Missing PL level {level} for variable {base}")
    return np.asarray(ds_pl[base].sel({level_coord: int(level)}).values, dtype=np.float32).squeeze()


def _load_constant_forcings_for_size(
    npz_candidates: Sequence[str | Path],
    target_size: int,
) -> tuple[dict[str, np.ndarray], str | None, list[str]]:
    constant_values: dict[str, np.ndarray] = {}
    tried_paths: list[str] = []

    for candidate in npz_candidates:
        npz_path = Path(candidate)
        if not npz_path.exists():
            continue
        tried_paths.append(str(npz_path))
        with np.load(npz_path) as npz:
            current: dict[str, np.ndarray] = {}
            for key in npz.files:
                arr = np.asarray(npz[key], dtype=np.float32).reshape(-1)
                if arr.size == target_size:
                    current[key] = arr
        if current:
            constant_values = current
            return constant_values, str(npz_path), tried_paths

    return constant_values, None, tried_paths


def load_lres_fields_from_grib(
    sfc_grib: str | Path,
    pl_grib: str | Path,
    lres_channel_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    ds_sfc = _open_cfgrib_dataset(sfc_grib)
    ds_pl = _open_cfgrib_dataset(pl_grib)
    fields: dict[str, np.ndarray] = {}
    for name in lres_channel_names:
        base, level = split_level_channel(name)
        if level is None:
            fields[name] = _extract_sfc_field(ds_sfc, name)
        else:
            fields[name] = _extract_pl_field(ds_pl, base, level)
    valid_time = ds_sfc.get("valid_time")
    valid_time = None if valid_time is None else np.asarray(valid_time.values).squeeze()
    return fields, valid_time


def read_hres_static_from_grib(hres_grib: str | Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    import earthkit.data as ekd  # pylint: disable=import-outside-toplevel

    ds_hr = _open_cfgrib_dataset(hres_grib)
    z = np.asarray(ds_hr["z"].values, dtype=np.float32).squeeze() if "z" in ds_hr else None
    lsm = np.asarray(ds_hr["lsm"].values, dtype=np.float32).squeeze() if "lsm" in ds_hr else None
    return z, lsm


def fill_hres_features(
    x_hres,
    name_to_idx_hres: Mapping[str, int],
    lat_hres: np.ndarray,
    lon_hres: np.ndarray,
    dt: datetime,
    device,
    *,
    z: np.ndarray | None = None,
    lsm: np.ndarray | None = None,
    constant_forcings_npz: str | Path | None = DEFAULT_CONSTANT_FORCINGS_NPZ,
) -> None:
    import earthkit.data as ekd  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    from anemoi.transform.grids.unstructured import (
        UnstructuredGridFieldList,
    )  # pylint: disable=import-outside-toplevel

    constant_values: dict[str, np.ndarray] = {}
    constant_source: str | None = None
    tried_constant_npz: list[str] = []
    npz_candidates: list[str | Path] = []
    if constant_forcings_npz is not None:
        npz_candidates.append(constant_forcings_npz)
        for fallback in FALLBACK_CONSTANT_FORCINGS_NPZ:
            if str(fallback) != str(constant_forcings_npz):
                npz_candidates.append(fallback)
        constant_values, constant_source, tried_constant_npz = _load_constant_forcings_for_size(
            npz_candidates, lat_hres.size
        )

    # Prefer NPZ constants for strict parity with runner constant forcings.
    constant_features = (
        "z",
        "lsm",
        "cos_latitude",
        "sin_latitude",
        "cos_longitude",
        "sin_longitude",
    )
    for name in constant_features:
        if name not in name_to_idx_hres:
            continue
        if name in constant_values:
            arr = constant_values[name]
        elif name == "z" and z is not None:
            arr = np.asarray(z, dtype=np.float32).reshape(-1)
        elif name == "lsm" and lsm is not None:
            arr = np.asarray(lsm, dtype=np.float32).reshape(-1)
        elif name in {"cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude"}:
            lat_rad = np.deg2rad(lat_hres.astype(np.float64))
            lon_rad = np.deg2rad(lon_hres.astype(np.float64))
            geometric = {
                "cos_latitude": np.cos(lat_rad),
                "sin_latitude": np.sin(lat_rad),
                "cos_longitude": np.cos(lon_rad),
                "sin_longitude": np.sin(lon_rad),
            }
            arr = geometric[name].astype(np.float32)
        else:
            tried = ", ".join(tried_constant_npz) if tried_constant_npz else "(no existing NPZ found)"
            raise KeyError(
                f"Missing required high-res constant forcing '{name}'. "
                f"Provide it in {constant_forcings_npz!s} or the bundle. "
                f"Expected flattened size={lat_hres.size}. "
                f"Resolved NPZ source={constant_source or 'none'}. "
                f"Tried: {tried}"
            )
        x_hres[0, 0, 0, :, name_to_idx_hres[name]] = torch.from_numpy(arr).to(device)

    # Compute dynamic forcings via earthkit to match anemoi-inference behavior.
    dynamic_features = [
        name
        for name in (
            "cos_julian_day",
            "sin_julian_day",
            "cos_local_time",
            "sin_local_time",
            "insolation",
        )
        if name in name_to_idx_hres
    ]
    if dynamic_features:
        source = UnstructuredGridFieldList.from_values(
            latitudes=np.asarray(lat_hres, dtype=np.float64),
            longitudes=np.asarray(lon_hres, dtype=np.float64),
        )
        forcings = ekd.from_source("forcings", source, date=[dt], param=dynamic_features)
        computed = {
            f.metadata("param"): np.asarray(f.to_numpy(flatten=True), dtype=np.float32)
            for f in forcings
        }
        for name in dynamic_features:
            if name not in computed:
                raise KeyError(f"Missing computed forcing from earthkit: {name}")
            x_hres[0, 0, 0, :, name_to_idx_hres[name]] = torch.from_numpy(
                computed[name]
            ).to(device)


def load_inputs_from_bundle_numpy(
    bundle_nc: str | Path | xr.Dataset,
    name_to_idx_lres: Mapping[str, int],
    name_to_idx_hres: Mapping[str, int],
    *,
    valid_time_override=None,
    constant_forcings_npz: str | Path | None = DEFAULT_CONSTANT_FORCINGS_NPZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bundle, should_close = _borrow_or_open_bundle_dataset(bundle_nc)
    try:
        n_lres = int(bundle.sizes["point_lres"])
        n_hres = int(bundle.sizes["point_hres"])

        x_lres = np.zeros((n_lres, len(name_to_idx_lres)), dtype=np.float32)
        x_hres = np.zeros((n_hres, len(name_to_idx_hres)), dtype=np.float32)

        levels_bundle = (
            set(int(v) for v in np.asarray(bundle["level"].values).tolist())
            if "level" in bundle
            else set()
        )

        # Read grid coordinates and valid time early so they are available for
        # computing dynamic temporal features on the LR side when absent from bundle.
        lat_lres = bundle["lat_lres"].values.astype(np.float32)
        lon_lres = bundle["lon_lres"].values.astype(np.float32)
        lat_hres = bundle["lat_hres"].values.astype(np.float32)
        lon_hres = bundle["lon_hres"].values.astype(np.float32)
        z = bundle["in_hres_z"].values.astype(np.float32) if "in_hres_z" in bundle else None
        lsm = bundle["in_hres_lsm"].values.astype(np.float32) if "in_hres_lsm" in bundle else None
        dt = parse_valid_time(valid_time_override, bundle.attrs.get("case_valid_time"))

        _DYNAMIC_TEMPORAL_FEATURES = frozenset(
            (
                "cos_julian_day", "sin_julian_day",
                "cos_local_time", "sin_local_time",
                "cos_solar_zenith_angle",
                "insolation",
            )
        )
        # Geometric constants computable directly from grid coordinates.
        _GEOMETRIC_CONSTANT_FEATURES = frozenset(
            ("cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude")
        )
        _lres_dynamic_cache: dict[str, np.ndarray] = {}

        # Load LR constant forcings (z, lsm, …) from npz if available for this grid size.
        _lres_npz_constants: dict[str, np.ndarray] = {}
        _lres_npz_candidates = [constant_forcings_npz] + list(FALLBACK_CONSTANT_FORCINGS_NPZ)
        _lres_npz_constants, _, _ = _load_constant_forcings_for_size(
            _lres_npz_candidates, lat_lres.size
        )

        for name, idx in name_to_idx_lres.items():
            base, level = split_level_channel(name)
            if level is None:
                field_name = f"in_lres_{name}"
                if field_name not in bundle:
                    if name in _DYNAMIC_TEMPORAL_FEATURES:
                        # Compute via earthkit on the LR grid (same approach as HRES).
                        if not _lres_dynamic_cache:
                            import earthkit.data as ekd  # pylint: disable=import-outside-toplevel
                            from anemoi.transform.grids.unstructured import (
                                UnstructuredGridFieldList,
                            )  # pylint: disable=import-outside-toplevel

                            needed = [
                                n for n in _DYNAMIC_TEMPORAL_FEATURES if f"in_lres_{n}" not in bundle
                                and n in name_to_idx_lres
                            ]
                            source_lres = UnstructuredGridFieldList.from_values(
                                latitudes=np.asarray(lat_lres, dtype=np.float64),
                                longitudes=np.asarray(lon_lres, dtype=np.float64),
                            )
                            forcings_lres = ekd.from_source(
                                "forcings", source_lres, date=[dt], param=needed
                            )
                            _lres_dynamic_cache = {
                                f.metadata("param"): np.asarray(
                                    f.to_numpy(flatten=True), dtype=np.float32
                                )
                                for f in forcings_lres
                            }
                        if name not in _lres_dynamic_cache:
                            raise KeyError(
                                f"Missing bundle field: {field_name} (dynamic feature not returned by earthkit)"
                            )
                        raw = _lres_dynamic_cache[name]
                    elif name in _GEOMETRIC_CONSTANT_FEATURES:
                        # Trivially computable from grid coordinates.
                        lat_rad = np.deg2rad(lat_lres.astype(np.float64))
                        lon_rad = np.deg2rad(lon_lres.astype(np.float64))
                        _geo = {
                            "cos_latitude": np.cos(lat_rad),
                            "sin_latitude": np.sin(lat_rad),
                            "cos_longitude": np.cos(lon_rad),
                            "sin_longitude": np.sin(lon_rad),
                        }
                        raw = _geo[name].astype(np.float32)
                    elif name in _lres_npz_constants:
                        # Load from pre-built forcings npz (z, lsm, slor, …).
                        raw = _lres_npz_constants[name]
                    elif name in OPTIONAL_ZERO_LRES_SFC_VARS:
                        raw = np.zeros(n_lres, dtype=np.float32)
                    else:
                        raise KeyError(f"Missing bundle field: {field_name}")
                else:
                    candidate = bundle[field_name]
                    if "level" in candidate.dims:
                        if name in _lres_npz_constants:
                            raw = _lres_npz_constants[name]
                        else:
                            raise KeyError(
                                f"Missing bundle field: {field_name} (found pressure-level field where single-level input was expected)"
                            )
                    else:
                        raw = candidate.values.astype(np.float32)
            else:
                field_name = f"in_lres_{base}"
                if field_name not in bundle:
                    raise KeyError(f"Missing bundle field: {field_name}")
                if level not in levels_bundle:
                    raw = _interpolate_level_points(bundle[field_name], sorted(levels_bundle), int(level))
                else:
                    raw = bundle[field_name].sel(level=int(level)).values.astype(np.float32)
            x_lres[:, idx] = np.asarray(raw, dtype=np.float32).reshape(-1)

        import torch  # pylint: disable=import-outside-toplevel

        x_hres_tensor = torch.zeros((1, 1, 1, n_hres, len(name_to_idx_hres)), dtype=torch.float32)
        fill_hres_features(
            x_hres_tensor,
            name_to_idx_hres,
            lat_hres,
            lon_hres,
            dt,
            "cpu",
            z=z,
            lsm=lsm,
            constant_forcings_npz=constant_forcings_npz,
        )
        x_hres[:, :] = x_hres_tensor[0, 0, 0].numpy()

        return x_lres, x_hres, lon_lres, lat_lres, lon_hres, lat_hres
    finally:
        if should_close:
            try:
                bundle.close()
            except Exception:
                pass


def fill_inputs_from_bundle(
    bundle_nc: str | Path,
    x_lres,
    x_hres,
    name_to_idx_lres: Mapping[str, int],
    name_to_idx_hres: Mapping[str, int],
    device,
    *,
    valid_time_override=None,
    constant_forcings_npz: str | Path | None = DEFAULT_CONSTANT_FORCINGS_NPZ,
) -> None:
    import torch  # pylint: disable=import-outside-toplevel
    x_lres_np, x_hres_np, _, _, _, _ = load_inputs_from_bundle_numpy(
        bundle_nc,
        name_to_idx_lres,
        name_to_idx_hres,
        valid_time_override=valid_time_override,
        constant_forcings_npz=constant_forcings_npz,
    )
    if int(x_lres_np.shape[0]) != int(x_lres.shape[3]):
        raise RuntimeError("LRES grid-size mismatch between bundle and model template")
    if int(x_hres_np.shape[0]) != int(x_hres.shape[3]):
        raise RuntimeError("HRES grid-size mismatch between bundle and model template")
    x_lres[0, 0, 0, :, :] = torch.from_numpy(x_lres_np).to(device)
    x_hres[0, 0, 0, :, :] = torch.from_numpy(x_hres_np).to(device)


def extract_target_from_bundle_dataset(
    bundle: xr.Dataset,
    weather_states: Sequence[str],
) -> tuple[np.ndarray | None, int]:
    """Return optional high-res target truth from bundle as [point_hres, weather_state].

    Supported variable naming in bundle:
    - point fields: `target_hres_<name>` (also `out_hres_<name>`, `y_hres_<name>`)
    - level fields: `target_hres_<base>` with level-like coord (`target_level`/`level`/`isobaricInhPa`)
      where weather state names use `<base>_<level>` (e.g. `t_850`).
    """
    n_points = int(bundle.sizes["point_hres"])
    y = np.full((n_points, len(weather_states)), np.nan, dtype=np.float32)
    found_channels = 0
    prefixes = ("target_hres_", "out_hres_", "y_hres_")

    for i, name in enumerate(weather_states):
        base, level = split_level_channel(name)
        candidates: list[str] = []
        for pref in prefixes:
            candidates.append(f"{pref}{name}")
            candidates.append(f"{pref}{base}")

        channel = None
        for var_name in candidates:
            if var_name not in bundle:
                continue
            da = bundle[var_name]
            if level is not None:
                selected = False
                for lev_name in ("target_level", "level", "isobaricInhPa"):
                    if lev_name in da.coords:
                        levels = np.asarray(da[lev_name].values).astype(int).tolist()
                        if int(level) not in levels:
                            break
                        da = da.sel({lev_name: int(level)})
                        selected = True
                        break
                if not selected and any(n in da.coords for n in ("target_level", "level", "isobaricInhPa")):
                    continue

            ok = True
            for dim in list(da.dims):
                if dim == "point_hres":
                    continue
                if da.sizes[dim] != 1:
                    ok = False
                    break
                da = da.isel({dim: 0})
            if not ok:
                continue

            vals = np.asarray(da.values, dtype=np.float32).reshape(-1)
            if vals.size != n_points:
                continue
            channel = vals
            break

        if channel is None:
            continue
        y[:, i] = channel
        found_channels += 1

    if found_channels == 0:
        return None, 0
    return y, found_channels


def extract_target_from_bundle(
    bundle_nc: str | Path | xr.Dataset,
    weather_states: Sequence[str],
) -> tuple[np.ndarray | None, int]:
    bundle, should_close = _borrow_or_open_bundle_dataset(bundle_nc)
    try:
        return extract_target_from_bundle_dataset(bundle, weather_states)
    finally:
        if should_close:
            try:
                bundle.close()
            except Exception:
                pass


def _to_1d_points(da: xr.DataArray) -> np.ndarray:
    da = da.copy()
    for dim in list(da.dims):
        if dim in ("values", "latitude", "longitude", "point_lres", "point_hres"):
            continue
        if da.sizes[dim] != 1:
            raise ValueError(f"Unexpected non-singleton dim {dim}={da.sizes[dim]}")
        da = da.isel({dim: 0})
    return np.asarray(da.values, dtype=np.float32).reshape(-1)


def _to_2d_level_points(da: xr.DataArray) -> np.ndarray:
    da = da.copy()
    keep = {"isobaricInhPa", "values", "level"}
    for d in list(da.dims):
        if d not in keep:
            if da.sizes[d] != 1:
                raise ValueError(
                    f"Unexpected non-singleton dim for 2D field: {d}={da.sizes[d]} in {da.name}"
                )
            da = da.isel({d: 0})
    if set(da.dims) == {"isobaricInhPa", "values"}:
        da = da.transpose("isobaricInhPa", "values")
    elif set(da.dims) == {"level", "values"}:
        da = da.transpose("level", "values")
    else:
        raise ValueError(f"Unexpected dims for PL field {da.name}: {da.dims}")
    return np.asarray(da.values, dtype=np.float32)


def _select_step(ds: xr.Dataset, step_hours: int | None) -> xr.Dataset:
    if step_hours is None:
        return ds
    if "step" not in ds.dims and "step" not in ds.coords:
        return ds
    target = np.timedelta64(int(step_hours), "h")
    try:
        return ds.sel(step=target)
    except Exception as exc:
        raise ValueError(f"Failed to select step={step_hours}h in dataset") from exc


def _select_member(
    ds: xr.Dataset, member: int | None, *, allow_missing: bool = False
) -> xr.Dataset:
    if member is None:
        return ds
    if "number" in ds.coords:
        try:
            return ds.sel(number=int(member))
        except KeyError:
            if allow_missing and ds.sizes.get("number", 0) == 1:
                return ds.isel(number=0)
            raise
    if "number" in ds.dims:
        try:
            return ds.sel(number=int(member))
        except KeyError:
            if allow_missing and ds.sizes.get("number", 0) == 1:
                return ds.isel(number=0)
            raise
    if "ensemble_member" in ds.coords:
        try:
            return ds.sel(ensemble_member=int(member))
        except KeyError:
            if allow_missing and ds.sizes.get("ensemble_member", 0) == 1:
                return ds.isel(ensemble_member=0)
            raise
    if "ensemble_member" in ds.dims:
        try:
            return ds.sel(ensemble_member=int(member))
        except KeyError:
            if allow_missing and ds.sizes.get("ensemble_member", 0) == 1:
                return ds.isel(ensemble_member=0)
            raise
    return ds


def build_input_bundle_from_grib(
    *,
    lres_sfc_grib: str | Path,
    lres_pl_grib: str | Path,
    hres_grib: str | Path,
    out_nc: str | Path,
    step_hours: int | None = None,
    member: int | None = None,
    out_zarr: str | Path | None = None,
    target_sfc_grib: str | Path | None = None,
    target_pl_grib: str | Path | None = None,
    require_target_fields: bool = True,
) -> Path:
    ds_sfc = _open_cfgrib_dataset(lres_sfc_grib)
    ds_pl = _open_cfgrib_dataset(lres_pl_grib)
    ds_hres = _open_cfgrib_dataset(hres_grib)

    ds_sfc = _select_step(ds_sfc, step_hours)
    ds_pl = _select_step(ds_pl, step_hours)
    ds_hres = _select_step(ds_hres, step_hours)

    ds_sfc = _select_member(ds_sfc, member)
    ds_pl = _select_member(ds_pl, member)
    ds_hres = _select_member(ds_hres, member, allow_missing=True)

    lat_lres = _to_1d_points(ds_sfc["latitude"])
    lon_lres = _to_1d_points(ds_sfc["longitude"])
    lat_hres = _to_1d_points(ds_hres["latitude"])
    lon_hres = _to_1d_points(ds_hres["longitude"])

    level_coord = _get_pl_level_coord(ds_pl)
    levels = np.asarray(ds_pl[level_coord].values, dtype=np.int32).reshape(-1)

    coords = {
        "point_lres": np.arange(lat_lres.shape[0], dtype=np.int32),
        "point_hres": np.arange(lat_hres.shape[0], dtype=np.int32),
        "level": levels,
        "lat_lres": ("point_lres", lat_lres),
        "lon_lres": ("point_lres", lon_lres),
        "lat_hres": ("point_hres", lat_hres),
        "lon_hres": ("point_hres", lon_hres),
    }

    data_vars = {}

    sfc_map = {
        "u10": "in_lres_10u",
        "v10": "in_lres_10v",
        "d2m": "in_lres_2d",
        "t2m": "in_lres_2t",
        "cp": "in_lres_cp",
        "hcc": "in_lres_hcc",
        "lcc": "in_lres_lcc",
        "mcc": "in_lres_mcc",
        "msl": "in_lres_msl",
        "skt": "in_lres_skt",
        "sp": "in_lres_sp",
        "ssrd": "in_lres_ssrd",
        "strd": "in_lres_strd",
        "tcc": "in_lres_tcc",
        "tcw": "in_lres_tcw",
        "tp": "in_lres_tp",
    }
    for src, dst in sfc_map.items():
        if src not in ds_sfc:
            if src in OPTIONAL_ZERO_LRES_SFC_VARS:
                data_vars[dst] = ("point_lres", np.zeros(lat_lres.shape[0], dtype=np.float32))
                continue
            raise KeyError(f"Missing SFC variable in input: {src}")
        data_vars[dst] = ("point_lres", _to_1d_points(ds_sfc[src]))

    pl_vars = ["q", "t", "u", "v", "w", "z"]
    for v in pl_vars:
        if v not in ds_pl:
            raise KeyError(f"Missing PL variable in input: {v}")
        data_vars[f"in_lres_{v}"] = (("level", "point_lres"), _to_2d_level_points(ds_pl[v]))

    hres_map = {"z": "in_hres_z", "lsm": "in_hres_lsm"}
    for src, dst in hres_map.items():
        if src not in ds_hres:
            continue
        data_vars[dst] = ("point_hres", _to_1d_points(ds_hres[src]))

    auto_sfc, auto_pl = (None, None)
    if require_target_fields and target_sfc_grib is None and target_pl_grib is None:
        auto_sfc, auto_pl = _infer_target_gribs_from_hres(hres_grib)
    target_sfc_grib = target_sfc_grib or auto_sfc
    target_pl_grib = target_pl_grib or auto_pl

    target_level_coord: np.ndarray | None = None
    if target_sfc_grib:
        ds_target_sfc = _open_cfgrib_dataset(target_sfc_grib)
        ds_target_sfc = _select_step(ds_target_sfc, step_hours)
        ds_target_sfc = _select_member(ds_target_sfc, member, allow_missing=True)
        target_map_sfc = {
            "u10": "target_hres_10u",
            "v10": "target_hres_10v",
            "d2m": "target_hres_2d",
            "t2m": "target_hres_2t",
            "msl": "target_hres_msl",
            "skt": "target_hres_skt",
            "sp": "target_hres_sp",
            "tcw": "target_hres_tcw",
        }
        for src, dst in target_map_sfc.items():
            if src not in ds_target_sfc:
                continue
            vals = _to_1d_points(ds_target_sfc[src])
            if vals.size != lat_hres.size:
                raise ValueError(
                    f"Target SFC field {src} point count {vals.size} != point_hres {lat_hres.size}"
                )
            data_vars[dst] = ("point_hres", vals)

    if target_pl_grib:
        ds_target_pl = _open_cfgrib_dataset(target_pl_grib)
        ds_target_pl = _select_step(ds_target_pl, step_hours)
        ds_target_pl = _select_member(ds_target_pl, member, allow_missing=True)
        level_coord_target = _get_pl_level_coord(ds_target_pl)
        target_levels = np.asarray(ds_target_pl[level_coord_target].values, dtype=np.int32).reshape(-1)
        target_level_coord = target_levels
        for var in ("q", "t", "u", "v", "w", "z"):
            if var not in ds_target_pl:
                continue
            vals = _to_2d_level_points(ds_target_pl[var])
            if vals.shape[1] != lat_hres.size:
                raise ValueError(
                    f"Target PL field {var} point count {vals.shape[1]} != point_hres {lat_hres.size}"
                )
            data_vars[f"target_hres_{var}"] = (("target_level", "point_hres"), vals)

    has_target = any(name.startswith("target_hres_") for name in data_vars)
    if require_target_fields and not has_target:
        raise ValueError(
            "No target_hres_* fields were added to bundle. "
            "Provide --target-sfc-grib/--target-pl-grib or place matching target *_y.grib files near hres input."
        )

    bundle = xr.Dataset(data_vars=data_vars, coords=coords)
    if target_level_coord is not None:
        bundle = bundle.assign_coords(target_level=target_level_coord)
    valid_time = (
        str(np.asarray(ds_sfc["valid_time"].values).squeeze())
        if "valid_time" in ds_sfc
        else "unknown"
    )
    bundle.attrs["case_valid_time"] = valid_time
    bundle.attrs["source_lres_sfc"] = str(lres_sfc_grib)
    bundle.attrs["source_lres_pl"] = str(lres_pl_grib)
    bundle.attrs["source_hres"] = str(hres_grib)
    if target_sfc_grib:
        bundle.attrs["source_target_sfc"] = str(target_sfc_grib)
    if target_pl_grib:
        bundle.attrs["source_target_pl"] = str(target_pl_grib)
    bundle.attrs["has_target_hres_fields"] = "yes" if has_target else "no"
    if has_target:
        bundle.attrs["description"] = (
            "Combined low-res + high-res feature inputs for local inference. "
            "Includes target_hres_* fields for truth-aware evaluation."
        )
    else:
        bundle.attrs["description"] = (
            "Combined low-res + high-res feature inputs for local inference. "
            "Created without target_hres_* fields because --allow-missing-target-unsafe was used. "
            "Prediction-only and non-canonical for truth-aware evaluation."
        )
        bundle.attrs["missing_target_policy"] = "bundle_without_target_hres_due_to_allow_missing_target_unsafe"
    if step_hours is not None:
        bundle.attrs["step_hours"] = int(step_hours)
    if member is not None:
        bundle.attrs["member"] = int(member)

    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    try:
        bundle.to_netcdf(out_nc)
    except Exception:
        bundle.to_netcdf(out_nc, engine="scipy")
    if out_zarr:
        out_zarr = Path(out_zarr)
        out_zarr.parent.mkdir(parents=True, exist_ok=True)
        bundle.to_zarr(out_zarr, mode="w", consolidated=True)
    return out_nc


def _default_bundle_name(lres_sfc_grib: str | Path) -> str:
    name = Path(lres_sfc_grib).name
    if name.endswith("_sfc.grib"):
        name = name[: -len("_sfc.grib")]
    elif name.endswith(".grib"):
        name = name[: -len(".grib")]
    return f"{name}_input_bundle.nc"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build input bundle NetCDF from GRIB.")
    parser.add_argument("--lres-sfc-grib", required=True)
    parser.add_argument("--lres-pl-grib", required=True)
    parser.add_argument("--hres-grib", required=True)
    parser.add_argument(
        "--target-sfc-grib",
        default="",
        help="Optional high-res surface GRIB containing target/truth fields to store in bundle.",
    )
    parser.add_argument(
        "--target-pl-grib",
        default="",
        help="Optional high-res pressure-level GRIB containing target/truth fields to store in bundle.",
    )
    parser.add_argument(
        "--allow-missing-target",
        action="store_true",
        help="Deprecated alias. Use --allow-missing-target-unsafe instead.",
    )
    parser.add_argument(
        "--allow-missing-target-unsafe",
        action="store_true",
        help=(
            "Explicitly allow creating bundle without target_hres_* fields. "
            "Unsafe: output is prediction-only and non-canonical for truth-aware evaluation."
        ),
    )
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--out-zarr",
        default="",
        help="Optional Zarr output for faster chunked reads.",
    )
    parser.add_argument(
        "--out-root",
        default="/home/ecm5702/hpcperm/data/input_data",
        help="Base output folder for input data.",
    )
    parser.add_argument("--grid", default="", help="Resolution tag (e.g. o320, o96).")
    parser.add_argument(
        "--step-hours",
        type=int,
        default=None,
        help="Select a single step (hours) from multi-step GRIBs.",
    )
    parser.add_argument(
        "--member",
        type=int,
        default=None,
        help="Select a single member if GRIBs contain an ensemble dimension.",
    )
    args = parser.parse_args()
    if args.allow_missing_target:
        raise SystemExit(
            "--allow-missing-target is deprecated. "
            "Use --allow-missing-target-unsafe for an explicit prediction-only escape hatch."
        )

    out_path = args.out
    if not out_path:
        grid = args.grid or "grid"
        out_dir = Path(args.out_root) / grid
        out_path = out_dir / _default_bundle_name(args.lres_sfc_grib)

    out = build_input_bundle_from_grib(
        lres_sfc_grib=args.lres_sfc_grib,
        lres_pl_grib=args.lres_pl_grib,
        hres_grib=args.hres_grib,
        out_nc=out_path,
        step_hours=args.step_hours,
        member=args.member,
        out_zarr=args.out_zarr or None,
        target_sfc_grib=args.target_sfc_grib or None,
        target_pl_grib=args.target_pl_grib or None,
        require_target_fields=not args.allow_missing_target_unsafe,
    )
    print(f"Saved bundle: {out}")


if __name__ == "__main__":
    main()
