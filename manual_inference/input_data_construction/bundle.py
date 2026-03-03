from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import xarray as xr

DEFAULT_CONSTANT_FORCINGS_NPZ = "/home/ecm5702/hpcperm/data/o320-forcings.npz"

SFC_TO_CFGRIB = {
    "10u": "u10",
    "10v": "v10",
    "2d": "d2m",
    "2t": "t2m",
    "msl": "msl",
    "skt": "skt",
    "sp": "sp",
    "tcw": "tcw",
}


def split_level_channel(name: str) -> tuple[str, int | None]:
    match = re.match(r"^(.*)_([0-9]+)$", name)
    if match is None:
        return name, None
    return match.group(1), int(match.group(2))


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


def load_lres_fields_from_grib(
    sfc_grib: str | Path,
    pl_grib: str | Path,
    lres_channel_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    import earthkit.data as ekd  # pylint: disable=import-outside-toplevel

    ds_sfc = ekd.from_source("file", str(sfc_grib)).to_xarray(engine="cfgrib")
    ds_pl = ekd.from_source("file", str(pl_grib)).to_xarray(engine="cfgrib")
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

    ds_hr = ekd.from_source("file", str(hres_grib)).to_xarray(engine="cfgrib")
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
    if constant_forcings_npz is not None:
        npz_path = Path(constant_forcings_npz)
        if npz_path.exists():
            with np.load(npz_path) as npz:
                for key in npz.files:
                    arr = np.asarray(npz[key], dtype=np.float32).reshape(-1)
                    if arr.size == lat_hres.size:
                        constant_values[key] = arr

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
        else:
            raise KeyError(
                f"Missing required high-res constant forcing '{name}'. "
                f"Provide it in {constant_forcings_npz!s} or the bundle."
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

    bundle = _open_bundle_dataset(bundle_nc)
    if int(bundle.sizes["point_lres"]) != int(x_lres.shape[3]):
        raise RuntimeError("LRES grid-size mismatch between bundle and model template")
    if int(bundle.sizes["point_hres"]) != int(x_hres.shape[3]):
        raise RuntimeError("HRES grid-size mismatch between bundle and model template")

    levels_bundle = (
        set(int(v) for v in np.asarray(bundle["level"].values).tolist())
        if "level" in bundle
        else set()
    )
    for name, idx in name_to_idx_lres.items():
        base, level = split_level_channel(name)
        if level is None:
            field_name = f"in_lres_{name}"
            if field_name not in bundle:
                raise KeyError(f"Missing bundle field: {field_name}")
            raw = bundle[field_name].values.astype(np.float32)
        else:
            field_name = f"in_lres_{base}"
            if field_name not in bundle:
                raise KeyError(f"Missing bundle field: {field_name}")
            if level not in levels_bundle:
                raise KeyError(f"Missing level {level} for {base}")
            raw = bundle[field_name].sel(level=int(level)).values.astype(np.float32)
        x_lres[0, 0, 0, :, idx] = torch.from_numpy(raw).to(device)

    lat_hres = bundle["lat_hres"].values.astype(np.float32)
    lon_hres = bundle["lon_hres"].values.astype(np.float32)
    z = bundle["in_hres_z"].values.astype(np.float32) if "in_hres_z" in bundle else None
    lsm = (
        bundle["in_hres_lsm"].values.astype(np.float32) if "in_hres_lsm" in bundle else None
    )
    dt = parse_valid_time(valid_time_override, bundle.attrs.get("case_valid_time"))
    fill_hres_features(
        x_hres,
        name_to_idx_hres,
        lat_hres,
        lon_hres,
        dt,
        device,
        z=z,
        lsm=lsm,
        constant_forcings_npz=constant_forcings_npz,
    )


def extract_target_from_bundle(
    bundle_nc: str | Path,
    weather_states: Sequence[str],
) -> tuple[np.ndarray | None, int]:
    """Return optional high-res target truth from bundle as [point_hres, weather_state].

    Supported variable naming in bundle:
    - point fields: `target_hres_<name>` (also `out_hres_<name>`, `y_hres_<name>`)
    - level fields: `target_hres_<base>` with level-like coord (`target_level`/`level`/`isobaricInhPa`)
      where weather state names use `<base>_<level>` (e.g. `t_850`).
    """
    bundle = _open_bundle_dataset(bundle_nc)
    try:
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
    finally:
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
) -> Path:
    import earthkit.data as ekd  # pylint: disable=import-outside-toplevel

    ds_sfc = ekd.from_source("file", str(lres_sfc_grib)).to_xarray(engine="cfgrib")
    ds_pl = ekd.from_source("file", str(lres_pl_grib)).to_xarray(engine="cfgrib")
    ds_hres = ekd.from_source("file", str(hres_grib)).to_xarray(engine="cfgrib")

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
        "msl": "in_lres_msl",
        "skt": "in_lres_skt",
        "sp": "in_lres_sp",
        "tcw": "in_lres_tcw",
    }
    for src, dst in sfc_map.items():
        if src not in ds_sfc:
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

    target_level_coord: np.ndarray | None = None
    if target_sfc_grib:
        ds_target_sfc = ekd.from_source("file", str(target_sfc_grib)).to_xarray(engine="cfgrib")
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
        ds_target_pl = ekd.from_source("file", str(target_pl_grib)).to_xarray(engine="cfgrib")
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
    bundle.attrs["description"] = (
        "Combined low-res + high-res feature inputs for local inference. "
        "Optional target_hres_* fields may be included for truth-aware evaluation."
    )
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
    )
    print(f"Saved bundle: {out}")


if __name__ == "__main__":
    main()
