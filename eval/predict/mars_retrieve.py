"""MARS retrieval and predictions_*.nc assembly for PrepML output.

Retrieves PrepML predictions from MARS/FDB using an expver identifier,
loads truth/input from reference GRIBs, and assembles them into the
same predictions_YYYYMMDD_stepNNN.nc format as manual inference.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)

_SURFACE_PARAMS = {
    "10u", "10v", "2d", "2t", "cp", "hcc", "lcc", "mcc",
    "msl", "skt", "sp", "ssrd", "strd", "tcc", "tcw", "tp",
}

_PL_RE = re.compile(r"^([a-z]+)_(\d+)$")


def weather_state_to_mars(state: str) -> dict[str, Any]:
    """Map a weather state name to MARS request parameters.

    Surface states (e.g. '2t', '10u') -> {'param': '2t', 'levtype': 'sfc'}
    Pressure-level states (e.g. 'z_500') -> {'param': 'z', 'levtype': 'pl', 'level': 500}
    """
    if state in _SURFACE_PARAMS:
        return {"param": state, "levtype": "sfc"}

    m = _PL_RE.match(state)
    if m:
        return {"param": m.group(1), "levtype": "pl", "level": int(m.group(2))}

    raise ValueError(
        f"Unknown weather state '{state}'. "
        f"Expected a surface param ({sorted(_SURFACE_PARAMS)}) "
        f"or a pressure-level param like 'z_500', 't_850'."
    )


def group_weather_states_for_mars(
    states: list[str],
) -> dict[str, dict[str, list]]:
    """Group weather states into MARS request groups by levtype.

    Returns:
        {"sfc": {"params": [...]}, "pl": {"params": [...], "levels": [...]}}
    """
    groups: dict[str, dict[str, list]] = {
        "sfc": {"params": []},
        "pl": {"params": [], "levels": []},
    }
    seen_pl: set[str] = set()
    for state in states:
        mapped = weather_state_to_mars(state)
        levtype = mapped["levtype"]
        if levtype == "sfc":
            groups["sfc"]["params"].append(mapped["param"])
        else:
            param = mapped["param"]
            if param not in seen_pl:
                groups["pl"]["params"].append(param)
                seen_pl.add(param)
            level = mapped["level"]
            if level not in groups["pl"]["levels"]:
                groups["pl"]["levels"].append(level)
    return groups


def build_prediction_request(
    *,
    expver: str,
    date: str,
    step: int,
    members: list[int],
    output_mars: dict[str, str],
    weather_states: list[str],
) -> list[dict[str, Any]]:
    """Build MARS RETRIEVE request(s) for PrepML predictions.

    Returns a list of request dicts (one per levtype present in weather_states).
    """
    groups = group_weather_states_for_mars(weather_states)
    number_str = "/".join(str(m) for m in members)

    requests: list[dict[str, Any]] = []

    # Base request fields. Include "domain": "g" to force grid-point output
    # and avoid earthkit's spectral→grid expansion which uses enormous memory.
    base = {
        "class": output_mars["class"],
        "stream": output_mars["stream"],
        "type": output_mars["type"],
        "expver": expver,
        "date": date,
        "time": "0000",
        "step": step,
        "number": number_str,
        "domain": "g",
    }

    if groups["sfc"]["params"]:
        requests.append({
            **base,
            "levtype": "sfc",
            "param": groups["sfc"]["params"],
        })

    if groups["pl"]["params"]:
        requests.append({
            **base,
            "levtype": "pl",
            "param": groups["pl"]["params"],
            "levelist": sorted(groups["pl"]["levels"]),
        })

    return requests


def retrieve_predictions_from_mars(
    *,
    expver: str,
    date: str,
    step: int,
    members: list[int],
    output_mars: dict[str, str],
    weather_states: list[str],
) -> "xr.Dataset":
    """Retrieve PrepML predictions from MARS and return as xarray Dataset.

    Requires earthkit.data and MARS access (AC login nodes only).
    """
    import earthkit.data as ekd
    import xarray as xr

    requests = build_prediction_request(
        expver=expver, date=date, step=step, members=members,
        output_mars=output_mars, weather_states=weather_states,
    )
    datasets = []
    for req in requests:
        LOG.info("MARS retrieve: expver=%s date=%s step=%d levtype=%s", expver, date, step, req["levtype"])
        ds = ekd.from_source("mars", req).to_xarray(squeeze=False)
        if "level_type" in ds.dims:
            ds = ds.squeeze(dim="level_type", drop=True)
        datasets.append(ds)

    return xr.merge(datasets)


def _reshape_to_prediction_format(
    ds_pred: "xr.Dataset",
    weather_states: list[str],
) -> "xr.Dataset":
    """Reshape raw MARS xarray dataset to predictions_*.nc format.

    Produces y_pred array with (ensemble_member, weather_state, grid_point_hres) dims.
    """
    import pandas as pd
    import xarray as xr

    if "values" in ds_pred.dims:
        ds_pred = ds_pred.rename({"values": "grid_point_hres"})
    grid_points = range(ds_pred.sizes["grid_point_hres"])
    ds_pred = ds_pred.assign_coords(grid_point_hres=grid_points)

    if "number" in ds_pred.dims:
        ds_pred = ds_pred.rename({"number": "ensemble_member"})

    if "latitude" in ds_pred:
        ds_pred = ds_pred.rename({"latitude": "lat_hres", "longitude": "lon_hres"})
    ds_pred["lon_hres"] = ((ds_pred.lon_hres + 180) % 360) - 180

    state_arrays = []
    for state in weather_states:
        mapped = weather_state_to_mars(state)
        param = mapped["param"]
        if mapped["levtype"] == "pl":
            level = mapped["level"]
            arr = ds_pred[param].sel(level=level).drop_vars("level")
        else:
            arr = ds_pred[param]
        state_arrays.append(arr)

    ds_pred["y_pred"] = xr.concat(
        state_arrays,
        dim=pd.Index(weather_states, name="weather_state"),
    )
    ds_pred.y_pred.attrs["lon"] = "lon_hres"
    ds_pred.y_pred.attrs["lat"] = "lat_hres"

    if "step" in ds_pred.coords:
        ds_pred = ds_pred.assign_coords(
            step=ds_pred.step.values.astype("timedelta64[h]").astype(int)
        )

    for var in ds_pred.variables.values():
        if "_earthkit" in var.attrs:
            del var.attrs["_earthkit"]

    return ds_pred


def assemble_predictions_file(
    *,
    expver: str,
    date: str,
    step: int,
    members: list[int],
    output_mars: dict[str, str],
    weather_states: list[str],
    bundle_dir: str | Path,
    bundle_filename_tpl: str,
    output_dir: Path,
) -> Path:
    """Retrieve predictions from MARS, load truth+input from bundles, write predictions_*.nc.

    Args:
        bundle_dir: directory containing input bundle .nc files (predict.input_root)
        bundle_filename_tpl: bundle filename template with {date}, {member:02d}, {step:03d} placeholders

    Returns path to written file.
    """
    import pandas as pd
    import xarray as xr

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"predictions_{date}_step{int(step):03d}.nc"

    # 1. Retrieve predictions from MARS and extract y_pred
    LOG.info("Retrieving predictions for date=%s step=%d", date, step)
    ds_raw = retrieve_predictions_from_mars(
        expver=expver, date=date, step=step, members=members,
        output_mars=output_mars, weather_states=weather_states,
    )
    ds_pred = _reshape_to_prediction_format(ds_raw, weather_states)
    # Extract only what we need, then free the raw MARS data
    y_pred_ds = ds_pred[["y_pred", "lon_hres", "lat_hres"]].load()
    del ds_raw, ds_pred

    # 2. Load truth (y) and input (x) from bundle
    bundle_dir = Path(bundle_dir)
    ds_bundle = _load_bundle_truth_and_input(
        bundle_dir, bundle_filename_tpl, date, step, weather_states,
    )

    # 3. Merge y_pred + y + x + coords
    # Keep y_pred's coords (lon_hres, lat_hres already wrapped to -180..180)
    # and drop duplicate coords from bundle to avoid merge conflicts.
    if ds_bundle is not None:
        bundle_vars = {k: ds_bundle[k] for k in ds_bundle.data_vars
                       if k not in y_pred_ds and k not in y_pred_ds.coords}
        ds_out = y_pred_ds.assign(bundle_vars)
    else:
        ds_out = y_pred_ds
        LOG.warning("No bundle truth found for date=%s step=%d — writing y_pred only", date, step)

    # 4. Write
    if out_path.exists():
        out_path.unlink()
    ds_out.to_netcdf(out_path, mode="w")
    LOG.info("Wrote %s", out_path)
    return out_path


def _extract_state_from_bundle(
    ds: "xr.Dataset",
    prefix: str,
    state: str,
    grid_dim: str,
    level_dim: str | None = None,
) -> "np.ndarray":
    """Extract a single weather state from bundle variables.

    Maps weather state names to bundle variable names:
      '2t' -> f'{prefix}_2t' (surface, shape: (grid_points,))
      'z_500' -> f'{prefix}_z' at level=500 (PL, shape: (grid_points,))
    """
    mapped = weather_state_to_mars(state)
    param = mapped["param"]
    var_name = f"{prefix}_{param}"

    if var_name not in ds:
        raise KeyError(f"Bundle variable '{var_name}' not found for state '{state}'")

    arr = ds[var_name]
    if mapped["levtype"] == "pl":
        level = mapped["level"]
        if level_dim and level_dim in arr.dims:
            arr = arr.sel({level_dim: level})
        else:
            raise KeyError(
                f"PL variable '{var_name}' has no '{level_dim}' dim for level={level}"
            )

    return arr.values


def _load_bundle_truth_and_input(
    bundle_dir: Path,
    bundle_filename_tpl: str,
    date: str,
    step: int,
    weather_states: list[str],
    member: int = 1,
) -> "xr.Dataset | None":
    """Load truth (y) and input (x) from an input bundle file.

    Uses member 1 by default — truth is the same across ensemble members.
    Returns a dataset with 'y', 'x', coordinate arrays, or None if bundle not found.
    """
    import pandas as pd
    import xarray as xr

    bundle_name = bundle_filename_tpl.format(date=date, member=member, step=step)
    bundle_path = bundle_dir / bundle_name
    if not bundle_path.exists():
        LOG.warning("Bundle not found: %s", bundle_path)
        return None

    LOG.info("Loading truth+input from bundle: %s", bundle_path)
    # Use netCDF4 directly instead of xarray to avoid OOM — xarray's
    # open_dataset loads all metadata/chunks greedily even with drop_variables.
    import netCDF4 as nc
    import pandas as pd
    import xarray as xr

    try:
        ds = nc.Dataset(str(bundle_path), "r")
    except Exception:
        LOG.warning("Failed to open bundle: %s", bundle_path, exc_info=True)
        return None

    try:
        # Build y (truth) from target_hres_* variables
        y_arrays = []
        for state in weather_states:
            mapped = weather_state_to_mars(state)
            param = mapped["param"]
            var_name = f"target_hres_{param}"
            if var_name not in ds.variables:
                LOG.warning("Bundle missing truth variable %s", var_name)
                return None
            arr = ds[var_name][:]
            if mapped["levtype"] == "pl":
                # Select the target level from the target_level dimension
                level = mapped["level"]
                target_levels = ds["target_level"][:] if "target_level" in ds.variables else None
                if target_levels is not None:
                    level_idx = int(np.where(target_levels == level)[0][0])
                    arr = arr[level_idx]
                else:
                    LOG.warning("No target_level dim for PL variable %s", var_name)
                    return None
            y_arrays.append(arr)

        y = np.stack(y_arrays, axis=0)  # (weather_state, grid_point_hres)

        # Build x (input) from in_lres_* variables
        x_arrays = []
        for state in weather_states:
            mapped = weather_state_to_mars(state)
            param = mapped["param"]
            var_name = f"in_lres_{param}"
            if var_name not in ds.variables:
                LOG.debug("Input variable %s not in bundle, skipping x", var_name)
                x_arrays = []
                break
            arr = ds[var_name][:]
            if mapped["levtype"] == "pl":
                level = mapped["level"]
                levels = ds["level"][:] if "level" in ds.variables else None
                if levels is not None:
                    level_idx = int(np.where(levels == level)[0][0])
                    arr = arr[level_idx]
                else:
                    x_arrays = []
                    break
            x_arrays.append(arr)

        # Load coordinate arrays
        coords_data: dict[str, np.ndarray] = {}
        for coord in ("lat_hres", "lon_hres", "lat_lres", "lon_lres"):
            if coord in ds.variables:
                coords_data[coord] = ds[coord][:]

    except Exception:
        LOG.warning("Failed to extract truth/input from bundle: %s", bundle_path, exc_info=True)
        return None
    finally:
        ds.close()

    # Build xarray dataset from numpy arrays
    result_vars: dict[str, Any] = {}

    result_vars["y"] = xr.DataArray(
        y,
        dims=["weather_state", "grid_point_hres"],
        coords={
            "weather_state": pd.Index(weather_states, name="weather_state"),
            "grid_point_hres": np.arange(y.shape[1]),
        },
    )
    result_vars["y"].attrs["lon"] = "lon_hres"
    result_vars["y"].attrs["lat"] = "lat_hres"

    if x_arrays:
        x = np.stack(x_arrays, axis=0)
        result_vars["x"] = xr.DataArray(
            x,
            dims=["weather_state", "grid_point_lres"],
            coords={
                "weather_state": pd.Index(weather_states, name="weather_state"),
                "grid_point_lres": np.arange(x.shape[1]),
            },
        )
        result_vars["x"].attrs["lon"] = "lon_lres"
        result_vars["x"].attrs["lat"] = "lat_lres"

    for coord_name, coord_arr in coords_data.items():
        dim = "grid_point_hres" if "hres" in coord_name else "grid_point_lres"
        result_vars[coord_name] = xr.DataArray(coord_arr, dims=[dim])

    return xr.Dataset(result_vars)
