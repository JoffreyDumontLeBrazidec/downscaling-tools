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

    if groups["sfc"]["params"]:
        requests.append({
            "class": output_mars["class"],
            "stream": output_mars["stream"],
            "type": output_mars["type"],
            "expver": expver,
            "date": date,
            "time": "0000",
            "step": step,
            "number": number_str,
            "levtype": "sfc",
            "param": groups["sfc"]["params"],
        })

    if groups["pl"]["params"]:
        requests.append({
            "class": output_mars["class"],
            "stream": output_mars["stream"],
            "type": output_mars["type"],
            "expver": expver,
            "date": date,
            "time": "0000",
            "step": step,
            "number": number_str,
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
    truth_root: str | Path,
    output_dir: Path,
) -> Path:
    """Retrieve predictions from MARS, load truth, assemble and write predictions_*.nc.

    Returns path to written file.
    """
    import xarray as xr

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"predictions_{date}_step{int(step):03d}.nc"

    LOG.info("Retrieving predictions for date=%s step=%d", date, step)
    ds_raw = retrieve_predictions_from_mars(
        expver=expver, date=date, step=step, members=members,
        output_mars=output_mars, weather_states=weather_states,
    )
    ds_pred = _reshape_to_prediction_format(ds_raw, weather_states)

    truth_root = Path(truth_root)
    ds_truth = _load_truth_reference(truth_root, date, step, weather_states)

    if ds_truth is not None:
        ds_out = xr.merge([ds_pred[["y_pred", "lon_hres", "lat_hres"]], ds_truth])
    else:
        ds_out = ds_pred[["y_pred", "lon_hres", "lat_hres"]]
        LOG.warning("No truth reference found for date=%s step=%d", date, step)

    if out_path.exists():
        out_path.unlink()
    ds_out.to_netcdf(out_path, mode="w")
    LOG.info("Wrote %s", out_path)
    return out_path


def _load_truth_reference(
    truth_root: Path,
    date: str,
    step: int,
    weather_states: list[str],
) -> "xr.Dataset | None":
    """Load truth (y) and input (x) from reference GRIBs under truth_root.

    Discovers available GRIB files under truth_root/grib/ and loads them.
    Returns None if no matching GRIB files are found.
    """
    import earthkit.data as ekd
    import xarray as xr

    grib_dir = truth_root / "grib"
    if not grib_dir.exists():
        LOG.warning("Truth GRIB dir not found: %s", grib_dir)
        return None

    candidates = sorted(grib_dir.glob(f"enfo_reference_*{date}*.grib"))
    if not candidates:
        candidates = sorted(grib_dir.glob("enfo_reference_*early-august*.grib"))
    if not candidates:
        candidates = sorted(grib_dir.glob("*.grib"))

    if not candidates:
        return None

    grib_path = candidates[0]
    LOG.info("Loading truth reference from %s", grib_path)
    try:
        ds_ref = ekd.from_source("file", str(grib_path)).to_xarray(squeeze=False)
    except Exception:
        LOG.warning("Failed to load truth GRIB: %s", grib_path, exc_info=True)
        return None

    return ds_ref
