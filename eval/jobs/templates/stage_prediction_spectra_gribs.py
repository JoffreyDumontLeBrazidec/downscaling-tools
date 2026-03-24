#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from eccodes import (
    codes_clone,
    codes_get,
    codes_get_array,
    codes_grib_new_from_file,
    codes_release,
    codes_set_values,
    codes_write,
)


PRED_RE = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")


@dataclass(frozen=True)
class VariableConfig:
    weather_state: str
    dir_name: str


VARIABLE_CONFIGS: dict[str, VariableConfig] = {
    "10u": VariableConfig(weather_state="10u", dir_name="10u_sfc"),
    "10v": VariableConfig(weather_state="10v", dir_name="10v_sfc"),
    "2t": VariableConfig(weather_state="2t", dir_name="2t_sfc"),
    "sp": VariableConfig(weather_state="sp", dir_name="sp_sfc"),
    "t_850": VariableConfig(weather_state="t_850", dir_name="t_850"),
    "z_500": VariableConfig(weather_state="z_500", dir_name="z_500"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage prediction NetCDF members into no-poles GRIB files that match the "
            "canonical ECMWF reference_spectra directory layout."
        )
    )
    p.add_argument("--predictions-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--template-root", default="/home/ecm5702/hpcperm/reference_spectra/enfo_o1280")
    p.add_argument("--weather-states", default="10u,10v,2t,sp,t_850,z_500")
    p.add_argument("--summary-path", default="")
    return p.parse_args()


def parse_weather_states(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def discover_prediction_files(pred_dir: Path) -> list[tuple[Path, int, int]]:
    out: list[tuple[Path, int, int]] = []
    for path in sorted(pred_dir.glob("predictions_*.nc")):
        match = PRED_RE.match(path.name)
        if not match:
            continue
        out.append((path, int(match.group(1)), int(match.group(2))))
    if not out:
        raise FileNotFoundError(f"No predictions_*.nc files found in {pred_dir}")
    return out


def first_template_path(template_root: Path, dir_name: str) -> Path:
    candidates = sorted((template_root / dir_name).glob("*_nopoles.grb"))
    if not candidates:
        raise FileNotFoundError(f"No *_nopoles.grb templates found in {template_root / dir_name}")
    return candidates[0]


def read_template_grid(template_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with template_path.open("rb") as handle:
        gid = codes_grib_new_from_file(handle)
        if gid is None:
            raise RuntimeError(f"Could not read template {template_path}")
        try:
            lat = np.asarray(codes_get_array(gid, "latitudes"), dtype=np.float64)
            lon = np.asarray(codes_get_array(gid, "longitudes"), dtype=np.float64)
            lat_first = float(codes_get(gid, "latitudeOfFirstGridPointInDegrees"))
        finally:
            codes_release(gid)
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon, lat_first


def write_values_from_template(template_path: Path, out_path: Path, values: np.ndarray) -> None:
    with template_path.open("rb") as handle:
        gid = codes_grib_new_from_file(handle)
        if gid is None:
            raise RuntimeError(f"Could not read template {template_path}")
        clone = codes_clone(gid)
        codes_release(gid)
    try:
        codes_set_values(clone, np.asarray(values, dtype=np.float64))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as handle:
            codes_write(clone, handle)
    finally:
        codes_release(clone)


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.predictions_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    template_root = Path(args.template_root).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_states = parse_weather_states(args.weather_states)
    unknown = [state for state in requested_states if state not in VARIABLE_CONFIGS]
    if unknown:
        raise ValueError(f"Unsupported weather states: {unknown}")

    pred_files = discover_prediction_files(pred_dir)
    template_paths = {
        state: first_template_path(template_root, VARIABLE_CONFIGS[state].dir_name)
        for state in requested_states
    }

    with xr.open_dataset(pred_files[0][0]) as ds0:
        pred_lat = np.asarray(ds0["lat_hres"].values, dtype=np.float64)
        pred_lon = np.asarray(ds0["lon_hres"].values, dtype=np.float64)
        pred_lon = ((pred_lon + 180.0) % 360.0) - 180.0
        ds0_states = [str(value) for value in ds0["weather_state"].values.tolist()]

    missing_in_predictions = [state for state in requested_states if state not in ds0_states]
    if missing_in_predictions:
        raise ValueError(f"Prediction files do not contain weather states: {missing_in_predictions}")

    ref_lat, ref_lon, lat_first = read_template_grid(template_paths[requested_states[0]])
    pole_mask = np.abs(pred_lat) <= lat_first + 0.001
    if int(pole_mask.sum()) != ref_lat.size:
        raise ValueError(
            f"Pole-mask count {int(pole_mask.sum())} does not match template count {ref_lat.size}"
        )

    masked_lat = pred_lat[pole_mask]
    masked_lon = pred_lon[pole_mask]
    max_abs_lat_diff = float(np.max(np.abs(masked_lat - ref_lat)))
    max_abs_lon_diff = float(np.max(np.abs(masked_lon - ref_lon)))
    if not np.allclose(masked_lat, ref_lat, atol=1e-5):
        raise ValueError(f"Prediction latitudes do not align with template: max_abs_diff={max_abs_lat_diff}")
    if max_abs_lon_diff > 1e-3:
        raise ValueError(f"Prediction longitudes do not align with template: max_abs_diff={max_abs_lon_diff}")

    grib_files_written = 0
    dates: list[int] = []
    steps: list[int] = []
    member_ids_seen: set[int] = set()

    for path, ymd, step in pred_files:
        dates.append(ymd)
        steps.append(step)
        with xr.open_dataset(path) as ds:
            weather_states = [str(value) for value in ds["weather_state"].values.tolist()]
            state_to_index = {name: idx for idx, name in enumerate(weather_states)}
            y_pred = ds["y_pred"].isel(sample=0)
            if "ensemble_member" in ds.coords:
                member_ids = [int(value) for value in ds["ensemble_member"].values.tolist()]
            else:
                member_ids = [1]

            for member_pos, member_id in enumerate(member_ids):
                member_ids_seen.add(member_id)
                for state in requested_states:
                    cfg = VARIABLE_CONFIGS[state]
                    arr = np.asarray(
                        y_pred.isel(ensemble_member=member_pos, weather_state=state_to_index[state]).values,
                        dtype=np.float64,
                    )
                    out_path = out_dir / cfg.dir_name / f"1_{ymd}_{step}_{member_id}_nopoles.grb"
                    write_values_from_template(template_paths[state], out_path, arr[pole_mask])
                    grib_files_written += 1

    summary = {
        "predictions_dir": str(pred_dir),
        "out_dir": str(out_dir),
        "template_root": str(template_root),
        "weather_states": requested_states,
        "prediction_files": [str(path) for path, _, _ in pred_files],
        "dates": sorted(set(dates)),
        "steps_hours": sorted(set(steps)),
        "ensemble_members": sorted(member_ids_seen),
        "pole_mask_point_count": int(pole_mask.sum()),
        "template_point_count": int(ref_lat.size),
        "max_abs_lat_diff": max_abs_lat_diff,
        "max_abs_lon_diff": max_abs_lon_diff,
        "grib_files_written": grib_files_written,
        "template_paths": {state: str(path) for state, path in template_paths.items()},
    }

    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path
        else (out_dir / "staging_summary.json")
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote staging summary: {summary_path}")


if __name__ == "__main__":
    main()
