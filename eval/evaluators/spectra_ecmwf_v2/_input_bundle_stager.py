#!/usr/bin/env python3
"""Stage PrepML input bundle NC files as input-grid GRIBs for spectra_ecmwf.

Reads eefo_*_input_bundle.nc files and writes one GRIB per
(date, step, member, weather_state) in the same nopoles layout used by
_grib_stager.py so that runner.py can hand them to gptosp + _amplitude_computer.
"""
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
    codes_get_array,
    codes_grib_new_from_file,
    codes_release,
    codes_set_long,
    codes_set_string,
    codes_set_values,
    codes_write,
)

BUNDLE_RE = re.compile(
    r".*_date(\d{8})_time\d+_mem(\d+)_step(\d+)h_input_bundle\.nc$"
)

@dataclass(frozen=True)
class BundleVar:
    nc_var: str
    level: int | None
    dir_name: str

VARIABLE_MAP: dict[str, BundleVar] = {
    "10u":  BundleVar("in_lres_10u", None, "10u_sfc"),
    "10v":  BundleVar("in_lres_10v", None, "10v_sfc"),
    "2t":   BundleVar("in_lres_2t",  None, "2t_sfc"),
    "sp":   BundleVar("in_lres_sp",  None, "sp_sfc"),
    "msl":  BundleVar("in_lres_msl", None, "msl_sfc"),
    "t_850":BundleVar("in_lres_t",   850,  "t_850"),
    "z_500":BundleVar("in_lres_z",   500,  "z_500"),
}


# (shortName, level, typeOfLevel)  — None level = surface
GRIB_PARAM: dict[str, tuple[str, int | None, str]] = {
    "10u":   ("10u",  None, "heightAboveGround"),
    "10v":   ("10v",  None, "heightAboveGround"),
    "2t":    ("2t",   None, "heightAboveGround"),
    "sp":    ("sp",   None, "surface"),
    "msl":   ("msl",  None, "surface"),
    "t_850": ("t",    850,  "isobaricInhPa"),
    "z_500": ("z",    500,  "isobaricInhPa"),
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage input bundle NC files as input-grid GRIBs for spectra_ecmwf."
    )
    p.add_argument("--bundles-dir",    required=True)
    p.add_argument("--out-dir",        required=True)
    p.add_argument("--template-grib",  required=True, help="Path to the input-grid template GRIB, e.g. o96-template.grib or o320-template.grib")
    p.add_argument("--weather-states", default="10u,10v,2t,sp,t_850,z_500")
    p.add_argument("--date-list",      default="ALL")
    p.add_argument("--step-list",      default="ALL")
    p.add_argument("--member-list",    default="ALL")
    p.add_argument("--summary-path",   default="")
    return p.parse_args()


def csv_matches(raw: str, needle: int) -> bool:
    if raw == "ALL":
        return True
    return str(needle).strip() in {v.strip() for v in raw.split(",")}


def discover_bundles(bundles_dir: Path) -> list[tuple[Path, int, int, int]]:
    out: list[tuple[Path, int, int, int]] = []
    # Any input grid: the lane decides which bundles live here (o96 for
    # o96->o320, o320 for o320->o1280, ...). BUNDLE_RE below still
    # validates the name shape, so a stray file cannot slip through.
    for p in sorted(bundles_dir.glob("eefo_*_input_bundle.nc")):
        m = BUNDLE_RE.match(p.name)
        if not m:
            continue
        out.append((p, int(m.group(1)), int(m.group(2)), int(m.group(3))))
    if not out:
        raise FileNotFoundError(f"No eefo_*_input_bundle.nc files found in {bundles_dir}")
    return out


def load_template_grib(template_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) arrays from the template GRIB."""
    with template_path.open("rb") as fh:
        gid = codes_grib_new_from_file(fh)
        if gid is None:
            raise RuntimeError(f"Cannot read template GRIB: {template_path}")
        try:
            lat = np.asarray(codes_get_array(gid, "latitudes"),  dtype=np.float64)
            lon = np.asarray(codes_get_array(gid, "longitudes"), dtype=np.float64)
        finally:
            codes_release(gid)
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon


def write_grib(template_path: Path, out_path: Path, values: np.ndarray, *,
               short_name: str, level: int | None, type_of_level: str = "surface") -> None:
    with template_path.open("rb") as fh:
        gid = codes_grib_new_from_file(fh)
        if gid is None:
            raise RuntimeError(f"Cannot read template GRIB: {template_path}")
        clone = codes_clone(gid)
        codes_release(gid)
    try:
        codes_set_string(clone, "typeOfLevel", type_of_level)
        codes_set_string(clone, "shortName", short_name)
        if level is not None:
            codes_set_long(clone, "level", level)
        codes_set_values(clone, np.asarray(values, dtype=np.float64))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as fh:
            codes_write(clone, fh)
    finally:
        codes_release(clone)


def main() -> None:
    args = parse_args()
    bundles_dir   = Path(args.bundles_dir).expanduser().resolve()
    out_dir       = Path(args.out_dir).expanduser().resolve()
    template_path = Path(args.template_grib).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    states = [s.strip() for s in args.weather_states.split(",") if s.strip()]
    unknown = [s for s in states if s not in VARIABLE_MAP]
    if unknown:
        raise ValueError(f"Unsupported weather states: {unknown}")

    tmpl_lat, tmpl_lon = load_template_grib(template_path)
    n_tmpl = len(tmpl_lat)

    bundles = discover_bundles(bundles_dir)
    written = 0
    for path, date, member, step in bundles:
        if not csv_matches(args.date_list,   date):
            continue
        if not csv_matches(args.member_list, member):
            continue
        if not csv_matches(args.step_list,   step):
            continue

        with xr.open_dataset(path) as ds:
            bnd_lat = np.asarray(ds["lat_lres"].values, dtype=np.float64)
            bnd_lon = np.asarray(ds["lon_lres"].values, dtype=np.float64)
            bnd_lon = ((bnd_lon + 180.0) % 360.0) - 180.0

            if len(bnd_lat) != n_tmpl:
                # Build a reindex from bundle lat/lon → template lat/lon order
                from scipy.spatial import cKDTree
                tree = cKDTree(np.column_stack([bnd_lat, bnd_lon]))
                _, idx = tree.query(np.column_stack([tmpl_lat, tmpl_lon]))
            else:
                idx = None

            for state in states:
                bv = VARIABLE_MAP[state]
                arr = np.asarray(ds[bv.nc_var].values, dtype=np.float64)
                if bv.level is not None:
                    level_coord = np.asarray(ds["level"].values)
                    lev_idx = int(np.where(level_coord == bv.level)[0][0])
                    arr = arr[lev_idx]
                arr = arr.ravel()
                if idx is not None:
                    arr = arr[idx]
                out_path = out_dir / bv.dir_name / f"1_{date}_{step}_{member}_nopoles.grb"
                if out_path.exists():
                    continue
                sn, lv, tol = GRIB_PARAM[state]
                write_grib(template_path, out_path, arr, short_name=sn, level=lv, type_of_level=tol)
                written += 1

    summary = {
        "bundles_dir":    str(bundles_dir),
        "out_dir":        str(out_dir),
        "template_grib":  str(template_path),
        "weather_states": states,
        "grib_files_written": written,
    }
    sp = Path(args.summary_path) if args.summary_path else out_dir / "staging_summary.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote staging summary: {sp}")


if __name__ == "__main__":
    main()
