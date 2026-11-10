#!/usr/bin/env python3
"""Stage PrepML input bundle NC files as input-grid GRIBs for spectra_ecmwf.

Reads eefo_*_input_bundle.nc files and writes one GRIB per
(date, step, member, weather_state) in the same nopoles layout used by
_grib_stager.py so that runner.py can hand them to gptosp + _amplitude_computer.

With --predictions-dir instead of --bundles-dir it stages the NATIVE input ``x``
(sample, ensemble_member, grid_point_lres, weather_state) of the prediction files,
on lat_lres/lon_lres, onto the input-grid template (rule 41, 9 Nov 2026: the input
is scored on its own grid, never interpolated to the output grid first). That mode
never interpolates: the native points must be the template's points, in the same
order or as an exact permutation, otherwise it stops.
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
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bundles-dir")
    src.add_argument("--predictions-dir",
                     help="Stage the native input x of predictions_*.nc (on lat_lres/lon_lres) instead of bundles")
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



def _nn_index_great_circle(src_lat, src_lon, dst_lat, dst_lon) -> np.ndarray:
    """Nearest-neighbour index from source points to destination points.

    Distances are great-circle: both grids are projected onto the unit sphere
    before the KD-tree query, so the result does not distort near the poles or
    across the +/-180 longitude seam.  Same construction as build_nn_index in
    eval/shared/precip/sources.py, kept local so this stager stays a
    standalone script.
    """
    from scipy.spatial import cKDTree

    def to_xyz(lat, lon):
        la = np.radians(np.asarray(lat, dtype=np.float64))
        lo = np.radians(np.asarray(lon, dtype=np.float64))
        cos_la = np.cos(la)
        return np.column_stack((cos_la * np.cos(lo), cos_la * np.sin(lo),
                                np.sin(la)))

    tree = cKDTree(to_xyz(src_lat, src_lon))
    _dist, idx = tree.query(to_xyz(dst_lat, dst_lon), k=1, workers=-1)
    return idx.astype(np.int64)


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


PRED_RE = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")
# A point counts as the same point when it sits within this angle of the template's (about 1 m).
_SAME_POINT_DEG = 1e-5


def native_point_order(src_lat, src_lon, tmpl_lat, tmpl_lon) -> np.ndarray | None:
    """How the native points map onto the template, WITHOUT interpolation.

    Returns None when the two point sets are identical and in the same order, else
    the permutation idx with template value i = native value idx[i]. Raises when the
    native points are not the template's points (another grid, or a regional subset):
    the native input is then not on the template grid, and remapping it would be the
    interpolation rule 41 forbids.
    """
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = ((np.asarray(src_lon, dtype=np.float64) + 180.0) % 360.0) - 180.0
    if src_lat.size != tmpl_lat.size:
        raise ValueError(
            f"native input has {src_lat.size} points but the template grid has {tmpl_lat.size}; "
            "the native field is not on the template grid (regional file, or the wrong template)"
        )
    dlon = np.abs(((src_lon - tmpl_lon) + 180.0) % 360.0 - 180.0)
    if np.abs(src_lat - tmpl_lat).max() <= _SAME_POINT_DEG and dlon.max() <= 1e-3:
        return None
    idx = _nn_index_great_circle(src_lat, src_lon, tmpl_lat, tmpl_lon)
    la, lo = np.radians(src_lat[idx]), np.radians(src_lon[idx])
    tla, tlo = np.radians(tmpl_lat), np.radians(tmpl_lon)
    cosd = np.sin(la) * np.sin(tla) + np.cos(la) * np.cos(tla) * np.cos(lo - tlo)
    worst = float(np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0))).max())
    if worst > _SAME_POINT_DEG or np.unique(idx).size != idx.size:
        raise ValueError(
            f"native input points are not a permutation of the template's (worst distance {worst:.3g} deg); "
            "refusing to interpolate the native input"
        )
    return idx


def stage_native_predictions(pred_dir: Path, out_dir: Path, template_path: Path, states: list[str], *,
                             date_list: str = "ALL", step_list: str = "ALL",
                             member_list: str = "ALL") -> dict:
    """Write one native-grid GRIB per (date, step, member, state) from the x of predictions_*.nc."""
    tmpl_lat, tmpl_lon = load_template_grib(template_path)
    written = 0
    order_kind = None
    members_seen: set[int] = set()
    dates: set[int] = set()
    steps: set[int] = set()
    files = sorted(p for p in pred_dir.glob("predictions_*.nc") if PRED_RE.match(p.name))
    if not files:
        raise FileNotFoundError(f"No predictions_*.nc files found in {pred_dir}")
    for path in files:
        m = PRED_RE.match(path.name)
        date, step = int(m.group(1)), int(m.group(2))
        if not (csv_matches(date_list, date) and csv_matches(step_list, step)):
            continue
        with xr.open_dataset(path) as ds:
            if "x" not in ds.variables or "lat_lres" not in ds.variables:
                raise ValueError(f"{path} has no native input x on lat_lres/lon_lres")
            idx = native_point_order(ds["lat_lres"].values.reshape(-1), ds["lon_lres"].values.reshape(-1),
                                     tmpl_lat, tmpl_lon)
            order_kind = order_kind or ("same order" if idx is None else "exact permutation")
            ws = [str(v) for v in ds["weather_state"].values.tolist()]
            missing = [s for s in states if s not in ws]
            if missing:
                raise ValueError(f"{path}: x carries no {missing}")
            x = ds["x"].isel(sample=0) if "sample" in ds["x"].dims else ds["x"]
            if "ensemble_member" in x.dims:
                ids = ([int(v) for v in ds["ensemble_member"].values.tolist()] if "ensemble_member" in ds.variables
                       else list(range(1, x.sizes["ensemble_member"] + 1)))
            else:
                ids = [1]
            dates.add(date)
            steps.add(step)
            for pos, member in enumerate(ids):
                if not csv_matches(member_list, member):
                    continue
                members_seen.add(member)
                for state in states:
                    sel = x.isel(weather_state=ws.index(state))
                    if "ensemble_member" in sel.dims:
                        sel = sel.isel(ensemble_member=pos)
                    arr = np.asarray(sel.values, dtype=np.float64).reshape(-1)
                    if idx is not None:
                        arr = arr[idx]
                    out_path = out_dir / VARIABLE_MAP[state].dir_name / f"1_{date}_{step}_{member}_nopoles.grb"
                    if out_path.exists():
                        continue
                    sn, lv, tol = GRIB_PARAM[state]
                    write_grib(template_path, out_path, arr, short_name=sn, level=lv, type_of_level=tol)
                    written += 1
    return {
        "source": "x (native input grid)",
        "predictions_dir": str(pred_dir),
        "out_dir": str(out_dir),
        "template_grib": str(template_path),
        "weather_states": states,
        "dates": sorted(dates),
        "steps_hours": sorted(steps),
        "ensemble_members": sorted(members_seen),
        "grib_files_written": written,
        "template_point_count": int(tmpl_lat.size),
        "source_point_count": int(tmpl_lat.size),
        "interpolated_to_template": False,
        "interpolation_method": "none_native_grid",
        "point_order": order_kind,
    }


def main() -> None:
    args = parse_args()
    if args.predictions_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        states = [s.strip() for s in args.weather_states.split(",") if s.strip()]
        unknown = [s for s in states if s not in VARIABLE_MAP]
        if unknown:
            raise ValueError(f"Unsupported weather states: {unknown}")
        summary = stage_native_predictions(
            Path(args.predictions_dir).expanduser().resolve(), out_dir,
            Path(args.template_grib).expanduser().resolve(), states,
            date_list=args.date_list, step_list=args.step_list, member_list=args.member_list)
        sp = Path(args.summary_path) if args.summary_path else out_dir / "staging_summary.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote staging summary: {sp}")
        return
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
    src_point_count = 0
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
            src_point_count = len(bnd_lat)

            if len(bnd_lat) != n_tmpl:
                # Source and template grids differ, so map source values onto
                # the template by nearest neighbour.  When the template is the
                # TARGET grid (o1280) rather than the source grid (o320) this
                # is exactly the driver interpolation the spectral comparison
                # needs: it puts the coarse driver on the same grid, and hence
                # the same spectral support, as the model and the truth.
                #
                # Nearest neighbour is measured as a GREAT-CIRCLE distance on
                # the unit sphere, matching the project's gate-verified
                # build_nn_index (eval/shared/precip/sources.py).  A KD-tree
                # on raw (lat, lon) is wrong near the poles, where a degree of
                # longitude is far shorter than a degree of latitude, and
                # across the +/-180 seam, where 179.9 and -179.9 are adjacent
                # on the sphere but maximally distant in lat/lon coordinates.
                idx = _nn_index_great_circle(bnd_lat, bnd_lon, tmpl_lat, tmpl_lon)
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
        # Auditability: a reader must be able to tell, from the summary alone,
        # which grid these spectra live on and whether the driver was
        # interpolated to get there.  template_point_count is the field the
        # support check compares against the model and truth sides.
        "template_point_count": int(n_tmpl),
        "source_point_count":   int(src_point_count) if src_point_count else None,
        "interpolated_to_template": bool(src_point_count and src_point_count != n_tmpl),
        "interpolation_method": ("nearest_neighbour_great_circle"
                                 if (src_point_count and src_point_count != n_tmpl)
                                 else "none_same_grid"),
    }
    sp = Path(args.summary_path) if args.summary_path else out_dir / "staging_summary.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote staging summary: {sp}")


if __name__ == "__main__":
    main()
