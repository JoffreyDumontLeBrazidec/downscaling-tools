"""Single-member 10 m wind-speed cutout maps: EEFO input / ENFO truth / prediction arms.

Renders the member-level case-inspection map set (one PNG per panel, shared
colour scale, projection and title style) that used to be produced by ad-hoc
scripts during the September-2025 j9f3/j95z review. Two source modes, freely
mixed in one invocation:

* ``--run key=<predictions_dir>`` — a directory of retrieved
  ``predictions_<date>_step<SSS>.nc`` files (the standalone
  ``eval.predict.prepml --retrieve`` output). ``x`` (EEFO O320 driver),
  ``y`` (embedded same-index ENFO member) and ``y_pred`` are read from the
  first run's file; every additional run contributes its own ``y_pred`` panel.
* ``--grib key=<file>`` — a GRIB file holding 10u/10v for ONE member
  (e.g. ``fdb read`` of an rd expver, or a MARS pull of od enfo/eefo),
  for steps that predictions files do not cover (typically step 0).

The embedded ``y`` is the *same-index* ENFO member — a genuine ENFO member,
but an independent realization from the EEFO driver (EEFO/ENFO are not
paired); the panel is labelled "Operational ENFO" accordingly.

Diagnostic maps only — nothing here scores anything.

Canonical invocation: ``python -m eval.cli membermaps ...`` (also runnable as
``python -m eval._backends.region_plotting.plot_member_wind_maps``).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .plotting.manifest import write_manifest

DEFAULT_EXTENT = (-45.0, 55.0, 27.0, 72.0)
DEFAULT_MARGIN = 8.0
DEFAULT_HRES_RES = 0.08
DEFAULT_LRES_RES = 0.28
DEFAULT_VMAX = 25.0
RENDER_DPI = 140
# Latitude scale factor for the nearest-neighbour lookup: reduced Gaussian
# rows are denser in latitude than longitude, so an isotropic lookup would
# smear rows; 1.4 keeps the lookup roughly isotropic in grid spacing.
LAT_LOOKUP_SCALE = 1.4

DEFAULT_TITLES = {
    "eefo": "EEFO input · O320",
    "enfo": "Operational ENFO · O1280",
    "control": "Control · O1280",
    "guided": "Guided · O1280",
}


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="membermaps",
        add_help=add_help,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run", action="append", default=[], metavar="KEY=DIR",
        help="Prediction arm: KEY=<predictions dir>. Repeatable; the first run also provides the eefo/enfo panels.",
    )
    p.add_argument(
        "--grib", action="append", default=[], metavar="KEY=FILE",
        help="Extra panel from a single-member GRIB file holding 10u/10v (e.g. step-0 fields from FDB/MARS). Repeatable.",
    )
    p.add_argument(
        "--title", action="append", default=[], metavar="KEY=TITLE",
        help="Override the first title line for a panel key (defaults: eefo/enfo/control/guided presets, else KEY · O1280).",
    )
    p.add_argument("--date", required=True, help="Init date YYYYMMDD.")
    p.add_argument("--step", type=int, required=True, help="Lead time in hours (predictions file suffix for --run panels).")
    p.add_argument("--member", type=int, required=True, help="Ensemble member number (selects within --run files; label-only for --grib panels, which are already single-member).")
    p.add_argument("--output-dir", required=True, help="Directory for the PNGs and manifest.")
    p.add_argument("--no-input", action="store_true", default=False, help="Skip the eefo (input) panel.")
    p.add_argument("--no-truth", action="store_true", default=False, help="Skip the enfo (embedded truth) panel.")
    p.add_argument("--extent", nargs=4, type=float, default=list(DEFAULT_EXTENT), metavar=("LONMIN", "LONMAX", "LATMIN", "LATMAX"), help=f"Map extent (default: {DEFAULT_EXTENT}).")
    p.add_argument("--vmax", type=float, default=DEFAULT_VMAX, help=f"Colour-scale maximum in m/s (default: {DEFAULT_VMAX}).")
    p.add_argument("--region-tag", default="europe-cutout", help="Region tag used in output filenames (default: europe-cutout).")
    p.add_argument("--time", default="0000", help="Init time HHMM (default: 0000).")
    p.add_argument("--proj-lon", type=float, default=5.0, help="Lambert conformal central longitude (default: 5.0).")
    p.add_argument("--proj-lat", type=float, default=50.0, help="Lambert conformal central latitude (default: 50.0).")
    return p


def _parse_kv(specs: list[str], what: str) -> dict[str, str]:
    """Parse repeated KEY=VALUE options, preserving order."""
    out: dict[str, str] = {}
    for spec in specs:
        key, sep, value = spec.partition("=")
        if not sep or not key or not value:
            raise SystemExit(f"Bad --{what} spec {spec!r}: expected KEY=VALUE.")
        out[key] = value
    return out


def nearest_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    val: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    margin: float = DEFAULT_MARGIN,
    res: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour resampling of unstructured points to a regular lon/lat grid.

    Returns (grid_lons, grid_lats, grid_values); the grid covers extent+margin
    so a conic projection's corners stay filled.
    """
    from scipy.spatial import cKDTree

    lon = np.where(lon > 180.0, lon - 360.0, lon)
    m = ((lon >= extent[0] - margin) & (lon <= extent[1] + margin)
         & (lat >= extent[2] - margin) & (lat <= extent[3] + margin))
    if not np.any(m):
        raise ValueError("No source points fall inside the requested extent.")
    tree = cKDTree(np.column_stack([lon[m], lat[m] * LAT_LOOKUP_SCALE]))
    gx = np.arange(extent[0] - margin, extent[1] + margin, res)
    gy = np.arange(extent[2] - margin, extent[3] + margin, res)
    grid_x, grid_y = np.meshgrid(gx, gy)
    _, idx = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel() * LAT_LOOKUP_SCALE]), workers=-1)
    return gx, gy, val[m][idx].reshape(grid_y.shape)


def _member_slice(da, member: int) -> np.ndarray:
    """Select one member as (grid_point, weather_state), matching the spectra proxy convention."""
    d = da.isel(sample=0) if "sample" in da.dims else da
    if "ensemble_member" in d.dims:
        d = d.sel(ensemble_member=member)
    arr = np.asarray(d.values, dtype=np.float64)
    if d.dims and d.dims[0] == "weather_state" and arr.ndim == 2:
        arr = arr.T
    return arr


def _wind_speed(arr: np.ndarray, states: list[str]) -> np.ndarray:
    return np.hypot(arr[:, states.index("10u")], arr[:, states.index("10v")])


def read_grib_wind(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lat, lon, wind speed) from a GRIB file holding 10u and 10v for one member."""
    import earthkit.data as ekd

    comp: dict[str, np.ndarray] = {}
    lat = lon = None
    for field in ekd.from_source("file", str(path)):
        short_name = field.metadata().get("shortName")
        if short_name in ("10u", "10v"):
            ll = field.to_latlon()
            comp[short_name] = np.asarray(field.to_numpy(), dtype=np.float64).reshape(-1)
            lat = np.asarray(ll["lat"], dtype=np.float64).reshape(-1)
            lon = np.asarray(ll["lon"], dtype=np.float64).reshape(-1)
    missing = {"10u", "10v"} - set(comp)
    if missing:
        raise SystemExit(f"{path}: GRIB file is missing {sorted(missing)}.")
    return lat, lon, np.hypot(comp["10u"], comp["10v"])


def _render(
    *,
    out_path: Path,
    title: str,
    member: int,
    date: str,
    time: str,
    step: int,
    lat: np.ndarray,
    lon: np.ndarray,
    val: np.ndarray,
    res: float,
    extent: tuple[float, float, float, float],
    vmax: float,
    proj_lon: float,
    proj_lat: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    gx, gy, grid = nearest_grid(lat, lon, val, extent=extent, res=res)
    init_dt = datetime.strptime(date + time, "%Y%m%d%H%M")
    valid_dt = init_dt + timedelta(hours=step)

    proj = ccrs.LambertConformal(central_longitude=proj_lon, central_latitude=proj_lat)
    fig = plt.figure(figsize=(11, 7.5))
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    mesh = ax.pcolormesh(gx, gy, grid, transform=ccrs.PlateCarree(), cmap="viridis",
                         vmin=0.0, vmax=vmax, shading="auto", rasterized=True)
    ax.coastlines(resolution="50m", linewidth=1.0)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)
    ax.set_title(
        f"{title}\n10 m wind speed · member {member}\n"
        f"init {init_dt:%Y-%m-%d %H} UTC · h{step:03d} · valid {valid_dt:%Y-%m-%d %H} UTC",
        fontsize=13,
    )
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.04, aspect=40, extend="max")
    cbar.set_label("10 m wind speed (m/s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=RENDER_DPI)
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    import xarray as xr

    runs = _parse_kv(args.run, "run")
    gribs = _parse_kv(args.grib, "grib")
    titles = {**DEFAULT_TITLES, **_parse_kv(args.title, "title")}
    if not runs and not gribs:
        raise SystemExit("Nothing to plot: give at least one --run or --grib panel.")
    extent = tuple(args.extent)
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float]] = []
    for i, (key, run_dir) in enumerate(runs.items()):
        pred_file = Path(run_dir).expanduser() / f"predictions_{args.date}_step{args.step:03d}.nc"
        if not pred_file.exists():
            raise SystemExit(f"{pred_file} not found (run {key!r}).")
        ds = xr.open_dataset(pred_file, decode_timedelta=False)
        states = [str(s) for s in ds["weather_state"].values]
        lat_h, lon_h = ds["lat_hres"].values, ds["lon_hres"].values
        if i == 0:
            if not args.no_input:
                panels.append(("eefo", ds["lat_lres"].values, ds["lon_lres"].values,
                               _wind_speed(_member_slice(ds["x"], args.member), states), DEFAULT_LRES_RES))
            if not args.no_truth:
                panels.append(("enfo", lat_h, lon_h,
                               _wind_speed(_member_slice(ds["y"], args.member), states), DEFAULT_HRES_RES))
        panels.append((key, lat_h, lon_h,
                       _wind_speed(_member_slice(ds["y_pred"], args.member), states), DEFAULT_HRES_RES))
    for key, grib_path in gribs.items():
        lat, lon, wind = read_grib_wind(grib_path)
        res = DEFAULT_HRES_RES if lat.size > 2_000_000 else DEFAULT_LRES_RES
        panels.append((key, lat, lon, wind, res))

    outputs: list[str] = []
    for key, lat, lon, wind, res in panels:
        out_path = out_dir / (
            f"{key}_10mwind_init{args.date}_n{args.member:03d}_{args.region_tag}_f{args.step:03d}.png"
        )
        _render(
            out_path=out_path, title=titles.get(key, f"{key.capitalize()} · O1280"),
            member=args.member, date=args.date, time=args.time, step=args.step,
            lat=lat, lon=lon, val=wind, res=res, extent=extent, vmax=args.vmax,
            proj_lon=args.proj_lon, proj_lat=args.proj_lat,
        )
        outputs.append(str(out_path))
        print(f"saved {out_path}", flush=True)

    write_manifest(out_root=out_dir, payload={
        "tool": "membermaps",
        "date": args.date, "time": args.time, "step": args.step, "member": args.member,
        "runs": runs, "gribs": gribs, "extent": list(extent), "vmax": args.vmax,
        "outputs": outputs,
    }, filename=f"membermaps_manifest_init{args.date}_n{args.member:03d}_f{args.step:03d}.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
