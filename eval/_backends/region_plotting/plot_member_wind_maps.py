"""Single-member field cutout maps: EEFO input / ENFO truth / prediction arms.

Renders 10 m wind speed by default; ``--variable`` also takes ``msl`` (mean
sea level pressure), ``2t`` (2 m temperature), ``t_850`` (850 hPa temperature)
and ``z_500`` (500 hPa geopotential height) from the same files. See
``VARIABLES`` for the field table.

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
# High-pass cutoff for --field fine, in degrees. 0.6 deg sits just below what
# the O320 driving input can resolve, so what survives the filter is the part
# of the field the model had to invent rather than inherit.
DEFAULT_FINE_CUT_DEG = 0.6

DEFAULT_TITLES = {
    "eefo": "EEFO input · O320",
    "enfo": "Operational ENFO · O1280",
    "control": "Control · O1280",
    "guided": "Guided · O1280",
}

# Renderable fields. Each entry fixes the source weather states, the filename
# token, the colour scale and the labels, so adding a field is a table entry
# rather than a new code path. "wind10m" reproduces the original hard-wired
# behaviour exactly, including its colour scale and filename token.
VARIABLES: dict[str, dict] = {
    "wind10m": {
        "states": ("10u", "10v"),
        "combine": "hypot",
        "token": "10mwind",
        "scale": 1.0,
        "offset": 0.0,
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": DEFAULT_VMAX,
        "extend": "max",
        "subtitle": "10 m wind speed",
        "cbar_label": "10 m wind speed (m/s)",
        "fine_vmax": 2.5,
    },
    "msl": {
        "states": ("msl",),
        "combine": "single",
        "token": "msl",
        "scale": 0.01,  # Pa -> hPa
        "offset": 0.0,
        "cmap": "RdBu_r",
        "vmin": 960.0,
        "vmax": 1040.0,
        "extend": "both",
        "subtitle": "Mean sea level pressure",
        "cbar_label": "Mean sea level pressure (hPa)",
        "fine_vmax": 0.8,
    },
    "2t": {
        "states": ("2t",),
        "combine": "single",
        "token": "2t",
        "scale": 1.0,
        "offset": -273.15,  # K -> degC
        "cmap": "RdYlBu_r",
        # The bulk of the field over these regions sits between about -13 and
        # +32 degC; the ends saturate over ice sheets and desert, which is why
        # extend is "both".
        "vmin": -20.0,
        "vmax": 35.0,
        "extend": "both",
        "subtitle": "2 m temperature",
        "cbar_label": "2 m temperature (degC)",
        "fine_vmax": 3.0,
    },
    "t_850": {
        "states": ("t_850",),
        "combine": "single",
        "token": "t850",
        "scale": 1.0,
        "offset": -273.15,  # K -> degC
        "cmap": "RdYlBu_r",
        # Observed span over the Europe cutout / wide North Atlantic in late
        # September 2025 is about -10 to +29 degC; both ends extend.
        "vmin": -10.0,
        "vmax": 30.0,
        "extend": "both",
        "subtitle": "850 hPa temperature",
        "cbar_label": "850 hPa temperature (degC)",
        "fine_vmax": 1.2,
    },
    "z_500": {
        "states": ("z_500",),
        "combine": "single",
        "token": "z500",
        "scale": 1.0 / 98.0665,  # m2/s2 -> decametres of geopotential height
        "offset": 0.0,
        "cmap": "viridis",
        # Observed span over the same regions and season is about 523-592 dam.
        "vmin": 522.0,
        "vmax": 592.0,
        "extend": "both",
        "subtitle": "500 hPa geopotential height",
        "cbar_label": "500 hPa geopotential height (dam)",
        "fine_vmax": 0.3,
    },
}


def resolve_scale(args: argparse.Namespace) -> tuple[dict, float, float]:
    """(spec, vmin, vmax) for the requested variable, honouring explicit overrides."""
    spec = VARIABLES[args.variable]
    if getattr(args, "field", "value") == "fine":
        # The high-pass field is a departure from a local mean, so it is
        # centred on zero and needs a symmetric diverging scale.
        half = spec["fine_vmax"] if args.vmax is None else args.vmax
        return spec, -half, half
    vmin = spec["vmin"] if args.vmin is None else args.vmin
    vmax = spec["vmax"] if args.vmax is None else args.vmax
    return spec, vmin, vmax


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
    p.add_argument("--member", type=int, default=1, help="Ensemble member number (selects within --run files; label-only for --grib panels, which are already single-member).")
    p.add_argument(
        "--members", default=None, metavar="SPEC",
        help="Render every listed member as one multi-panel figure per source instead of "
             "one figure for a single member. SPEC is 'all', a range like '1-10', or a "
             "comma-separated list like '1,3,5'. --member is then only the label used when "
             "a --grib panel carries no member axis.",
    )
    p.add_argument(
        "--grid-cols", type=int, default=5,
        help="Columns in the member grid produced by --members (default: 5).",
    )
    p.add_argument("--output-dir", required=True, help="Directory for the PNGs and manifest.")
    p.add_argument("--no-input", action="store_true", default=False, help="Skip the eefo (input) panel.")
    p.add_argument("--no-truth", action="store_true", default=False, help="Skip the enfo (embedded truth) panel.")
    p.add_argument("--extent", nargs=4, type=float, default=list(DEFAULT_EXTENT), metavar=("LONMIN", "LONMAX", "LATMIN", "LATMAX"), help=f"Map extent (default: {DEFAULT_EXTENT}).")
    p.add_argument("--variable", choices=sorted(VARIABLES), default="wind10m",
                   help="Field to render (default: wind10m, the 10 m wind speed).")
    p.add_argument("--vmin", type=float, default=None, help="Colour-scale minimum (default: the variable's own).")
    p.add_argument("--vmax", type=float, default=None, help=f"Colour-scale maximum (default: the variable's own; {DEFAULT_VMAX} m/s for wind10m).")
    p.add_argument(
        "--field", choices=("value", "fine"), default="value",
        help="value (default): the field itself. fine: only the scales below --fine-cut-deg, "
             "i.e. the detail the O320 input could not carry, on a symmetric diverging scale.",
    )
    p.add_argument(
        "--fine-cut-deg", type=float, default=DEFAULT_FINE_CUT_DEG,
        help=f"High-pass cutoff in degrees for --field fine (default: {DEFAULT_FINE_CUT_DEG}).",
    )
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


def _combine(cols: list[np.ndarray], spec: dict) -> np.ndarray:
    """Reduce the spec's source states to one field, in the spec's own units."""
    val = np.hypot(cols[0], cols[1]) if spec["combine"] == "hypot" else cols[0]
    return val * spec["scale"] + spec["offset"]


def _field(arr: np.ndarray, states: list[str], spec: dict) -> np.ndarray:
    """Extract one renderable field from a (grid_point, weather_state) array."""
    missing = [s for s in spec["states"] if s not in states]
    if missing:
        raise SystemExit(
            f"Prediction file has no weather state(s) {missing}; it carries {states}."
        )
    return _combine([arr[:, states.index(s)] for s in spec["states"]], spec)


def read_grib_field(path: str | Path, spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lat, lon, field) from a GRIB file holding the spec's states for one member."""
    import earthkit.data as ekd

    wanted = set(spec["states"])
    comp: dict[str, np.ndarray] = {}
    lat = lon = None
    for field in ekd.from_source("file", str(path)):
        short_name = field.metadata().get("shortName")
        if short_name in wanted:
            ll = field.to_latlon()
            comp[short_name] = np.asarray(field.to_numpy(), dtype=np.float64).reshape(-1)
            lat = np.asarray(ll["lat"], dtype=np.float64).reshape(-1)
            lon = np.asarray(ll["lon"], dtype=np.float64).reshape(-1)
    missing = wanted - set(comp)
    if missing:
        raise SystemExit(f"{path}: GRIB file is missing {sorted(missing)}.")
    return lat, lon, _combine([comp[s] for s in spec["states"]], spec)


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
    spec: dict,
    vmin: float,
    vmax: float,
    proj_lon: float,
    proj_lat: float,
    field: str = "value",
    fine_cut_deg: float = DEFAULT_FINE_CUT_DEG,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    gx, gy, grid = nearest_grid(lat, lon, val, extent=extent, res=res)
    if field == "fine":
        from scipy.ndimage import gaussian_filter
        # Subtract a Gaussian smooth of the regridded field, so only the scales
        # below the cutoff survive. sigma is half the cutoff.
        grid = grid - gaussian_filter(grid, fine_cut_deg / res / 2.0, mode="nearest")
    init_dt = datetime.strptime(date + time, "%Y%m%d%H%M")
    valid_dt = init_dt + timedelta(hours=step)

    # Render under matplotlib's default rcParams so the output is identical no
    # matter what the importing context (e.g. eval.cli's import chain) has
    # tweaked — keeps these maps reproducible pixel-for-pixel across entry
    # points.
    fine_note = f" · scales below {fine_cut_deg:g} deg" if field == "fine" else ""
    with matplotlib.rc_context({k: v for k, v in matplotlib.rcParamsDefault.items() if k != "backend"}):
        proj = ccrs.LambertConformal(central_longitude=proj_lon, central_latitude=proj_lat)
        fig = plt.figure(figsize=(11, 7.5))
        ax = plt.axes(projection=proj)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        cmap = "RdBu_r" if field == "fine" else spec["cmap"]
        mesh = ax.pcolormesh(gx, gy, grid, transform=ccrs.PlateCarree(), cmap=cmap,
                             vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
        ax.coastlines(resolution="50m", linewidth=1.0)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)
        ax.set_title(
            f"{title}\n{spec['subtitle']}"
            f"{fine_note} · member {member}\n"
            f"init {init_dt:%Y-%m-%d %H} UTC · h{step:03d} · valid {valid_dt:%Y-%m-%d %H} UTC",
            fontsize=13,
        )
        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.04, aspect=40,
                            extend=spec["extend"])
        cbar.set_label(
            spec["cbar_label"] + " · fine-scale part" if field == "fine" else spec["cbar_label"]
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=RENDER_DPI)
        plt.close(fig)



def parse_members(spec: str, available: list[int]) -> list[int]:
    """Members named by a --members spec: 'all', '1-10', or '1,3,5'."""
    text = str(spec).strip().lower()
    if text in ("all", "*"):
        return list(available)
    wanted: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            wanted.extend(range(int(lo), int(hi) + 1))
        else:
            wanted.append(int(part))
    missing = [m for m in wanted if m not in available]
    if missing:
        raise SystemExit(
            f"--members asks for {missing}, but the file carries members {available}."
        )
    return wanted


def _render_grid(
    *,
    out_path: Path,
    title: str,
    members: list[int],
    values: list[np.ndarray],
    date: str,
    time: str,
    step: int,
    lat: np.ndarray,
    lon: np.ndarray,
    res: float,
    extent: tuple[float, float, float, float],
    spec: dict,
    vmin: float,
    vmax: float,
    proj_lon: float,
    proj_lat: float,
    ncols: int = 5,
    field: str = "value",
    fine_cut_deg: float = DEFAULT_FINE_CUT_DEG,
) -> None:
    """One figure holding every member of a single source on a shared colour scale.

    The members are regridded and drawn exactly as the single-member renderer
    does, so a panel of this figure and the corresponding standalone PNG show
    the same field; only the layout differs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    grids = []
    for val in values:
        gx, gy, grid = nearest_grid(lat, lon, val, extent=extent, res=res)
        if field == "fine":
            from scipy.ndimage import gaussian_filter
            grid = grid - gaussian_filter(grid, fine_cut_deg / res / 2.0, mode="nearest")
        grids.append((gx, gy, grid))

    init_dt = datetime.strptime(date + time, "%Y%m%d%H%M")
    valid_dt = init_dt + timedelta(hours=step)
    nrows = int(np.ceil(len(members) / float(ncols)))
    fine_note = f" · scales below {fine_cut_deg:g} deg" if field == "fine" else ""

    with matplotlib.rc_context({k: v for k, v in matplotlib.rcParamsDefault.items()
                                if k != "backend"}):
        proj = ccrs.LambertConformal(central_longitude=proj_lon, central_latitude=proj_lat)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.5 * ncols, 3.6 * nrows),
            subplot_kw={"projection": proj},
        )
        axes = np.atleast_1d(axes).ravel()
        cmap = "RdBu_r" if field == "fine" else spec["cmap"]
        mesh = None
        for ax, member, (gx, gy, grid) in zip(axes, members, grids):
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            mesh = ax.pcolormesh(gx, gy, grid, transform=ccrs.PlateCarree(), cmap=cmap,
                                 vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
            ax.coastlines(resolution="50m", linewidth=0.7)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3)
            peak = float(np.nanmax(grid)) if field == "value" else float(np.nanmax(np.abs(grid)))
            ax.set_title(f"member {member} · max {peak:.1f}", fontsize=10)
        for ax in axes[len(members):]:
            ax.set_visible(False)
        fig.suptitle(
            f"{title}\n{spec['subtitle']}{fine_note} · {len(members)} members\n"
            f"init {init_dt:%Y-%m-%d %H} UTC · h{step:03d} · valid {valid_dt:%Y-%m-%d %H} UTC",
            fontsize=13,
        )
        if mesh is not None:
            cbar = fig.colorbar(mesh, ax=axes.tolist(), orientation="horizontal",
                                pad=0.04, aspect=50, fraction=0.05, extend=spec["extend"])
            cbar.set_label(
                spec["cbar_label"] + " · fine-scale part" if field == "fine"
                else spec["cbar_label"]
            )
        fig.savefig(out_path, dpi=RENDER_DPI, bbox_inches="tight")
        plt.close(fig)


def run_member_grid(args: argparse.Namespace) -> int:
    """--members: one multi-panel figure per source, members side by side."""
    import xarray as xr

    runs = _parse_kv(args.run, "run")
    gribs = _parse_kv(args.grib, "grib")
    titles = {**DEFAULT_TITLES, **_parse_kv(args.title, "title")}
    if not runs:
        raise SystemExit("--members needs at least one --run panel (GRIB files hold one member).")
    if gribs:
        raise SystemExit("--members and --grib cannot be combined: a GRIB panel has no member axis.")
    extent = tuple(args.extent)
    spec, vmin, vmax = resolve_scale(args)
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    token = spec["token"] if args.field == "value" else f"{spec['token']}-fine"
    outputs: list[str] = []
    members: list[int] = []

    # (key, lat, lon, resolution, [field per member])
    panels: list[tuple[str, np.ndarray, np.ndarray, float, list[np.ndarray]]] = []
    for i, (key, run_dir) in enumerate(runs.items()):
        pred_file = Path(run_dir).expanduser() / f"predictions_{args.date}_step{args.step:03d}.nc"
        if not pred_file.exists():
            raise SystemExit(f"{pred_file} not found (run {key!r}).")
        ds = xr.open_dataset(pred_file, decode_timedelta=False)
        states = [str(s) for s in ds["weather_state"].values]
        lat_h, lon_h = ds["lat_hres"].values, ds["lon_hres"].values
        available = [int(m) for m in np.asarray(ds["ensemble_member"].values).reshape(-1)]
        if not members:
            members = parse_members(args.members, available)
        if i == 0:
            if not args.no_input:
                panels.append(("eefo", ds["lat_lres"].values, ds["lon_lres"].values,
                               DEFAULT_LRES_RES,
                               [_field(_member_slice(ds["x"], m), states, spec) for m in members]))
            if not args.no_truth:
                panels.append(("enfo", lat_h, lon_h, DEFAULT_HRES_RES,
                               [_field(_member_slice(ds["y"], m), states, spec) for m in members]))
        panels.append((key, lat_h, lon_h, DEFAULT_HRES_RES,
                       [_field(_member_slice(ds["y_pred"], m), states, spec) for m in members]))

    for key, lat, lon, res, values in panels:
        out_path = out_dir / (
            f"{key}_{token}_init{args.date}_members{len(members):02d}"
            f"_{args.region_tag}_f{args.step:03d}.png"
        )
        _render_grid(
            out_path=out_path, title=titles.get(key, f"{key.capitalize()} · O1280"),
            members=members, values=values, date=args.date, time=args.time, step=args.step,
            lat=lat, lon=lon, res=res, extent=extent, spec=spec, vmin=vmin, vmax=vmax,
            proj_lon=args.proj_lon, proj_lat=args.proj_lat, ncols=int(args.grid_cols),
            field=args.field, fine_cut_deg=args.fine_cut_deg,
        )
        outputs.append(str(out_path))
        print(f"saved {out_path}", flush=True)

    write_manifest(out_root=out_dir, payload={
        "tool": "membermaps", "mode": "member_grid",
        "date": args.date, "time": args.time, "step": args.step, "members": members,
        "variable": args.variable, "field": args.field, "fine_cut_deg": args.fine_cut_deg,
        "runs": runs, "extent": list(extent), "vmin": vmin, "vmax": vmax,
        "outputs": outputs,
    }, filename=(
        f"membermaps_manifest_{token}_init{args.date}"
        f"_members{len(members):02d}_f{args.step:03d}.json"
    ))
    return 0


def run(args: argparse.Namespace) -> int:
    import xarray as xr

    if getattr(args, "members", None):
        return run_member_grid(args)

    runs = _parse_kv(args.run, "run")
    gribs = _parse_kv(args.grib, "grib")
    titles = {**DEFAULT_TITLES, **_parse_kv(args.title, "title")}
    if not runs and not gribs:
        raise SystemExit("Nothing to plot: give at least one --run or --grib panel.")
    extent = tuple(args.extent)
    spec, vmin, vmax = resolve_scale(args)
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
                               _field(_member_slice(ds["x"], args.member), states, spec), DEFAULT_LRES_RES))
            if not args.no_truth:
                panels.append(("enfo", lat_h, lon_h,
                               _field(_member_slice(ds["y"], args.member), states, spec), DEFAULT_HRES_RES))
        panels.append((key, lat_h, lon_h,
                       _field(_member_slice(ds["y_pred"], args.member), states, spec), DEFAULT_HRES_RES))
    for key, grib_path in gribs.items():
        lat, lon, val = read_grib_field(grib_path, spec)
        res = DEFAULT_HRES_RES if lat.size > 2_000_000 else DEFAULT_LRES_RES
        panels.append((key, lat, lon, val, res))

    outputs: list[str] = []
    # "fine" panels get their own filename token so the two field kinds never
    # overwrite each other in a shared output directory.
    token = spec["token"] if args.field == "value" else f"{spec['token']}-fine"
    for key, lat, lon, val, res in panels:
        out_path = out_dir / (
            f"{key}_{token}_init{args.date}_n{args.member:03d}"
            f"_{args.region_tag}_f{args.step:03d}.png"
        )
        _render(
            out_path=out_path, title=titles.get(key, f"{key.capitalize()} · O1280"),
            member=args.member, date=args.date, time=args.time, step=args.step,
            lat=lat, lon=lon, val=val, res=res, extent=extent, spec=spec,
            vmin=vmin, vmax=vmax,
            proj_lon=args.proj_lon, proj_lat=args.proj_lat,
            field=args.field, fine_cut_deg=args.fine_cut_deg,
        )
        outputs.append(str(out_path))
        print(f"saved {out_path}", flush=True)

    write_manifest(out_root=out_dir, payload={
        "tool": "membermaps",
        "date": args.date, "time": args.time, "step": args.step, "member": args.member,
        "variable": args.variable,
        "field": args.field, "fine_cut_deg": args.fine_cut_deg,
        "runs": runs, "gribs": gribs, "extent": list(extent),
        "vmin": vmin, "vmax": vmax,
        "outputs": outputs,
    }, filename=(
        f"membermaps_manifest_{token}_init{args.date}"
        f"_n{args.member:03d}_f{args.step:03d}.json"
    ))
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
