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

Canonical invocation: ``python -m eval.cli zoom_maps ...`` (also runnable as
``python -m eval.evaluators.zoom_maps.core.plot_member_wind_maps``).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from eval.shared.manifest import write_manifest

DEFAULT_EXTENT = (-45.0, 55.0, 27.0, 72.0)
DEFAULT_MARGIN = 8.0
DEFAULT_HRES_RES = 0.08
DEFAULT_LRES_RES = 0.28
DEFAULT_VMAX = 25.0
RENDER_DPI = 150  # house PNG resolution (eval.plotting.save_figure); kept for importers
# Latitude scale factor for the nearest-neighbour lookup: reduced Gaussian
# rows are denser in latitude than longitude, so an isotropic lookup would
# smear rows; 1.4 keeps the lookup roughly isotropic in grid spacing.
LAT_LOOKUP_SCALE = 1.4
# High-pass scale for --field fine, in degrees. 0.6 deg sits just below what the
# O320 driving input can resolve, so the fully-transmitted part of the filtered
# field is what the model had to invent rather than inherit. The Gaussian
# rolloff also passes half the amplitude at 1.6 deg, which the driver DOES
# carry; that leakage is inherited correctly by both truth and model, so it
# biases a truth-versus-model contrast towards agreement, never away from it.
DEFAULT_FINE_CUT_DEG = 0.6

DEFAULT_TITLES = {
    "eefo": "Input (EEFO O320)",
    "enfo": "Truth (operational ENFO O1280, same member number)",
    "control": "Model, control arm (O1280)",
    "guided": "Model, guided arm (O1280)",
}


def default_title(key: str) -> str:
    """Panel title for a source key without an explicit --title."""
    if key.lower() == "model":
        return "Model (O1280)"
    return DEFAULT_TITLES.get(key, f"Model, {key} (O1280)")

# Renderable fields. Each entry fixes the source weather states, the filename
# token and the fixed colour scale, so adding a field is a table entry rather
# than a new code path. Names, display units, the native-to-display conversion
# and the field colour maps come from eval.plotting.variables (``_with_house``
# below): msl in hPa, z_500 in dam, temperatures in K, wind in m s-1. The fixed
# colour ranges are given in those display units. "wind10m" keeps the original
# token and 0-25 m s-1 scale.
_VARIABLE_KEYS = {"wind10m": "10ff", "msl": "msl", "2t": "2t", "t_850": "t_850", "z_500": "z_500"}

VARIABLES: dict[str, dict] = {
    "wind10m": {
        "states": ("10u", "10v"),
        "combine": "hypot",
        "token": "10mwind",
        "vmin": 0.0,
        "vmax": DEFAULT_VMAX,
        "extend": "max",
        "fine_vmax": 2.5,
    },
    "msl": {
        "states": ("msl",),
        "combine": "single",
        "token": "msl",
        "vmin": 960.0,
        "vmax": 1040.0,
        "extend": "both",
        "fine_vmax": 0.8,
    },
    "2t": {
        "states": ("2t",),
        "combine": "single",
        "token": "2t",
        # The bulk of the field over these regions sits between about 260 and
        # 305 K (-13 to +32 degC); the ends saturate over ice sheets and desert,
        # which is why extend is "both". Same range as the former -20..35 degC.
        "vmin": 253.15,
        "vmax": 308.15,
        "extend": "both",
        "fine_vmax": 3.0,
    },
    "t_850": {
        "states": ("t_850",),
        "combine": "single",
        "token": "t850",
        # Observed span over the Europe cutout / wide North Atlantic in late
        # September 2025 is about 263-302 K (-10 to +29 degC); both ends extend.
        "vmin": 263.15,
        "vmax": 303.15,
        "extend": "both",
        "fine_vmax": 1.2,
    },
    "z_500": {
        "states": ("z_500",),
        "combine": "single",
        "token": "z500",
        # Observed span over the same regions and season is about 523-592 dam.
        "vmin": 522.0,
        "vmax": 592.0,
        "extend": "both",
        "fine_vmax": 0.3,
    },
}


def _with_house(name: str, spec: dict) -> dict:
    """Fill name, unit, conversion and colour map of a VARIABLES entry from the house table."""
    from eval.plotting.variables import variable_spec

    house = variable_spec(_VARIABLE_KEYS[name])
    scale, offset = house.scale_offset
    return {
        **spec,
        "scale": scale,
        "offset": offset,
        "cmap": house.cmap,
        "subtitle": house.name,
        "cbar_label": house.label,
        "house_key": house.key,
    }


VARIABLES = {name: _with_house(name, spec) for name, spec in VARIABLES.items()}


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
        prog="zoom_maps",
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
    p.add_argument("--trajectory-npz", action="append", default=[], metavar="KEY=FILE",
                   help="Opt-in saved seeding_fields.npz comparison; repeat for three models, with target from the first.")
    p.add_argument("--rotation-instrument", default=None, help="Path to tc_rotation.py for saved-NPZ panels.")
    p.add_argument("--seed", type=int, default=1000, help="Saved free-sample seed for --trajectory-npz.")
    p.add_argument("--storm-half-width-km", type=float, default=350, help="Saved-NPZ panel half width.")
    p.add_argument("--band-vmax", type=float, default=6, help="Symmetric wind-speed band colour limit in m/s.")
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
    p.add_argument("--vmin", type=float, default=None, help="Colour-scale minimum in display units (hPa, K, dam, m/s; default: the variable's own).")
    p.add_argument("--vmax", type=float, default=None, help=f"Colour-scale maximum in display units (default: the variable's own; {DEFAULT_VMAX} m/s for wind10m).")
    p.add_argument(
        "--field", choices=("value", "fine"), default="value",
        help="value (default): the field itself. fine: a high-pass keeping the scales at and "
             "below --fine-cut-deg, i.e. the detail the O320 input could not carry, on a "
             "symmetric diverging scale.",
    )
    p.add_argument(
        "--fine-cut-deg", type=float, default=DEFAULT_FINE_CUT_DEG,
        help="High-pass scale in degrees for --field fine (default: "
             f"{DEFAULT_FINE_CUT_DEG}). Gaussian rolloff, not a brick wall: it transmits 99%% "
             "at this wavelength and 50%% at 2.67x it.",
    )
    p.add_argument("--region-tag", default="europe-cutout", help="Region tag used in output filenames (default: europe-cutout).")
    p.add_argument("--time", default="0000", help="Init time HHMM (default: 0000).")
    p.add_argument("--proj-lon", type=float, default=5.0, help="Lambert conformal central longitude (default: 5.0). Ignored when the extent crosses the dateline (plate carree is used).")
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


def _projection(extent, proj_lon: float, proj_lat: float):
    """Shared projection rule; an explicit --proj-lon/--proj-lat recentres the Lambert cone."""
    import cartopy.crs as ccrs
    from eval.plotting.maps_helpers import lambert_for, region_projection

    proj = region_projection(*extent)
    if isinstance(proj, ccrs.PlateCarree):
        return proj
    return lambert_for(proj_lon, proj_lat)


def _style_for(spec: dict, field: str):
    """(colour map, colour-bar label) for a value or a fine-scale (high-pass) panel."""
    from eval.plotting import variable_spec

    house = variable_spec(spec["house_key"])
    if field == "fine":
        # The high-pass field is a departure from a local mean: zero-centred, RdBu_r.
        return "RdBu_r", f"{house.name}, fine-scale part ({house.unit})"
    return house.field_cmap(), spec["cbar_label"]


def _fine_note(field: str, fine_cut_deg: float) -> str:
    # State the real filter response, not just the parameter: the high-pass passes
    # 99% at fine_cut_deg and 50% at 2.67x it (Gaussian rolloff, not a brick wall).
    if field != "fine":
        return ""
    return f", high-pass (full below {fine_cut_deg:g}°, half at {2.67 * fine_cut_deg:.1f}°)"


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
    """One map panel, written as ``out_path`` (PNG, 150 dpi) plus a PDF sibling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from eval.plotting import add_geography, eval_style, save_figure

    gx, gy, grid = nearest_grid(lat, lon, val, extent=extent, res=res)
    if field == "fine":
        from scipy.ndimage import gaussian_filter
        # Subtract a Gaussian smooth of the regridded field: a high-pass built as
        # identity minus low-pass, in real space, with no transform involved. sigma
        # is half --fine-cut-deg, which makes the high-pass transmit 99% at a
        # wavelength of fine_cut_deg and 50% at 2.67x it. The rolloff is gradual, so
        # the panel is not a sharp band.
        grid = grid - gaussian_filter(grid, fine_cut_deg / res / 2.0, mode="nearest")
    init_dt = datetime.strptime(date + time, "%Y%m%d%H%M")
    valid_dt = init_dt + timedelta(hours=step)
    cmap, cbar_label = _style_for(spec, field)

    with eval_style():
        fig = plt.figure(figsize=(11, 7.5))
        ax = fig.add_subplot(1, 1, 1, projection=_projection(extent, proj_lon, proj_lat))
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        mesh = ax.pcolormesh(gx, gy, grid, transform=ccrs.PlateCarree(), cmap=cmap,
                             vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
        add_geography(ax, coast_lw=0.9, label_size=8)
        ax.set_title(
            f"{title}\n{spec['subtitle']}{_fine_note(field, fine_cut_deg)}, member {member}\n"
            f"init {init_dt:%Y-%m-%d %H} UTC, lead time {step} h, valid {valid_dt:%Y-%m-%d %H} UTC",
            fontsize=12,
        )
        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.06, aspect=40, shrink=0.8,
                            extend=spec["extend"] if field == "value" else "both")
        cbar.set_label(cbar_label)
        save_figure(fig, out_path, close=True)


def parse_members(spec: str, available: list[int]) -> list[int]:
    """Members named by a --members spec: 'all', '1-10', or '1,3,5'.

    Members the file does not carry are dropped with a warning rather than
    treated as an error, because arms of the same campaign do not always hold the
    same members: a member whose inference job failed leaves a gap, and a figure
    of the nine members that exist is more useful than no figure at all.
    """
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
    present = [m for m in wanted if m in available]
    missing = [m for m in wanted if m not in available]
    if missing:
        print(f"warning: members {missing} are not in this file (it carries "
              f"{available}); rendering {present}", flush=True)
    if not present:
        raise SystemExit(f"--members asks for {wanted}, none of which the file carries.")
    return present


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
    from eval.plotting import add_geography, eval_style, save_figure, variable_spec

    grids = []
    for val in values:
        gx, gy, grid = nearest_grid(lat, lon, val, extent=extent, res=res)
        if field == "fine":
            from scipy.ndimage import gaussian_filter
            # Same real-space high-pass as the single-member renderer.
            grid = grid - gaussian_filter(grid, fine_cut_deg / res / 2.0, mode="nearest")
        grids.append((gx, gy, grid))

    init_dt = datetime.strptime(date + time, "%Y%m%d%H%M")
    valid_dt = init_dt + timedelta(hours=step)
    nrows = int(np.ceil(len(members) / float(ncols)))
    cmap, cbar_label = _style_for(spec, field)
    unit = variable_spec(spec["house_key"]).unit

    with eval_style():
        proj = _projection(extent, proj_lon, proj_lat)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.5 * ncols, 3.6 * nrows),
            subplot_kw={"projection": proj}, squeeze=False,
        )
        axes = np.atleast_1d(axes).ravel()
        mesh = None
        for i, (ax, member, (gx, gy, grid)) in enumerate(zip(axes, members, grids)):
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            mesh = ax.pcolormesh(gx, gy, grid, transform=ccrs.PlateCarree(), cmap=cmap,
                                 vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
            gl = add_geography(ax, label_size=6.5)
            if gl is not None:
                gl.left_labels = i % ncols == 0
                gl.bottom_labels = i + ncols >= len(members)
            peak = float(np.nanmax(grid)) if field == "value" else float(np.nanmax(np.abs(grid)))
            peak_word = "max" if field == "value" else "max |value|"
            ax.set_title(f"Member {member} ({peak_word} {peak:.1f} {unit})", fontsize=9)
        for ax in axes[len(members):]:
            ax.set_visible(False)
        # y above 1 keeps the three title lines clear of the first row of panel
        # titles; bbox_inches="tight" then crops back to the drawn extent.
        fig.suptitle(
            f"{title}\n{spec['subtitle']}{_fine_note(field, fine_cut_deg)}, {len(members)} members\n"
            f"init {init_dt:%Y-%m-%d %H} UTC, lead time {step} h, valid {valid_dt:%Y-%m-%d %H} UTC",
            y=1.06,
        )
        if mesh is not None:
            cbar = fig.colorbar(mesh, ax=axes.tolist(), orientation="horizontal",
                                pad=0.04, aspect=50, fraction=0.05,
                                extend=spec["extend"] if field == "value" else "both")
            cbar.set_label(cbar_label)
        save_figure(fig, out_path, close=True)


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

    # (key, lat, lon, resolution, [member label], [field per member])
    panels: list[tuple[str, np.ndarray, np.ndarray, float, list[int], list[np.ndarray]]] = []
    for i, (key, run_dir) in enumerate(runs.items()):
        pred_file = Path(run_dir).expanduser() / f"predictions_{args.date}_step{args.step:03d}.nc"
        if not pred_file.exists():
            raise SystemExit(f"{pred_file} not found (run {key!r}).")
        ds = xr.open_dataset(pred_file, decode_timedelta=False)
        states = [str(s) for s in ds["weather_state"].values]
        lat_h, lon_h = ds["lat_hres"].values, ds["lon_hres"].values
        available = [int(m) for m in np.asarray(ds["ensemble_member"].values).reshape(-1)]
        here = parse_members(args.members, available)
        if not members:
            members = here
        if i == 0:
            if not args.no_input:
                panels.append(("eefo", ds["lat_lres"].values, ds["lon_lres"].values,
                               DEFAULT_LRES_RES, here,
                               [_field(_member_slice(ds["x"], m), states, spec) for m in here]))
            if not args.no_truth:
                panels.append(("enfo", lat_h, lon_h, DEFAULT_HRES_RES, here,
                               [_field(_member_slice(ds["y"], m), states, spec) for m in here]))
        panels.append((key, lat_h, lon_h, DEFAULT_HRES_RES, here,
                       [_field(_member_slice(ds["y_pred"], m), states, spec) for m in here]))

    for key, lat, lon, res, member_labels, values in panels:
        out_path = out_dir / (
            f"{key}_{token}_init{args.date}_members{len(member_labels):02d}"
            f"_{args.region_tag}_f{args.step:03d}.png"
        )
        _render_grid(
            out_path=out_path, title=titles.get(key, default_title(key)),
            members=member_labels, values=values, date=args.date, time=args.time,
            step=args.step,
            lat=lat, lon=lon, res=res, extent=extent, spec=spec, vmin=vmin, vmax=vmax,
            proj_lon=args.proj_lon, proj_lat=args.proj_lat, ncols=int(args.grid_cols),
            field=args.field, fine_cut_deg=args.fine_cut_deg,
        )
        outputs.append(str(out_path))
        print(f"saved {out_path}", flush=True)

    write_manifest(out_root=out_dir, payload={
        "tool": "zoom_maps", "mode": "member_grid",
        "date": args.date, "time": args.time, "step": args.step, "members": members,
        "variable": args.variable, "field": args.field, "fine_cut_deg": args.fine_cut_deg,
        "runs": runs, "extent": list(extent), "vmin": vmin, "vmax": vmax,
        "outputs": outputs,
    }, filename=(
        f"zoom_maps_manifest_{token}_init{args.date}"
        f"_members{len(members):02d}_f{args.step:03d}.json"
    ))
    return 0


def run(args: argparse.Namespace) -> int:
    if getattr(args, "trajectory_npz", None):
        from .plot_trajectory_wind_maps import run as trajectory_run
        return trajectory_run(args)

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
            out_path=out_path, title=titles.get(key, default_title(key)),
            member=args.member, date=args.date, time=args.time, step=args.step,
            lat=lat, lon=lon, val=val, res=res, extent=extent, spec=spec,
            vmin=vmin, vmax=vmax,
            proj_lon=args.proj_lon, proj_lat=args.proj_lat,
            field=args.field, fine_cut_deg=args.fine_cut_deg,
        )
        outputs.append(str(out_path))
        print(f"saved {out_path}", flush=True)

    write_manifest(out_root=out_dir, payload={
        "tool": "zoom_maps",
        "date": args.date, "time": args.time, "step": args.step, "member": args.member,
        "variable": args.variable,
        "field": args.field, "fine_cut_deg": args.fine_cut_deg,
        "runs": runs, "gribs": gribs, "extent": list(extent),
        "vmin": vmin, "vmax": vmax,
        "outputs": outputs,
    }, filename=(
        f"zoom_maps_manifest_{token}_init{args.date}"
        f"_n{args.member:03d}_f{args.step:03d}.json"
    ))
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
