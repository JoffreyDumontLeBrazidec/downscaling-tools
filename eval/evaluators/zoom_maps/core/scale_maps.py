"""Zoom maps that pull large scales and fine scales apart, and the classic front view.

Two layouts of ``eval.cli zoom_maps`` (``--layout scales`` and ``--layout fronts``), both
for one date, lead, member and map box, from the retrieved ``predictions_*.nc`` files.
Diagnostic maps only: nothing is written back, and the downscaling outputs are never
filtered (owner's ruling, 30 Oct 2026, ablation note section 15.10). The split is how the
read looks at the two jobs the model does at once, never a change to the product.

``--layout scales`` (ablation note section 15.9, the TD-248 zoom maps)
    The target is an independent o1280 forecast, so truth and input drift apart at large
    scales with lead time, and the model's residual (model - input) carries some of that
    drift. Both residuals are split with the probabilistic evaluator's own low-pass ``P_L``
    (``eval/evaluators/probabilistic/core/scale_split.py``: an isotropic Gaussian in km,
    half power at ``--scale-cut-km``, 400 km by default): ``L = P_L(residual)`` is the
    large-scale change against the input, ``F = residual - L`` the fine detail.
    One figure of full fields, one row per variable (default 10 m wind speed and msl):
    input, truth, truth with its large-scale difference to the input removed
    (``input + F_truth``), model, model with its large-scale residual removed
    (``input + F_model``); msl isobars every 4 hPa on every panel. Then one residual
    figure per variable: rows truth and model, columns the full residual, its large part
    and its fine part, each column on one symmetric scale shared by the two rows. Titles
    give the rms of each panel over the map and, on the large column, the correlation of
    the model's large part with the truth's: near 1 means the model's large-scale change
    goes towards the truth (a correction), near 0 means it is drift of its own.
    The split is applied to the displayed scalar (wind speed, not its components), as the
    probabilistic evaluator splits ``10ff``. ``P_L`` is a taper, not a projection: about
    17 % of the power at 250 km sits in the large part, so a storm core is shared between
    the columns.

``--layout fronts`` (troughs and fronts, the Storm Ciaran read of 30 Oct 2026)
    Columns input, truth and one model panel per ``--run``; rows 10 m wind speed with
    wind arrows, the front temperature (``t_850`` when the file has it, else ``2t``), and
    the horizontal gradient of that temperature in K per 100 km, the frontal zone. msl
    isobars every 4 hPa on every panel. A cold front shows as a line of isobar kinks (the
    trough), a wind shift in the arrows and a band of strong temperature gradient on its
    cold side.

The input panels use ``x_interp`` (the o320 input on the o1280 points) when the file holds
it, otherwise the nearest o320 point of ``x``; the title says which.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from eval.shared.manifest import write_manifest

from .plot_member_wind_maps import (
    DEFAULT_HRES_RES,
    DEFAULT_TITLES,
    VARIABLES,
    _draw_layer,
    _field,
    _parse_kv,
    _projection,
    default_title,
    native_triangulation,
    nearest_grid,
    resolve_extent,
)

LAYOUTS = ("panels", "scales", "fronts")
DEFAULT_SCALE_ROWS = ("wind10m", "msl")
DEFAULT_SCALE_CUT_KM = 400.0
# Points read around the map box before the low-pass, in degrees of latitude: about 1100 km,
# twenty sigma of the 400 km cut, so P_L is never one-sided inside the map.
SCALE_MARGIN_DEG = 10.0
ISOBAR_STEP_HPA = 4.0
# Front gradient: regular grid spacing (degrees) and Gaussian smoothing (grid cells).
FRONT_GRID_DEG = 0.25
FRONT_SMOOTH_CELLS = 1.5
FRONT_ARROWS = 18              # arrows across the map width
KM_PER_DEG = 111.2


def add_layout_args(p: argparse.ArgumentParser) -> None:
    """The flags of the two layouts, added to the zoom_maps parser."""
    p.add_argument(
        "--layout", choices=LAYOUTS, default="panels",
        help="panels (default): one map per source for --variable. scales: full fields and "
             "residuals against the input split at --scale-cut-km into the large-scale change "
             "and the fine detail, for truth and model (analysis only, outputs are never "
             "filtered). fronts: wind with arrows, temperature and its gradient, with msl "
             "isobars, to read troughs and fronts.",
    )
    p.add_argument(
        "--scale-cut-km", type=float, default=DEFAULT_SCALE_CUT_KM,
        help="Half-power wavelength of the large-scale low-pass for --layout scales (default "
             "400 km, the probabilistic evaluator's split).",
    )
    p.add_argument(
        "--scale-rows", default=",".join(DEFAULT_SCALE_ROWS),
        help="Comma-separated variables of --layout scales, one row each (default wind10m,msl).",
    )
    p.add_argument(
        "--front-temperature", choices=("t_850", "2t"), default="t_850",
        help="Temperature of --layout fronts (default t_850; 2t when the file has no t_850).",
    )


# ------------------------------------------------------------------------- reading

def _select_member(da, member: int) -> np.ndarray:
    """(grid_point, weather_state) of one member; a single-member array serves every member."""
    d = da.isel(sample=0) if "sample" in da.dims else da
    if "ensemble_member" in d.dims:
        labels = [int(v) for v in np.asarray(d["ensemble_member"].values).reshape(-1)]
        d = d.sel(ensemble_member=member) if member in labels else d.isel(ensemble_member=0)
    arr = np.asarray(d.values, dtype=np.float64)
    if d.dims and d.dims[0] == "weather_state" and arr.ndim == 2:
        arr = arr.T
    return arr


def _unit_vectors(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def read_member(pred_file: Path, member: int, *, model_only: bool = False) -> dict:
    """lat, lon, states and the (point, state) arrays ``input``, ``truth``, ``model`` of one member."""
    import xarray as xr

    with xr.open_dataset(pred_file, decode_timedelta=False) as ds:
        states = [str(s) for s in ds["weather_state"].values]
        lat = np.asarray(ds["lat_hres"].values, dtype=np.float64).reshape(-1)
        lon = np.asarray(ds["lon_hres"].values, dtype=np.float64).reshape(-1)
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        out = {"lat": lat, "lon": lon, "states": states,
               "model": _select_member(ds["y_pred"], member)}
        if model_only:
            return out
        out["truth"] = _select_member(ds["y"], member)
        if "x_interp" in ds:
            out["input"] = _select_member(ds["x_interp"], member)
            out["input_note"] = "input = x_interp (o320 on the o1280 points)"
        else:
            from scipy.spatial import cKDTree

            lat_l = np.asarray(ds["lat_lres"].values, dtype=np.float64).reshape(-1)
            lon_l = np.asarray(ds["lon_lres"].values, dtype=np.float64).reshape(-1)
            _, idx = cKDTree(_unit_vectors(lat_l, lon_l)).query(_unit_vectors(lat, lon), workers=-1)
            out["input"] = _select_member(ds["x"], member)[idx]
            out["input_note"] = "input = nearest o320 point (the file has no x_interp)"
    return out


def subset_box(lat: np.ndarray, lon: np.ndarray, extent, margin_deg: float) -> np.ndarray:
    """Points inside the map box widened by ``margin_deg`` (longitude margin scaled by latitude)."""
    lon_min, lon_max, lat_min, lat_max = extent
    if lon_min > lon_max:
        raise SystemExit("--layout scales/fronts does not draw a box across the dateline.")
    # the poleward edge of the widened box needs the widest longitude margin
    poleward = min(max(abs(lat_min), abs(lat_max)) + margin_deg, 80.0)
    mlon = margin_deg / math.cos(math.radians(poleward))
    return ((lat >= lat_min - margin_deg) & (lat <= lat_max + margin_deg)
            & (lon >= lon_min - mlon) & (lon <= lon_max + mlon))


def in_box(lat: np.ndarray, lon: np.ndarray, extent) -> np.ndarray:
    lon_min, lon_max, lat_min, lat_max = extent
    return (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)


# --------------------------------------------------------------------- computation

def _rms(values: np.ndarray, mask: np.ndarray) -> float:
    v = values[mask]
    return float(np.sqrt(np.mean(np.square(v)))) if v.size else math.nan


def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Uncentred correlation over the mask: the area mean is part of a large-scale change."""
    a, b = a[mask], b[mask]
    den = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    return float(np.sum(a * b)) / den if den > 0 else math.nan


def split_residuals(inp: np.ndarray, truth: np.ndarray, model: np.ndarray, operator, mask: np.ndarray) -> dict:
    """The large/fine split of truth - input and model - input, and the numbers in the titles.

    ``operator`` is a ``LowPassOperator`` built for these points; ``mask`` marks the map.
    """
    rt, rm = truth - inp, model - inp
    lt, ft = operator.split(rt)
    lm, fm = operator.split(rm)
    fields = {
        "truth_full": rt, "truth_large": lt, "truth_fine": ft,
        "model_full": rm, "model_large": lm, "model_fine": fm,
        "truth_aligned": inp + ft, "model_aligned": inp + fm,
    }
    stats = {f"{k}_rms": _rms(v, mask) for k, v in fields.items() if not k.endswith("aligned")}
    stats["large_corr"] = _corr(lm, lt, mask)
    stats["fine_corr"] = _corr(fm, ft, mask)
    tl, tf = stats["truth_large_rms"] ** 2, stats["truth_fine_rms"] ** 2
    stats["truth_large_share"] = tl / (tl + tf) if (tl + tf) > 0 else math.nan
    stats["model_over_truth_large"] = (stats["model_large_rms"] / stats["truth_large_rms"]
                                       if stats["truth_large_rms"] > 0 else math.nan)
    return {"fields": fields, "stats": stats}


def temperature_gradient(lat: np.ndarray, lon: np.ndarray, values: np.ndarray, extent, *,
                         res: float = FRONT_GRID_DEG, smooth_cells: float = FRONT_SMOOTH_CELLS):
    """(grid lons, grid lats, |grad T| in K per 100 km) on a regular grid over the box.

    Nearest-neighbour regridding, a Gaussian of ``smooth_cells`` cells (the gradient of the
    raw o1280 field is mostly grid-scale noise and orography), then centred differences in km.
    """
    from scipy.ndimage import gaussian_filter

    gx, gy, grid = nearest_grid(lat, lon, values, extent=extent, res=res, margin=2.0)
    grid = gaussian_filter(grid, smooth_cells, mode="nearest")
    dy_km = res * KM_PER_DEG
    dx_km = res * KM_PER_DEG * np.maximum(np.cos(np.deg2rad(gy)), 0.05)[:, None]
    d_dy, d_dx = np.gradient(grid)
    return gx, gy, 100.0 * np.hypot(d_dx / dx_km, d_dy / dy_km)


# ------------------------------------------------------------------------ drawing

def _title_time(date: str, time: str, step: int) -> str:
    init_dt = datetime.strptime(date + time, "%Y%m%d%H%M")
    valid_dt = init_dt + timedelta(hours=step)
    return f"init {init_dt:%Y-%m-%d %H} UTC, lead time {step} h, valid {valid_dt:%Y-%m-%d %H} UTC"


def _axes_grid(fig, nrows: int, ncols: int, extent, proj):
    import cartopy.crs as ccrs

    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(nrows, ncols, i * ncols + j + 1, projection=proj)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            axes[i, j] = ax
    return axes


def _finish(ax, extent) -> None:
    import cartopy.crs as ccrs
    from eval.plotting import add_geography

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    add_geography(ax, coast_lw=0.7, label_size=6)


def _isobars(ax, tri, msl_hpa: np.ndarray) -> None:
    lo = ISOBAR_STEP_HPA * math.floor(np.nanmin(msl_hpa) / ISOBAR_STEP_HPA)
    hi = ISOBAR_STEP_HPA * math.ceil(np.nanmax(msl_hpa) / ISOBAR_STEP_HPA)
    if hi > lo:
        ax.tricontour(tri, msl_hpa, levels=np.arange(lo, hi + 0.1, ISOBAR_STEP_HPA),
                      colors="0.1", linewidths=0.5, zorder=5)


def _sym_limit(arrays: list[np.ndarray]) -> float:
    v = np.abs(np.concatenate([a.ravel() for a in arrays]))
    v = v[np.isfinite(v)]
    lim = float(np.percentile(v, 99.5)) if v.size else 1.0
    return lim if lim > 0 else 1.0


def _house(spec: dict):
    from eval.plotting import variable_spec

    return variable_spec(spec["house_key"])


def _out(out_dir: Path, key: str, kind: str, args) -> Path:
    return out_dir / f"{key}_{kind}_init{args.date}_n{args.member:03d}_{args.region_tag}_f{args.step:03d}.png"


def run_scales(args: argparse.Namespace) -> int:
    """--layout scales: full fields and the large/fine split of the residuals, truth and model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eval.evaluators.probabilistic.core.scale_split import LowPassOperator
    from eval.plotting import eval_style, save_figure

    runs = _parse_kv(args.run, "run")
    if not runs:
        raise SystemExit("--layout scales needs a --run predictions directory.")
    titles = {**DEFAULT_TITLES, **_parse_kv(args.title, "title")}
    rows = [v.strip() for v in str(args.scale_rows).split(",") if v.strip()]
    unknown = [v for v in rows if v not in VARIABLES]
    if unknown:
        raise SystemExit(f"--scale-rows: unknown variable(s) {unknown}; choose from {sorted(VARIABLES)}.")
    extent, box_line = resolve_extent(args)
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    proj = _projection(extent, args.proj_lon, args.proj_lat)
    cut = float(args.scale_cut_km)
    when = _title_time(args.date, args.time, args.step)

    outputs: list[str] = []
    stats_out: dict = {}
    for key, run_dir in runs.items():
        pred_file = Path(run_dir).expanduser() / f"predictions_{args.date}_step{args.step:03d}.nc"
        if not pred_file.exists():
            raise SystemExit(f"{pred_file} not found (run {key!r}).")
        data = read_member(pred_file, args.member)
        sub = subset_box(data["lat"], data["lon"], extent, SCALE_MARGIN_DEG)
        lat, lon = data["lat"][sub], data["lon"][sub]
        operator = LowPassOperator(lat, lon, None, cut_km=cut)
        mask = in_box(lat, lon, extent)
        states = data["states"]
        msl_spec = VARIABLES["msl"]
        has_msl = "msl" in states
        per_var: dict[str, dict] = {}
        for name in rows:
            spec = VARIABLES[name]
            inp, tru, mod = (_field(data[s][sub], states, spec) for s in ("input", "truth", "model"))
            per_var[name] = {"spec": spec, "input": inp, "truth": tru, "model": mod,
                             **split_residuals(inp, tru, mod, operator, mask)}
        isobars = {}
        if has_msl:
            m_in, m_tr, m_mo = (_field(data[s][sub], states, msl_spec) for s in ("input", "truth", "model"))
            split = split_residuals(m_in, m_tr, m_mo, operator, mask)["fields"]
            isobars = {"input": m_in, "truth": m_tr, "truth_aligned": split["truth_aligned"],
                       "model": m_mo, "model_aligned": split["model_aligned"]}
        model_title = titles.get(key, default_title(key))
        cols = [("input", f"{titles['eefo']}\n{data['input_note'].split(' = ')[1]}"),
                ("truth", "Truth (o1280)"),
                ("truth_aligned", f"Truth, difference to the input\nabove {cut:g} km removed"),
                ("model", model_title),
                ("model_aligned", f"{model_title},\nresidual above {cut:g} km removed")]

        # full fields: one row per variable
        with eval_style():
            fig = plt.figure(figsize=(4.2 * len(cols), 3.9 * len(rows) + 1.2))
            axes = _axes_grid(fig, len(rows), len(cols), extent, proj)
            iso_keys = list(isobars)
            for i, name in enumerate(rows):
                v = per_var[name]
                arrays = {"input": v["input"], "truth": v["truth"], "model": v["model"],
                          "truth_aligned": v["fields"]["truth_aligned"],
                          "model_aligned": v["fields"]["model_aligned"]}
                values = [arrays[c] for c, _ in cols] + [isobars[k] for k in iso_keys]
                tri, vals = native_triangulation(axes[i, 0], lat, lon, values, extent=extent, field="value",
                                                 fine_cut_deg=0.6, res=DEFAULT_HRES_RES)
                drawn = dict(zip([c for c, _ in cols] + [f"iso_{k}" for k in iso_keys], vals))
                house = _house(v["spec"])
                for j, (c, label) in enumerate(cols):
                    ax = axes[i, j]
                    mesh = _draw_layer(ax, ("tri", tri, drawn[c]), cmap=house.field_cmap(),
                                       vmin=v["spec"]["vmin"], vmax=v["spec"]["vmax"])
                    if f"iso_{c}" in drawn:
                        _isobars(ax, tri, drawn[f"iso_{c}"])
                    _finish(ax, extent)
                    box = mask
                    ax.set_title((f"{label}\n" if i == 0 else "")
                                 + f"min {arrays[c][box].min():.1f}, max {arrays[c][box].max():.1f}", fontsize=9)
                fig.colorbar(mesh, ax=list(axes[i]), orientation="vertical", shrink=0.85, pad=0.01,
                             extend=v["spec"]["extend"]).set_label(v["spec"]["cbar_label"])
            fig.suptitle(f"Full fields, {model_title}, member {args.member}; {when}\n{box_line}; "
                         + ("msl isobars every 4 hPa; " if has_msl else "")
                         + "analysis only: the model outputs are never filtered", fontsize=11)
            path = _out(out_dir, key, "scales-full", args)
            save_figure(fig, path, close=True)
            outputs.append(str(path))
            print(f"saved {path}", flush=True)

        # residuals: one figure per variable, rows truth / model, columns full / large / fine
        for name in rows:
            v = per_var[name]
            f, s = v["fields"], v["stats"]
            house = _house(v["spec"])
            parts = [("full", "residual against the input"), ("large", f"large part (above {cut:g} km)"),
                     ("fine", f"fine part (below {cut:g} km)")]
            with eval_style():
                fig = plt.figure(figsize=(13.5, 8.6))
                axes = _axes_grid(fig, 2, 3, extent, proj)
                values = [f[f"{who}_{part}"] for who in ("truth", "model") for part, _ in parts]
                tri, vals = native_triangulation(axes[0, 0], lat, lon, values, extent=extent, field="value",
                                                 fine_cut_deg=0.6, res=DEFAULT_HRES_RES)
                drawn = dict(zip([f"{who}_{part}" for who in ("truth", "model") for part, _ in parts], vals))
                for j, (part, label) in enumerate(parts):
                    lim = _sym_limit([f[f"truth_{part}"][mask], f[f"model_{part}"][mask]])
                    for i, who in enumerate(("truth", "model")):
                        ax = axes[i, j]
                        mesh = _draw_layer(ax, ("tri", tri, drawn[f"{who}_{part}"]), cmap="RdBu_r",
                                           vmin=-lim, vmax=lim)
                        _finish(ax, extent)
                        extra = ""
                        if i == 1 and part in ("large", "fine"):
                            extra = f", corr with truth {s[f'{part}_corr']:.2f}"
                        head = "Truth - input" if who == "truth" else "Model - input"
                        ax.set_title((f"{label}\n" if i == 0 else "")
                                     + f"{head}: rms {s[f'{who}_{part}_rms']:.2f} {house.unit}{extra}",
                                     fontsize=9)
                    fig.colorbar(mesh, ax=list(axes[:, j]), orientation="horizontal", shrink=0.85, pad=0.04,
                                 extend="both").set_label(f"{house.name} difference ({house.unit})")
                fig.suptitle(
                    f"{house.name}, {model_title}: large-scale change and fine detail against the input, "
                    f"member {args.member}; "
                    f"{when}\n{box_line}; split by the probabilistic evaluator's low-pass (half power at "
                    f"{cut:g} km); truth large share {s['truth_large_share']:.2f}, model/truth large rms "
                    f"{s['model_over_truth_large']:.2f}", fontsize=10)
                path = _out(out_dir, key, f"scales-resid-{v['spec']['token']}", args)
                save_figure(fig, path, close=True)
                outputs.append(str(path))
                print(f"saved {path}", flush=True)
            stats_out.setdefault(key, {})[name] = {k: round(float(x), 6) for k, x in s.items()}

    write_manifest(out_root=out_dir, payload={
        "tool": "zoom_maps", "layout": "scales", "date": args.date, "time": args.time, "step": args.step,
        "member": args.member, "rows": rows, "scale_cut_km": cut, "runs": runs, "extent": list(extent),
        "stats": stats_out, "outputs": outputs,
        "note": "analysis only; the model outputs are never filtered (ablation note s15.10)",
    }, filename=f"zoom_maps_manifest_scales_init{args.date}_n{args.member:03d}_f{args.step:03d}.json")
    return 0


def run_fronts(args: argparse.Namespace) -> int:
    """--layout fronts: wind and arrows, temperature, temperature gradient, with isobars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from eval.plotting import eval_style, save_figure

    runs = _parse_kv(args.run, "run")
    if not runs:
        raise SystemExit("--layout fronts needs a --run predictions directory.")
    titles = {**DEFAULT_TITLES, **_parse_kv(args.title, "title")}
    extent, box_line = resolve_extent(args)
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    proj = _projection(extent, args.proj_lon, args.proj_lat)
    when = _title_time(args.date, args.time, args.step)

    panels: list[tuple[str, str, np.ndarray]] = []    # (key, title, (point, state) array)
    lat = lon = states = None
    input_note = ""
    for i, (key, run_dir) in enumerate(runs.items()):
        pred_file = Path(run_dir).expanduser() / f"predictions_{args.date}_step{args.step:03d}.nc"
        if not pred_file.exists():
            raise SystemExit(f"{pred_file} not found (run {key!r}).")
        data = read_member(pred_file, args.member, model_only=i > 0)
        if i == 0:
            lat, lon, states = data["lat"], data["lon"], data["states"]
            sub = subset_box(lat, lon, extent, 2.0)
            lat, lon = lat[sub], lon[sub]
            input_note = data["input_note"]
            panels.append(("input", titles["eefo"], data["input"][sub]))
            panels.append(("truth", "Truth (o1280)", data["truth"][sub]))
        elif data["states"] != states or data["lat"].size != sub.size:
            raise SystemExit(f"run {key!r}: its file has another grid or other weather states than the first run.")
        panels.append((key, titles.get(key, default_title(key)), data["model"][sub]))

    temp_name = args.front_temperature if args.front_temperature in states else "2t"
    for need in ("10u", "10v", "msl", temp_name):
        if need not in states:
            raise SystemExit(f"--layout fronts needs {need!r}; the file carries {states}.")
    ws_spec, msl_spec, t_spec = VARIABLES["wind10m"], VARIABLES["msl"], VARIABLES[temp_name]
    ui, vi = states.index("10u"), states.index("10v")

    grads = [temperature_gradient(lat, lon, _field(arr, states, t_spec), extent) for _, _, arr in panels]
    gmax = float(np.nanpercentile(np.concatenate([g[2].ravel() for g in grads]), 99.5))
    gmax = gmax if gmax > 0 else 1.0
    t_house, ws_house = _house(t_spec), _house(ws_spec)

    with eval_style():
        ncols = len(panels)
        fig = plt.figure(figsize=(4.6 * ncols + 1.0, 12.0))
        axes = _axes_grid(fig, 3, ncols, extent, proj)
        for j, (key, label, arr) in enumerate(panels):
            ws = _field(arr, states, ws_spec)
            temp = _field(arr, states, t_spec)
            msl = _field(arr, states, msl_spec)
            tri, (ws_v, t_v, msl_v) = native_triangulation(axes[0, j], lat, lon, [ws, temp, msl], extent=extent,
                                                           field="value", fine_cut_deg=0.6, res=DEFAULT_HRES_RES)
            m0 = _draw_layer(axes[0, j], ("tri", tri, ws_v), cmap=ws_house.field_cmap(),
                             vmin=ws_spec["vmin"], vmax=ws_spec["vmax"])
            # wind arrows on a coarse regular grid
            step_deg = (extent[1] - extent[0]) / FRONT_ARROWS
            gx, gy, ug = nearest_grid(lat, lon, arr[:, ui], extent=extent, res=step_deg, margin=0.0)
            _, _, vg = nearest_grid(lat, lon, arr[:, vi], extent=extent, res=step_deg, margin=0.0)
            gxx, gyy = np.meshgrid(gx, gy)
            q = axes[0, j].quiver(gxx, gyy, ug, vg, transform=ccrs.PlateCarree(), scale=20.0 * FRONT_ARROWS,
                                  scale_units="width", width=0.0025, color="white", zorder=7)
            m1 = _draw_layer(axes[1, j], ("tri", tri, t_v), cmap=t_house.field_cmap(),
                             vmin=t_spec["vmin"], vmax=t_spec["vmax"])
            ggx, ggy, grad = grads[j]
            m2 = axes[2, j].pcolormesh(ggx, ggy, grad, transform=ccrs.PlateCarree(), cmap="YlOrRd",
                                       vmin=0.0, vmax=gmax, shading="auto", rasterized=True)
            for i in range(3):
                _isobars(axes[i, j], tri, msl_v)
                _finish(axes[i, j], extent)
            note = f"\n{input_note.split(' = ')[1]}" if key == "input" else ""
            axes[0, j].set_title(f"{label}{note}", fontsize=9)
        axes[0, -1].quiverkey(q, 0.85, -0.08, 20.0, "20 m s⁻¹", labelpos="E", coordinates="axes")
        fig.colorbar(m0, ax=list(axes[0]), shrink=0.85, pad=0.01, extend="max").set_label(ws_spec["cbar_label"])
        fig.colorbar(m1, ax=list(axes[1]), shrink=0.85, pad=0.01, extend="both").set_label(t_spec["cbar_label"])
        fig.colorbar(m2, ax=list(axes[2]), shrink=0.85, pad=0.01, extend="max").set_label(
            f"{t_house.name} gradient (K per 100 km)")
        fig.suptitle(f"Troughs and fronts, member {args.member}; {when}\n{box_line}; msl isobars every 4 hPa; "
                     f"gradient on a {FRONT_GRID_DEG:g}° grid smoothed over {FRONT_SMOOTH_CELLS:g} cells",
                     fontsize=11)
        path = _out(out_dir, "+".join(runs), "fronts", args)
        save_figure(fig, path, close=True)
        print(f"saved {path}", flush=True)

    write_manifest(out_root=out_dir, payload={
        "tool": "zoom_maps", "layout": "fronts", "date": args.date, "time": args.time, "step": args.step,
        "member": args.member, "temperature": temp_name, "runs": runs, "extent": list(extent),
        "gradient_vmax_k_per_100km": gmax, "outputs": [str(path)],
    }, filename=f"zoom_maps_manifest_fronts_init{args.date}_n{args.member:03d}_f{args.step:03d}.json")
    return 0
