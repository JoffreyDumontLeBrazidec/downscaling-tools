from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy import crs
from matplotlib.backends.backend_pdf import PdfPages

try:
    from .tc_events import EVENTS, TCEvent
    from .tc_vector_loading import (
        _event_point_mask,
        _prediction_point_coordinates,
        discover_prediction_files,
        normalize_lon,
        select_prediction_files_for_event,
    )
except ImportError:  # allow running as a script
    from tc_events import EVENTS, TCEvent
    from tc_vector_loading import (
        _event_point_mask,
        _prediction_point_coordinates,
        discover_prediction_files,
        normalize_lon,
        select_prediction_files_for_event,
    )

LOG = logging.getLogger(__name__)


CASES = [
    {
        "name": "dora",
        "lat": (5.5, 23.0),
        "lon": (-158.75, -141.0),
        "dates": [6],
        "time": 1,
        "msl_levels": np.linspace(985, 1015, 31),
        "wind_levels": np.linspace(0, 30, 31),
    },
    {
        "name": "fernanda",
        "lat": (1.0, 29.0),
        "lon": (-134.0, -105.0),
        "dates": [13],
        "time": 2,
        "msl_levels": np.linspace(985, 1015, 31),
        "wind_levels": np.linspace(0, 30, 31),
    },
    {
        "name": "hilary",
        "lat": (11.0, 29.0),
        "lon": (-120.0, -100.0),
        "dates": [17],
        "time": 1,
        "msl_levels": np.linspace(965, 1015, 31),
        "wind_levels": np.linspace(0, 35, 31),
    },
    {
        "name": "idalia",
        "lat": (11.0, 39.01),
        "lon": (-99.0, -71.01),
        "dates": [28],
        "time": 1,
        "msl_levels": np.linspace(985, 1015, 31),
        "wind_levels": np.linspace(0, 30, 31),
    },
    {
        "name": "franklin",
        "lat": (11.0, 39.0),
        "lon": (-79.0, -51.0),
        "dates": [28],
        "time": 1,
        "msl_levels": np.linspace(985, 1015, 31),
        "wind_levels": np.linspace(0, 30, 31),
    },
]


def _parse_members(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _candidate_expids(expver: str) -> list[str]:
    out = [expver]
    if not expver.startswith("ENFO_O320_"):
        out.insert(0, f"ENFO_O320_{expver}")
    # de-duplicate while preserving order
    seen = set()
    uniq: list[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def _resolve_expid_for_case(*, base_dir: str, case_name: str, year: str, month: str, day: int, expver: str) -> str:
    for expid in _candidate_expids(expver):
        p = Path(base_dir) / case_name / f"surface_pf_{expid}_{year}{month}{int(day):02d}.grib"
        if p.exists():
            return expid
    return expver


def _plot_member_data_legacy(
    *,
    data_dict: dict[str, np.ndarray],
    lon2: np.ndarray,
    lat2: np.ndarray,
    levels: np.ndarray,
    title: str,
    filename: str,
    members: list[int],
    time_index: int,
    day: int,
    out_dir: Path,
) -> None:
    datasets = [(data, key) for key, data in data_dict.items()]
    ncols, nrows = len(datasets), len(members)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(8 * ncols, 6 * nrows),
        subplot_kw={"projection": crs.PlateCarree()},
    )
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.92, wspace=-0.1, hspace=0.15)
    axs = np.array(axs).reshape(nrows, ncols)

    images = []
    for row, member in enumerate(members):
        for col, (data, label) in enumerate(datasets):
            ax = axs[row, col]
            arr = data[day, member, time_index][10:100, 10:100]
            im = ax.pcolormesh(
                lon2[10:100, 10:100],
                lat2[10:100, 10:100],
                arr,
                transform=crs.PlateCarree(),
                vmin=float(levels[0]),
                vmax=float(levels[-1]),
                shading="gouraud",
                cmap="viridis",
            )
            ax.contour(
                lon2[10:100, 10:100],
                lat2[10:100, 10:100],
                arr,
                transform=crs.PlateCarree(),
                levels=levels,
                colors="black",
                linewidths=0.5,
            )
            ax.set_title(f"mbr{member} {label}", fontsize=12)
            ax.coastlines()
            ax.grid(color="white", linestyle="--", linewidth=0.5)
            gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
            gl.top_labels = False
            gl.right_labels = False
            images.append(im)

    for row in range(nrows):
        cbar = fig.colorbar(
            images[row * ncols],
            ax=axs[row],
            orientation="vertical",
            shrink=0.8,
            pad=0.02,
        )
        cbar.ax.set_ylabel("")

    fig.suptitle(title, fontsize=14)
    out_path = out_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_retriever(case: dict, base_dir: str, year: str, month: str, resol: float) -> Any:
    # Import lazily so the module can be imported in test environments
    # where the metview executable is not available.
    from .tools.loading_data import DataRetriever

    lat = np.arange(case["lat"][0], case["lat"][1] + resol / 4, resol)
    lon = np.arange(case["lon"][0], case["lon"][1] + resol / 4, resol)
    return DataRetriever(
        f"{base_dir}/{case['name']}",
        case["dates"],
        year,
        month,
        resol,
        float(lat.min()),
        float(lat.max()),
        float(lon.min()),
        float(lon.max()),
    )


def run_tc_member_plots(
    *,
    expver: str,
    outdir: str,
    base_dir: str = "/home/ecm5702/hpcperm/data/tc",
    members: list[int] | None = None,
    year: str = "2023",
    month: str = "08",
    resol: float = 0.25,
) -> list[str]:
    members = members or [1, 2, 5, 7, 9]
    out_root = Path(outdir).expanduser().resolve() / "tc_members"
    out_root.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    analysis = "OPER_O320_0001"
    expid_enfo_o320 = "ENFO_O320_0001"
    expid_eefo_o96 = "EEFO_O96_0001"

    for case in CASES:
        retriever = _build_retriever(case, base_dir, year, month, resol)
        ml_expid = _resolve_expid_for_case(
            base_dir=base_dir,
            case_name=case["name"],
            year=year,
            month=month,
            day=case["dates"][0],
            expver=expver,
        )
        try:
            msl, wind10m = retriever.retrieve_all_data(
                analysis,
                expid_enfo_o320,
                expid_eefo_o96,
                [ml_expid],
            )
        except Exception as exc:
            LOG.warning("Skipping case %s: %s", case["name"], exc)
            continue

        lat = np.arange(case["lat"][0], case["lat"][1] + resol / 4, resol)
        lon = np.arange(case["lon"][0], case["lon"][1] + resol / 4, resol)
        lon2, lat2 = np.meshgrid(lon, lat)

        pf_keys = [k for k in msl.keys() if "OPER" not in k]
        if not pf_keys:
            LOG.warning("Skipping case %s: no ensemble datasets found", case["name"])
            continue
        max_member = min(int(msl[k].shape[1]) - 1 for k in pf_keys)
        safe_members = [m for m in members if 0 <= m <= max_member]
        if not safe_members:
            LOG.warning("Skipping case %s: no valid members in %s", case["name"], members)
            continue

        t_idx = int(case["time"])
        max_time = min(int(msl[k].shape[2]) for k in pf_keys) - 1
        if t_idx > max_time:
            LOG.warning("Skipping case %s: time index %s out of bounds", case["name"], t_idx)
            continue

        msl_data = {k: v for k, v in msl.items() if "OPER" not in k}
        wind_data = {k: v for k, v in wind10m.items() if "OPER" not in k}
        if not msl_data or not wind_data:
            LOG.warning("Skipping case %s: no non-OPER datasets available", case["name"])
            continue

        msl_name = f"{case['name']}_msl_fields_{case['dates'][0]}_{month}_step{t_idx}.png"
        wind_name = f"{case['name']}_wind10m_fields_{case['dates'][0]}_{month}_step{t_idx}.png"
        _plot_member_data_legacy(
            data_dict=msl_data,
            lon2=lon2,
            lat2=lat2,
            levels=case["msl_levels"],
            title="Mean Sea Level Pressure",
            filename=msl_name,
            members=safe_members,
            time_index=t_idx,
            day=0,
            out_dir=out_root,
        )
        _plot_member_data_legacy(
            data_dict=wind_data,
            lon2=lon2,
            lat2=lat2,
            levels=case["wind_levels"],
            title=f"Wind Speed 10m - {month} Lead Time {t_idx}",
            filename=wind_name,
            members=safe_members,
            time_index=t_idx,
            day=0,
            out_dir=out_root,
        )
        generated.extend([str(out_root / msl_name), str(out_root / wind_name)])
        LOG.info("Generated TC member plots for %s", case["name"])

    return generated


def _load_prediction_member_fields(
    nc_path: Path,
    cfg: TCEvent,
    members: list[int],
    regrid_resolution: float | None = None,
) -> dict[str, np.ndarray]:
    """Load input, prediction, and truth fields for selected members, masked to event bbox.

    Handles both structured (regular lat/lon) and unstructured (reduced Gaussian)
    grids. Unstructured grids are interpolated onto a regular grid within the bbox.

    Returns dict with keys: x_interp_msl, x_interp_wind, y_pred_msl, y_pred_wind,
    y_msl, y_wind, lat_axis, lon_axis.
    Arrays are shaped [len(members), nlat, nlon].
    """
    from scipy.interpolate import griddata

    with xr.open_dataset(nc_path) as ds:
        weather_states = ds["weather_state"].values.tolist()
        i_msl = weather_states.index("msl")
        i_u10 = weather_states.index("10u")
        i_v10 = weather_states.index("10v")

        lon_flat, lat_flat = _prediction_point_coordinates(ds)

        # Extract x_interp, y_pred, and y, selecting sample=0 and requested members
        # Shape after isel: [ensemble_member, grid_point_hres, weather_state]
        y_pred_raw = np.asarray(ds["y_pred"].isel(sample=0).values, dtype=np.float64)[members]
        y_raw = np.asarray(ds["y"].isel(sample=0).values, dtype=np.float64)[members]
        # x_interp is on the same hres grid as y_pred
        if "x_interp" in ds:
            x_interp_raw = np.asarray(ds["x_interp"].isel(sample=0).values, dtype=np.float64)[members]
            x_lres_raw = None
            x_lon_lres = None
            x_lat_lres = None
        elif "x" in ds:
            # Fallback: use x on the lres grid (different coordinates)
            x_interp_raw = None
            x_lres_raw = np.asarray(ds["x"].isel(sample=0).values, dtype=np.float64)[members]
            lon_lres_da = ds["lon_lres"]
            lat_lres_da = ds["lat_lres"]
            # Multi-ds predictions store lon/lat per-member; take first member.
            if lon_lres_da.ndim == 2:
                lon_lres_da = lon_lres_da.isel({lon_lres_da.dims[0]: 0})
                lat_lres_da = lat_lres_da.isel({lat_lres_da.dims[0]: 0})
            x_lon_lres = normalize_lon(np.asarray(lon_lres_da.values, dtype=np.float64))
            x_lat_lres = np.asarray(lat_lres_da.values, dtype=np.float64)
        else:
            x_interp_raw = None
            x_lres_raw = None
            x_lon_lres = None
            x_lat_lres = None

    # Mask to event bounding box points (hres grid)
    mask = _event_point_mask(lon_flat, lat_flat, cfg)
    if not np.any(mask):
        raise RuntimeError(f"No grid points inside event bbox for {cfg.name}")

    lon_bbox = lon_flat[mask]
    lat_bbox = lat_flat[mask]
    y_pred_bbox = y_pred_raw[:, mask, :]
    y_bbox = y_raw[:, mask, :]
    x_interp_bbox = x_interp_raw[:, mask, :] if x_interp_raw is not None else None

    # Mask lres input to event bbox (separate coordinate system)
    if x_lres_raw is not None:
        lres_mask = _event_point_mask(x_lon_lres, x_lat_lres, cfg)
        if np.any(lres_mask):
            x_lres_bbox = x_lres_raw[:, lres_mask, :]
            x_lon_bbox = x_lon_lres[lres_mask]
            x_lat_bbox = x_lat_lres[lres_mask]
        else:
            x_lres_bbox = None
    else:
        x_lres_bbox = None

    # Build regular target grid within bbox
    res = regrid_resolution or cfg.regrid_resolution
    west = normalize_lon(np.asarray([cfg.area_west], dtype=np.float64))[0]
    east = normalize_lon(np.asarray([cfg.area_east], dtype=np.float64))[0]
    lon_axis = np.arange(west, east + res / 4, res)
    lat_axis = np.arange(cfg.area_south, cfg.area_north + res / 4, res)
    target_lon, target_lat = np.meshgrid(lon_axis, lat_axis)

    def _interp_to_grid(flat_vals: np.ndarray) -> np.ndarray:
        """Interpolate [n_points] -> [nlat, nlon] via linear griddata."""
        return griddata(
            (lon_bbox, lat_bbox),
            flat_vals,
            (target_lon, target_lat),
            method="linear",
        )

    def _fields_for(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (msl[nmembers,nlat,nlon], wind[nmembers,nlat,nlon])."""
        msl_list, wind_list = [], []
        for mi in range(raw.shape[0]):
            msl_grid = _interp_to_grid(raw[mi, :, i_msl] / 100.0)
            u10_grid = _interp_to_grid(raw[mi, :, i_u10])
            v10_grid = _interp_to_grid(raw[mi, :, i_v10])
            wind_grid = np.sqrt(u10_grid ** 2 + v10_grid ** 2)
            msl_list.append(msl_grid)
            wind_list.append(wind_grid)
        return np.array(msl_list), np.array(wind_list)

    pred_msl, pred_wind = _fields_for(y_pred_bbox)
    truth_msl, truth_wind = _fields_for(y_bbox)

    result = {
        "y_pred_msl": pred_msl,
        "y_pred_wind": pred_wind,
        "y_msl": truth_msl,
        "y_wind": truth_wind,
        "lat_axis": lat_axis,
        "lon_axis": lon_axis,
    }

    if x_interp_bbox is not None:
        input_msl, input_wind = _fields_for(x_interp_bbox)
        result["x_interp_msl"] = input_msl
        result["x_interp_wind"] = input_wind
    elif x_lres_bbox is not None:
        # Interpolate lres input using its own coordinates
        def _interp_lres(flat_vals: np.ndarray) -> np.ndarray:
            return griddata(
                (x_lon_bbox, x_lat_bbox),
                flat_vals,
                (target_lon, target_lat),
                method="linear",
            )

        def _fields_lres(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            msl_list, wind_list = [], []
            for mi in range(raw.shape[0]):
                msl_grid = _interp_lres(raw[mi, :, i_msl] / 100.0)
                u10_grid = _interp_lres(raw[mi, :, i_u10])
                v10_grid = _interp_lres(raw[mi, :, i_v10])
                wind_grid = np.sqrt(u10_grid ** 2 + v10_grid ** 2)
                msl_list.append(msl_grid)
                wind_list.append(wind_grid)
            return np.array(msl_list), np.array(wind_list)

        input_msl, input_wind = _fields_lres(x_lres_bbox)
        result["x_interp_msl"] = input_msl
        result["x_interp_wind"] = input_wind

    return result


def _select_projection(cfg: TCEvent):
    """Pick LambertConformal for non-dateline-crossing bboxes, PlateCarree otherwise."""
    west = cfg.area_west
    east = cfg.area_east
    crosses_dateline = west > east  # e.g. 170 to -170
    if crosses_dateline:
        return crs.PlateCarree()
    central_lon = (west + east) / 2.0
    central_lat = (cfg.area_south + cfg.area_north) / 2.0
    return crs.LambertConformal(
        central_longitude=central_lon,
        central_latitude=central_lat,
    )


def _source_labels(cfg: TCEvent) -> tuple[str, str]:
    """Derive descriptive labels for input and target columns from event config."""
    def _fmt(expid: str) -> str:
        return expid.rsplit("_", 1)[0].replace("_", " ")  # "ENFO_O48_0001" → "ENFO O48"

    if cfg.iekm_target_grib_path:
        # Deterministic target lane (o48→o96, o1280→o2560)
        input_src = _fmt(cfg.expid_enfo_o320)  # e.g. "ENFO O48"
        analysis_res = cfg.analysis.split("_")[1]  # e.g. "O96"
        target_src = f"IEKM {analysis_res}"
    else:
        # Ensemble target lane (o96→o320, o320→o1280)
        input_src = _fmt(cfg.expid_eefo_o96)  # e.g. "EEFO O96"
        target_src = _fmt(cfg.expid_enfo_o320)  # e.g. "ENFO O320"
    return input_src, target_src


def _plot_member_page(
    fields: dict[str, np.ndarray],
    *,
    cfg: TCEvent,
    member_idx: int,
    member_label: int,
    step_hours: int,
    date_str: str,
    display_label: str,
) -> plt.Figure:
    """Create a 2x3 grid for one member: rows=[MSLP, Wind], cols=[Input, Prediction, Target]."""
    import cartopy.feature as cfeature
    import cmcrameri.cm as cmc
    from matplotlib.gridspec import GridSpec

    proj = _select_projection(cfg)
    input_src, target_src = _source_labels(cfg)

    bbox_aspect = abs(cfg.area_east - cfg.area_west) / (cfg.area_north - cfg.area_south)
    fig_height = 13.0 if bbox_aspect <= 1.0 else 13.0 / bbox_aspect

    fig = plt.figure(figsize=(18, fig_height))
    gs = GridSpec(
        2, 3,
        hspace=0.35, wspace=0.12,
        left=0.08, right=0.97, bottom=0.08, top=0.91,
    )

    map_axes = np.empty((2, 3), dtype=object)
    for row_i in range(2):
        for col_i in range(3):
            map_axes[row_i, col_i] = fig.add_subplot(gs[row_i, col_i], projection=proj)

    lat = fields["lat_axis"]
    lon = fields["lon_axis"]

    col_defs = [
        ("x_interp", f"{input_src} (input)"),
        ("y_pred", "Prediction (downscaled input)"),
        ("y", f"{target_src} (target)"),
    ]

    row_defs = [
        ("msl", "MSLP", cmc.vik, "hPa"),
        ("wind", "10m Wind Speed", cmc.batlow, "m/s"),
    ]

    # Shared color limits
    msl_keys = [k for k in ["x_interp_msl", "y_pred_msl", "y_msl"] if k in fields]
    wind_keys = [k for k in ["x_interp_wind", "y_pred_wind", "y_wind"] if k in fields]
    if cfg.member_map_msl_range is not None:
        msl_vmin, msl_vmax = cfg.member_map_msl_range
    else:
        msl_vmin = float(np.nanmin([np.nanmin(fields[k][member_idx]) for k in msl_keys]))
        msl_vmax = float(np.nanmax([np.nanmax(fields[k][member_idx]) for k in msl_keys]))
    if cfg.member_map_wind_range is not None:
        wind_vmin, wind_vmax = cfg.member_map_wind_range
    else:
        wind_vmin = 0.0
        wind_vmax = float(np.nanmax([np.nanmax(fields[k][member_idx]) for k in wind_keys]))
    var_vlims = {"msl": (msl_vmin, msl_vmax), "wind": (wind_vmin, wind_vmax)}

    row_images = {}

    for row_i, (var_suffix, row_label, cmap, unit) in enumerate(row_defs):
        vmin, vmax = var_vlims[var_suffix]
        contour_levels = np.linspace(vmin, vmax, 12)

        for col_i, (src_prefix, col_title) in enumerate(col_defs):
            ax = map_axes[row_i, col_i]
            field_key = f"{src_prefix}_{var_suffix}"

            if field_key not in fields:
                ax.text(
                    0.5, 0.5, "N/A",
                    transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, color="gray",
                )
                if row_i == 0:
                    ax.set_title(col_title, fontsize=13)
                ax.coastlines(linewidth=0.5)
                continue

            arr = fields[field_key][member_idx]

            im = ax.pcolormesh(
                lon, lat, arr,
                transform=crs.PlateCarree(),
                vmin=vmin, vmax=vmax,
                shading="nearest",
                cmap=cmap,
            )
            try:
                ax.contour(
                    lon, lat, arr,
                    transform=crs.PlateCarree(),
                    levels=contour_levels,
                    colors="black",
                    linewidths=0.4,
                    alpha=0.6,
                )
            except Exception:
                pass

            if row_i not in row_images:
                row_images[row_i] = im

            ax.coastlines(linewidth=0.6)
            try:
                ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray")
            except Exception:
                pass

            ax.set_extent(
                [cfg.area_west, cfg.area_east, cfg.area_south, cfg.area_north],
                crs=crs.PlateCarree(),
            )

            gl = ax.gridlines(
                draw_labels=True, dms=False,
                x_inline=False, y_inline=False,
                linewidth=0.3, alpha=0.5,
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"rotation": 0, "fontsize": 11, "va": "top"}
            gl.ylabel_style = {"fontsize": 11}
            gl.xpadding = 8
            if col_i > 0:
                gl.left_labels = False

            if row_i == 0:
                ax.set_title(col_title, fontsize=14)
            if col_i == 0:
                ax.text(
                    -0.14, 0.5, row_label,
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=14, rotation=90,
                )

    # Colorbars: positioned dynamically relative to each row's actual rendered bounds
    fig.canvas.draw()
    for row_i, (var_suffix, row_label, cmap, unit) in enumerate(row_defs):
        if row_i not in row_images:
            continue
        pos0 = map_axes[row_i, 0].get_position()  # leftmost axis
        pos2 = map_axes[row_i, 2].get_position()  # rightmost axis
        row_width = pos2.x1 - pos0.x0
        cbar_w = row_width * 0.6
        cbar_x = pos0.x0 + row_width * 0.2
        cbar_y = pos0.y0 - 0.05
        cax = fig.add_axes([cbar_x, cbar_y, cbar_w, 0.015])
        cbar = fig.colorbar(row_images[row_i], cax=cax, orientation="horizontal")
        cbar.set_label(unit, fontsize=13)
        cbar.ax.tick_params(labelsize=12)

    fig.suptitle(
        f"TC {cfg.name.capitalize()} | {display_label} | {date_str} +{step_hours}h | member {member_label}",
        fontsize=16, y=0.96,
    )
    return fig


def run_tc_member_plots_from_predictions(
    *,
    predictions_dir: str,
    outdir: str,
    run_label: str,
    display_label: str | None = None,
    event_names: list[str] | None = None,
    date: str,
    steps: list[int] | None = None,
    members: list[int] | None = None,
) -> list[str]:
    """Generate per-member spatial maps from prediction NC files.

    One multi-page PDF per event. Each page = one member for one step (2x3 grid).
    Pages ordered: step1/mbr0, step1/mbr1, ..., step2/mbr0, step2/mbr1, ...
    """
    display_label = display_label or run_label
    steps = steps or [24, 120]
    members = members or [0, 1, 2, 3, 4]

    pred_dir = Path(predictions_dir).expanduser().resolve()
    out_dir = Path(outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = discover_prediction_files(pred_dir)
    if not pred_files:
        raise FileNotFoundError(f"No predictions_*.nc files found in {pred_dir}")

    selected_events = event_names or list(EVENTS.keys())
    generated: list[str] = []

    for event_name in selected_events:
        cfg = EVENTS[event_name]
        event_pred_files = select_prediction_files_for_event(pred_files, cfg)
        if not event_pred_files:
            LOG.info("Skipping event=%s: no matching prediction files", event_name)
            continue

        # Filter to requested date and steps
        date_int = int(date)
        step_files = [
            (p, ymd, step) for p, ymd, step in event_pred_files
            if ymd == date_int and step in steps
        ]
        if not step_files:
            LOG.warning(
                "Skipping event=%s: no prediction files for date=%s steps=%s",
                event_name, date, steps,
            )
            continue

        safe_label = display_label.replace(" ", "_").replace("/", "_")
        pdf_name = f"tc_members_{event_name}_{safe_label}_{date}.pdf"
        pdf_path = out_dir / pdf_name

        with PdfPages(pdf_path) as pdf:
            for nc_path, ymd, step in sorted(step_files, key=lambda x: x[2]):
                LOG.info("Loading event=%s date=%s step=%d", event_name, date, step)
                try:
                    fields = _load_prediction_member_fields(nc_path, cfg, members)
                except Exception as exc:
                    LOG.warning("Failed to load %s: %s", nc_path.name, exc)
                    continue
                # One page per member
                for mi, mbr in enumerate(members):
                    LOG.info("  Plotting member %d (step=%dh)", mbr, step)
                    fig = _plot_member_page(
                        fields,
                        cfg=cfg,
                        member_idx=mi,
                        member_label=mbr,
                        step_hours=step,
                        date_str=date,
                        display_label=display_label,
                    )
                    pdf.savefig(fig, dpi=200, bbox_inches="tight")
                    plt.close(fig)

        LOG.info("Saved TC member maps PDF: %s", pdf_path)
        generated.append(str(pdf_path))

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TC member maps.")
    # Legacy GRIB-only mode
    parser.add_argument("--expver", default=None)
    # Prediction-based mode
    parser.add_argument("--predictions-dir", default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument(
        "--display-label",
        default="",
        help="Optional short label for plot titles.",
    )
    parser.add_argument(
        "--events",
        default="",
        help="Comma-separated subset of events from tc_events.py.",
    )
    parser.add_argument("--date", default=None, help="YYYYMMDD date to plot (predictions mode).")
    parser.add_argument(
        "--steps",
        default="24,120",
        help="Comma-separated forecast step hours (predictions mode).",
    )

    parser.add_argument("--outdir", required=True)
    parser.add_argument("--base-dir", default="/home/ecm5702/hpcperm/data/tc")
    parser.add_argument("--members", default="0,1,2,3,4")
    parser.add_argument("--year", default="2023")
    parser.add_argument("--month", default="08")
    parser.add_argument("--resol", type=float, default=0.25)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.predictions_dir:
        if not args.run_label:
            parser.error("--run-label is required when using --predictions-dir")
        if not args.date:
            parser.error("--date is required when using --predictions-dir")
        event_names = [v.strip() for v in args.events.split(",") if v.strip()] or None
        steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
        generated = run_tc_member_plots_from_predictions(
            predictions_dir=args.predictions_dir,
            outdir=args.outdir,
            run_label=args.run_label,
            display_label=args.display_label or None,
            event_names=event_names,
            date=args.date,
            steps=steps,
            members=_parse_members(args.members),
        )
        print(f"Generated {len(generated)} PDFs in {Path(args.outdir).expanduser().resolve()}")
    elif args.expver:
        generated = run_tc_member_plots(
            expver=args.expver,
            outdir=args.outdir,
            base_dir=args.base_dir,
            members=_parse_members(args.members),
            year=args.year,
            month=args.month,
            resol=args.resol,
        )
        print(f"Generated {len(generated)} files in {Path(args.outdir).expanduser().resolve() / 'tc_members'}")
    else:
        parser.error("Either --predictions-dir or --expver is required")


if __name__ == "__main__":
    main()
