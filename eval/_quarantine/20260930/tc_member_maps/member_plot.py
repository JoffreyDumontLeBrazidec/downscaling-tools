"""Member spatial maps — per-member 2x3 grid pages (MSLP / 10 m wind x input / model / truth)."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from .data_types import BoundingBox
from .experiment_config import TCExperimentConfig
from .plot_config import TCPlotConfig

from eval.plotting import (
    add_geography,
    axis_label,
    eval_style,
    extend_for,
    select_projection_bbox,
    shared_norm,
    shorten_run_label,
    variable_spec,
)

LOG = logging.getLogger(__name__)

# Folder names that the runner may pass as a run label; they say nothing about the model.
_LEAKED_RUN_LABELS = {"", "eval_inputs", "predictions", "prediction", "data"}


def _source_labels(exp_config: TCExperimentConfig | None) -> tuple[str, str]:
    """Derive descriptive labels for input and target columns."""
    if exp_config is None:
        return "Input", "Target"

    def _fmt(expid: str) -> str:
        return expid.rsplit("_", 1)[0].replace("_", " ")

    refs = exp_config.reference_expids
    if len(refs) >= 2:
        # Convention: first reference is the "target" (higher res), second is "input" (lower res)
        input_src = _fmt(refs[-1])
        target_src = _fmt(refs[0])
    elif len(refs) == 1:
        input_src = _fmt(refs[0])
        if exp_config.analysis_expid:
            analysis_res = exp_config.analysis_expid.split("_")[1]
            target_src = f"IEKM {analysis_res}"
        else:
            target_src = "Target"
    else:
        input_src = "Input"
        target_src = "Target"
    return input_src, target_src


def _date_text(date_str) -> str:
    text = str(date_str)
    return f"{text[:4]}-{text[4:6]}-{text[6:8]} 00 UTC" if len(text) == 8 and text.isdigit() else text


def _column_titles(exp_config: TCExperimentConfig | None) -> tuple[str, str, str]:
    """Descriptive column titles: input, model, truth (source names when they are known)."""
    input_src, target_src = _source_labels(exp_config)
    input_title = ("Input (interpolated to the target grid)" if input_src == "Input"
                   else f"Input ({input_src}, interpolated)")
    truth_title = "Truth" if target_src == "Target" else f"Truth ({target_src})"
    return input_title, "Model", truth_title


def _plot_member_page(
    fields: dict[str, np.ndarray],
    *,
    bbox: BoundingBox,
    plot_config: TCPlotConfig,
    exp_config: TCExperimentConfig | None,
    member_idx: int,
    member_label: int,
    step_hours: int,
    date_str: str,
    display_label: str,
    event_name: str,
) -> plt.Figure:
    """Create a 2x3 grid for one member: rows=[MSLP, 10 m wind], cols=[Input, Model, Truth].

    The fields arrive in display units (hPa and m s-1, see ``load_prediction_member_fields``).
    Each row shares one colour scale across its three panels.
    """
    with eval_style():
        return _draw_member_page(
            fields, bbox=bbox, plot_config=plot_config, exp_config=exp_config,
            member_idx=member_idx, member_label=member_label, step_hours=step_hours,
            date_str=date_str, display_label=display_label, event_name=event_name,
        )


def _draw_member_page(fields, *, bbox, plot_config, exp_config, member_idx, member_label,
                      step_hours, date_str, display_label, event_name) -> plt.Figure:
    from cartopy import crs
    from matplotlib.gridspec import GridSpec

    proj = select_projection_bbox(bbox)
    input_title, model_title, truth_title = _column_titles(exp_config)

    # Size the page to the box so the three columns sit close together (no wide gaps).
    lon_span = (bbox.east - bbox.west) % 360.0 or 360.0
    mid_lat = np.deg2rad((bbox.north + bbox.south) / 2.0)
    bbox_aspect = lon_span * max(np.cos(mid_lat), 0.3) / (bbox.north - bbox.south)
    panel_w = float(np.clip(4.6 * bbox_aspect, 3.2, 6.0))
    panel_h = float(np.clip(panel_w / bbox_aspect, 2.6, 5.0))
    fig = plt.figure(figsize=(3 * panel_w + 1.6, 2 * panel_h + 3.4))
    gs = GridSpec(
        2, 3,
        hspace=0.45, wspace=0.16,
        left=0.06, right=0.98, bottom=0.09, top=0.88,
    )

    map_axes = np.empty((2, 3), dtype=object)
    for row_i in range(2):
        for col_i in range(3):
            map_axes[row_i, col_i] = fig.add_subplot(gs[row_i, col_i], projection=proj)

    lat = fields["lat_axis"]
    lon = fields["lon_axis"]

    col_defs = [
        ("x_interp", input_title),
        ("y_pred", model_title),
        ("y", truth_title),
    ]

    # Rows: variable-table key, field suffix, fixed range from the event plot config.
    row_defs = [
        ("msl", "msl", plot_config.member_map_msl_range),
        ("10ff", "wind", plot_config.member_map_wind_range),
    ]

    row_images = {}
    row_extend = {}

    for row_i, (var_key, var_suffix, fixed_range) in enumerate(row_defs):
        spec = variable_spec(var_key)
        arrays = [fields[f"{p}_{var_suffix}"][member_idx] for p, _ in col_defs if f"{p}_{var_suffix}" in fields]
        # One colour scale for the whole row (input, model and truth are compared).
        if fixed_range is not None:
            norm = shared_norm(vmin=fixed_range[0], vmax=fixed_range[1])
        elif var_suffix == "wind":
            # full range: the storm extremes are what these maps are for
            norm = shared_norm(*arrays, vmin=0.0, q=(0.0, 100.0))
        else:
            norm = shared_norm(*arrays, q=(0.0, 100.0))
        row_extend[row_i] = extend_for(norm, *arrays)
        cmap = spec.field_cmap()

        for col_i, (src_prefix, col_title) in enumerate(col_defs):
            ax = map_axes[row_i, col_i]
            field_key = f"{src_prefix}_{var_suffix}"
            ax.set_extent([bbox.west, bbox.east, bbox.south, bbox.north], crs=crs.PlateCarree())
            gl = add_geography(ax, label_size=9)
            if gl is not None and col_i > 0:
                gl.left_labels = False
            ax.set_title(f"{col_title}\n{spec.name}", fontsize=11)

            if field_key not in fields:
                ax.text(
                    0.5, 0.5, "not available",
                    transform=ax.transAxes,
                    ha="center", va="center", fontsize=13, color="0.45",
                )
                continue

            arr = fields[field_key][member_idx]
            im = ax.pcolormesh(
                lon, lat, arr,
                transform=crs.PlateCarree(),
                norm=norm,
                shading="nearest",
                cmap=cmap,
                rasterized=True,
            )
            row_images.setdefault(row_i, im)

    # One horizontal colour bar per row, centred under the row, with the unit.
    fig.canvas.draw()
    for row_i, (var_key, _suffix, _range) in enumerate(row_defs):
        if row_i not in row_images:
            continue
        pos0 = map_axes[row_i, 0].get_position()
        pos2 = map_axes[row_i, 2].get_position()
        row_width = pos2.x1 - pos0.x0
        cax = fig.add_axes([pos0.x0 + row_width * 0.2, pos0.y0 - 0.065, row_width * 0.6, 0.014])
        cbar = fig.colorbar(row_images[row_i], cax=cax, orientation="horizontal",
                            extend=row_extend[row_i])
        cbar.set_label(axis_label(var_key))

    run_text = shorten_run_label(str(display_label or "").strip())
    run_part = "" if run_text.lower() in _LEAKED_RUN_LABELS else f" (model run {run_text})"
    fig.suptitle(
        f"Tropical cyclone {event_name.capitalize()}: ensemble member {member_label}, "
        f"forecast from {_date_text(date_str)} at lead time +{step_hours} h{run_part}",
        y=0.97,
    )
    return fig
