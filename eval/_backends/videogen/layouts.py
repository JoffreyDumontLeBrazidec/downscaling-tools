"""Layout renderers.

Each renderer has signature ``(frame, scene, norms, out_png) -> None`` and is
registered in ``LAYOUT_RENDERERS`` for dispatch from ``pipeline.render_one_frame``.

To add a new layout, write a function and append to ``LAYOUT_RENDERERS``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from .config import SceneConfig
from .data import Frame, bbox_mask, load_var_slice, resolve_inset_bbox
from .panels import (
    add_bbox_polyline,
    add_connector,
    cmap_for,
    label_for,
    render_field_panel,
    select_projection,
)


def _open_inset_fields(
    frame: Frame, scene: SceneConfig, vars_list: list[str],
) -> tuple[tuple, dict, tuple]:
    """Return (inset_bbox, per_var, bg_field_msl). per_var[var]={"lres":..., "hres":...}."""
    with xr.open_dataset(frame.nc_path) as ds:
        inset_bbox = resolve_inset_bbox(ds, scene)

        # Bg always reads MSL on lres (used for context map / norm reference).
        bg_lon, bg_lat, bg_vals = load_var_slice(
            ds, "lres", "msl", ensemble_member=scene.ensemble_member,
        )
        m_bg = bbox_mask(bg_lon, bg_lat, scene.bg_bbox)
        bg_field = (bg_lon[m_bg], bg_lat[m_bg], bg_vals[m_bg])

        per_var: dict[str, dict[str, tuple]] = {}
        for v in vars_list:
            lon_l, lat_l, vals_l = load_var_slice(
                ds, "lres", v, ensemble_member=scene.ensemble_member,
            )
            lon_h, lat_h, vals_h = load_var_slice(
                ds, "hres", v, ensemble_member=scene.ensemble_member,
            )
            m_l = bbox_mask(lon_l, lat_l, inset_bbox)
            m_h = bbox_mask(lon_h, lat_h, inset_bbox)
            per_var[v] = {
                "lres": (lon_l[m_l], lat_l[m_l], vals_l[m_l]),
                "hres": (lon_h[m_h], lat_h[m_h], vals_h[m_h]),
            }
    return inset_bbox, per_var, bg_field


# ---------------------------------------------------------------------------
# Layout: dual_row (4 zoomed panels in one row + regional bg map below)
# ---------------------------------------------------------------------------

def render_dual_row(
    frame: Frame, scene: SceneConfig,
    norms: dict[str, tuple[float, float]],
    out_png: Path,
) -> None:
    vars_list = list(scene.vars)
    if len(vars_list) != 2:
        raise ValueError(f"dual_row needs exactly 2 vars, got {vars_list}")

    inset_bbox, per_var, (bg_lon, bg_lat, bg_vals) = _open_inset_fields(frame, scene, vars_list)

    fig = plt.figure(figsize=(16, 10), dpi=scene.dpi)
    proj_inset = select_projection(inset_bbox)

    # Top row.
    PANEL_Y, PANEL_H, PANEL_W = 0.56, 0.33, 0.20
    panel_x = [0.040, 0.246, 0.512, 0.718]
    ax_top = []
    sc_handles: dict[str, object] = {}
    for col, x0 in enumerate(panel_x):
        ax = fig.add_axes([x0, PANEL_Y, PANEL_W, PANEL_H], projection=proj_inset)
        ax_top.append(ax)
        v = vars_list[col // 2]
        is_hres = (col % 2 == 1)
        lon, lat, vals = per_var[v]["hres" if is_hres else "lres"]
        res = scene.hres_resolution_deg if is_hres else scene.bg_resolution_deg
        title = f"{v}  —  {'O1280 prediction' if is_hres else 'O320 input'}"
        sc = render_field_panel(
            ax, lon, lat, vals,
            bbox=inset_bbox, resolution_deg=res,
            cmap=cmap_for(v), vmin=norms[v][0], vmax=norms[v][1],
            title=title, label_fontsize=8, left_labels=(col == 0),
        )
        if is_hres:
            sc_handles[v] = sc

    # Per-variable colorbars.
    for i, v in enumerate(vars_list):
        cax = fig.add_axes([0.05 + i * 0.475, 0.495, 0.41, 0.013])
        cb = fig.colorbar(sc_handles[v], cax=cax, orientation="horizontal")
        cb.set_label(label_for(v), fontsize=10)
        cb.outline.set_edgecolor("black")
        cb.outline.set_linewidth(1.0)
        cb.ax.tick_params(labelsize=9)

    # Regional MSL bg.
    proj_bg = select_projection(scene.bg_bbox)
    ax_bg = fig.add_axes([0.10, 0.05, 0.80, 0.36], projection=proj_bg)
    msl_vmin, msl_vmax = norms.get("msl", (float(bg_vals.min()), float(bg_vals.max())))
    render_field_panel(
        ax_bg, bg_lon, bg_lat, bg_vals,
        bbox=scene.bg_bbox, resolution_deg=scene.bg_resolution_deg,
        cmap=cmap_for("msl"), vmin=msl_vmin, vmax=msl_vmax,
        title="Regional context — O320 MSL",
        label_fontsize=9, n_contours=14,
    )
    add_bbox_polyline(ax_bg, inset_bbox, color="red", linewidth=1.8)
    add_connector(fig, ax_bg, inset_bbox[0], inset_bbox[3], ax_top[0], (0.0, 0.0))
    add_connector(fig, ax_bg, inset_bbox[1], inset_bbox[3], ax_top[-1], (1.0, 0.0))

    fig.suptitle(
        f"{scene.title}   ({scene.ckpt_label})   |   {frame.label()}",
        fontsize=15, y=0.965,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=scene.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Layout: single_inset (wide lres bg + hres prediction side-by-side)
# ---------------------------------------------------------------------------

def render_single_inset(
    frame: Frame, scene: SceneConfig,
    norms: dict[str, tuple[float, float]],
    out_png: Path,
) -> None:
    if not scene.vars:
        raise ValueError("single_inset needs at least 1 var")
    var = scene.vars[0]

    with xr.open_dataset(frame.nc_path) as ds:
        inset_bbox = resolve_inset_bbox(ds, scene)
        lon_l, lat_l, vals_l = load_var_slice(ds, "lres", var, ensemble_member=scene.ensemble_member)
        lon_h, lat_h, vals_h = load_var_slice(ds, "hres", var, ensemble_member=scene.ensemble_member)
        m_bg = bbox_mask(lon_l, lat_l, scene.bg_bbox)
        bg_lon, bg_lat, bg_vals = lon_l[m_bg], lat_l[m_bg], vals_l[m_bg]
        m_ins = bbox_mask(lon_h, lat_h, inset_bbox)
        ins_lon, ins_lat, ins_vals = lon_h[m_ins], lat_h[m_ins], vals_h[m_ins]

    fig = plt.figure(figsize=(15, 8.5), dpi=scene.dpi)
    proj_bg = select_projection(scene.bg_bbox)
    proj_ins = select_projection(inset_bbox)
    ax_bg = fig.add_axes([0.045, 0.13, 0.43, 0.76], projection=proj_bg)
    ax_hr = fig.add_axes([0.55,  0.13, 0.40, 0.76], projection=proj_ins)

    vmin, vmax = norms[var]
    cmap = cmap_for(var)
    sc = render_field_panel(
        ax_bg, bg_lon, bg_lat, bg_vals,
        bbox=scene.bg_bbox, resolution_deg=scene.bg_resolution_deg,
        cmap=cmap, vmin=vmin, vmax=vmax, title="O320 input  (x)",
    )
    render_field_panel(
        ax_hr, ins_lon, ins_lat, ins_vals,
        bbox=inset_bbox, resolution_deg=scene.hres_resolution_deg,
        cmap=cmap, vmin=vmin, vmax=vmax, title="O1280 prediction  (y_pred)",
    )
    add_bbox_polyline(ax_bg, inset_bbox, color="black", linewidth=1.8)
    add_connector(fig, ax_bg, inset_bbox[1], inset_bbox[3], ax_hr, (0.0, 1.0), color="black")
    add_connector(fig, ax_bg, inset_bbox[1], inset_bbox[2], ax_hr, (0.0, 0.0), color="black")

    fig.suptitle(
        f"{scene.title}   ({scene.ckpt_label})   |   {frame.label()}",
        fontsize=14, y=0.97,
    )
    cax = fig.add_axes([0.22, 0.05, 0.56, 0.018])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label(label_for(var), fontsize=11)
    cb.outline.set_edgecolor("black")
    cb.outline.set_linewidth(1.0)
    cb.ax.tick_params(labelsize=10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=scene.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dispatch table — add new layouts here.
# ---------------------------------------------------------------------------

LAYOUT_RENDERERS = {
    "dual_row": render_dual_row,
    "single_inset": render_single_inset,
}
