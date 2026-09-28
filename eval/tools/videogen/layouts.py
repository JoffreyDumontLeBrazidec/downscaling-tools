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
from eval.plotting import eval_style
from eval.plotting.style import PNG_DPI as PNG_DPI_STILL
from eval.plotting.maps_helpers import octahedral_grid_name, region_projection

from .panels import (
    add_bbox_polyline,
    add_connector,
    cmap_for,
    label_for,
    render_field_panel,
    short_name,
    to_display,
)


def select_projection(bbox):
    """Shared projection rule for a ``(lon_min, lon_max, lat_min, lat_max)`` box."""
    return region_projection(*bbox)


def _grids(ds: xr.Dataset) -> tuple[str, str]:
    """(input grid, output grid) names for titles, e.g. ("O320", "O1280")."""
    lres = octahedral_grid_name(int(ds.sizes.get("grid_point_lres", 0))) or "input grid"
    hres = octahedral_grid_name(int(ds.sizes.get("grid_point_hres", 0))) or "output grid"
    return lres, hres


def _display_norm(var: str, norm: tuple[float, float]) -> tuple[float, float]:
    lo, hi = to_display(var, [norm[0], norm[1]])
    return float(lo), float(hi)


def _save_dpi(scene: SceneConfig, out_png: Path) -> int:
    """Video frames keep the scene dpi (fixed frame size); the preview still uses 150 dpi."""
    return PNG_DPI_STILL if Path(out_png) == Path(scene.preview_path) else scene.dpi


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


def _open_grids(frame: Frame) -> tuple[str, str]:
    with xr.open_dataset(frame.nc_path) as ds:
        return _grids(ds)


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
    lres_grid, hres_grid = _open_grids(frame)

    with eval_style():
        fig = plt.figure(figsize=(16, 10), dpi=scene.dpi)
        proj_inset = select_projection(inset_bbox)

        # Top row. The colour bars sit ABOVE the panels (see below) so that the two lines that
        # join the box on the regional map to the outer panels cross nothing on their way up.
        PANEL_Y, PANEL_H, PANEL_W = 0.52, 0.33, 0.20
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
            who = f"Model ({hres_grid})" if is_hres else f"Input ({lres_grid})"
            title = f"{short_name(v)}\n{who}"
            vmin, vmax = _display_norm(v, norms[v])
            sc = render_field_panel(
                ax, lon, lat, to_display(v, vals),
                bbox=inset_bbox, resolution_deg=res,
                cmap=cmap_for(v), vmin=vmin, vmax=vmax,
                title=title, label_fontsize=8, left_labels=(col == 0),
            )
            if is_hres:
                sc_handles[v] = sc

        # Per-variable colorbars, above the two panels of the variable (ticks and label on top).
        for i, v in enumerate(vars_list):
            cax = fig.add_axes([0.05 + i * 0.475, 0.910, 0.41, 0.013])
            cb = fig.colorbar(sc_handles[v], cax=cax, orientation="horizontal")
            cb.ax.xaxis.set_ticks_position("top")
            cb.ax.xaxis.set_label_position("top")
            cb.set_label(label_for(v), fontsize=10)
            cb.outline.set_edgecolor("black")
            cb.outline.set_linewidth(1.0)
            cb.ax.tick_params(labelsize=9)

        # Regional MSL bg.
        proj_bg = select_projection(scene.bg_bbox)
        ax_bg = fig.add_axes([0.10, 0.05, 0.80, 0.36], projection=proj_bg)
        msl_vmin, msl_vmax = _display_norm(
            "msl", norms.get("msl", (float(bg_vals.min()), float(bg_vals.max()))))
        render_field_panel(
            ax_bg, bg_lon, bg_lat, to_display("msl", bg_vals),
            bbox=scene.bg_bbox, resolution_deg=scene.bg_resolution_deg,
            cmap=cmap_for("msl"), vmin=msl_vmin, vmax=msl_vmax,
            title=f"Regional context: input {short_name('msl').lower()} ({lres_grid}, hPa)",
            label_fontsize=9, n_contours=14,
        )
        add_bbox_polyline(ax_bg, inset_bbox, color="black", linewidth=1.8)
        add_connector(fig, ax_bg, inset_bbox[0], inset_bbox[3], ax_top[0], (0.0, 0.0))
        add_connector(fig, ax_bg, inset_bbox[1], inset_bbox[3], ax_top[-1], (1.0, 0.0))

        fig.suptitle(
            f"{scene.title}   ({scene.ckpt_label})   |   {frame.label()}",
            fontsize=15, y=1.005,
        )
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=_save_dpi(scene, out_png), bbox_inches="tight", facecolor="white")
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

    with xr.open_dataset(frame.nc_path) as ds:
        lres_grid, hres_grid = _grids(ds)

    with eval_style():
        fig = plt.figure(figsize=(15, 8.5), dpi=scene.dpi)
        proj_bg = select_projection(scene.bg_bbox)
        proj_ins = select_projection(inset_bbox)
        ax_bg = fig.add_axes([0.045, 0.13, 0.43, 0.76], projection=proj_bg)
        ax_hr = fig.add_axes([0.55,  0.13, 0.40, 0.76], projection=proj_ins)

        vmin, vmax = _display_norm(var, norms[var])
        cmap = cmap_for(var)
        sc = render_field_panel(
            ax_bg, bg_lon, bg_lat, to_display(var, bg_vals),
            bbox=scene.bg_bbox, resolution_deg=scene.bg_resolution_deg,
            cmap=cmap, vmin=vmin, vmax=vmax, title=f"Input ({lres_grid})",
        )
        render_field_panel(
            ax_hr, ins_lon, ins_lat, to_display(var, ins_vals),
            bbox=inset_bbox, resolution_deg=scene.hres_resolution_deg,
            cmap=cmap, vmin=vmin, vmax=vmax, title=f"Model ({hres_grid}), boxed area",
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
        fig.savefig(out_png, dpi=_save_dpi(scene, out_png), bbox_inches="tight", facecolor="white")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Dispatch table — add new layouts here.
# ---------------------------------------------------------------------------

LAYOUT_RENDERERS = {
    "dual_row": render_dual_row,
    "single_inset": render_single_inset,
}
