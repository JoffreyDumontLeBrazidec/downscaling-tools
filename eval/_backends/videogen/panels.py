"""Cartopy + cmcrameri panel rendering.

Visual style mirrors ``eval/_backends/tc/member_plot.py``:
  * LambertConformal projection picked per bbox,
  * pcolormesh with cmcrameri colormaps,
  * light black contour overlay,
  * cartopy coastlines / borders / gridlines.

Per-variable cmap and label are registry-driven (just edit the dicts below).
"""
from __future__ import annotations

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmcrameri.cm as cmc
import numpy as np
from matplotlib.colors import Colormap
from matplotlib.patches import ConnectionPatch

from .config import BBox
from .data import nearest_regrid


# ---------------------------------------------------------------------------
# Projection / cmap / label registries
# ---------------------------------------------------------------------------

def select_projection(bbox: BBox) -> ccrs.Projection:
    """LambertConformal for non-dateline bboxes; PlateCarree otherwise."""
    lon_min, lon_max, lat_min, lat_max = bbox
    if lon_min > lon_max:
        return ccrs.PlateCarree()
    return ccrs.LambertConformal(
        central_longitude=(lon_min + lon_max) / 2,
        central_latitude=(lat_min + lat_max) / 2,
    )


# Edit / extend these to support new variables.
CMAP_BY_VAR: dict[str, Colormap] = {
    "msl": cmc.vik,
    "wind": cmc.batlow,
    "2t": cmc.roma_r,
    "skt": cmc.roma_r,
    "tcw": cmc.davos_r,
    "z_500": cmc.batlow,
}

LABEL_BY_VAR: dict[str, str] = {
    "msl": "MSL pressure (Pa)",
    "wind": "10 m wind speed (m s$^{-1}$)",
    "2t": "2 m temperature (K)",
    "skt": "Skin temperature (K)",
    "tcw": "Total column water (kg m$^{-2}$)",
    "z_500": "500 hPa geopotential (m$^2$ s$^{-2}$)",
}


def cmap_for(var: str) -> Colormap:
    return CMAP_BY_VAR.get(var, cmc.batlow)


def label_for(var: str) -> str:
    return LABEL_BY_VAR.get(var, var)


# ---------------------------------------------------------------------------
# Panel renderer
# ---------------------------------------------------------------------------

def render_field_panel(
    ax,
    lon: np.ndarray, lat: np.ndarray, vals: np.ndarray,
    *,
    bbox: BBox,
    resolution_deg: float,
    cmap,
    vmin: float, vmax: float,
    title: str | None = None,
    label_fontsize: int = 9,
    n_contours: int = 12,
    left_labels: bool = True,
    bottom_labels: bool = True,
    contour_alpha: float = 0.55,
):
    """Regrid + pcolormesh + contour overlay on one cartopy axes.

    Returns the ``QuadMesh`` (use for a colorbar).
    """
    lon_edges, lat_edges, lon_centers, lat_centers, values_2d = nearest_regrid(
        lon, lat, vals, bbox=bbox, resolution_deg=resolution_deg,
    )
    im = ax.pcolormesh(
        lon_edges, lat_edges, values_2d,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="flat", rasterized=True,
    )
    if n_contours and n_contours > 0:
        try:
            levels = np.linspace(vmin, vmax, n_contours)
            ax.contour(
                lon_centers, lat_centers, values_2d,
                transform=ccrs.PlateCarree(),
                levels=levels, colors="black",
                linewidths=0.4, alpha=contour_alpha,
            )
        except Exception:
            # Degenerate fields (e.g. constant) — silently skip contours.
            pass

    ax.set_extent([bbox[0], bbox[1], bbox[2], bbox[3]], crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=0.6, color="black", zorder=5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", zorder=5)

    gl = ax.gridlines(
        draw_labels=True, dms=False,
        x_inline=False, y_inline=False,
        linewidth=0.3, color="gray", alpha=0.5,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = left_labels
    gl.bottom_labels = bottom_labels
    gl.xlabel_style = {"fontsize": label_fontsize}
    gl.ylabel_style = {"fontsize": label_fontsize}

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    if title is not None:
        ax.set_title(title, fontsize=11, pad=4)

    return im


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def add_bbox_polyline(ax, bbox: BBox, *, color: str = "red", linewidth: float = 1.8) -> None:
    """Draw a bbox as a PlateCarree-transformed polyline on a cartopy axes."""
    lons = [bbox[0], bbox[1], bbox[1], bbox[0], bbox[0]]
    lats = [bbox[2], bbox[2], bbox[3], bbox[3], bbox[2]]
    ax.plot(lons, lats, transform=ccrs.PlateCarree(),
            color=color, linewidth=linewidth, zorder=20)


def add_connector(
    fig,
    ax_from,                          # cartopy axes
    lon: float, lat: float,            # PlateCarree data point on ax_from
    ax_to,                             # any axes (cartopy or plain)
    end_xy: tuple[float, float],       # axes-fraction point on ax_to
    *,
    color: str = "red",
    linewidth: float = 0.9,
    alpha: float = 0.85,
) -> None:
    """Line from a geographic point on a cartopy axes to an axes-fraction
    point on another axes."""
    proj_xy = ax_from.projection.transform_point(lon, lat, ccrs.PlateCarree())
    con = ConnectionPatch(
        xyA=proj_xy, coordsA=ax_from.transData,
        xyB=end_xy, coordsB=ax_to.transAxes,
        color=color, linewidth=linewidth, alpha=alpha, zorder=30,
    )
    fig.add_artist(con)
