"""Cartopy + cmcrameri panel rendering.

House map style (eval.plotting):
  * projection from the shared select_projection rule, picked per bbox,
  * pcolormesh with the per-variable field colour maps of eval.plotting.variables,
  * light black contour overlay,
  * coastlines / borders / labelled grid lines from add_geography.

Per-variable cmap and label are registry-driven (built on the house variable table).
Box outlines and connector lines are black, because red means "model" in every figure.
"""
from __future__ import annotations

import cartopy.crs as ccrs
import numpy as np
from matplotlib import colormaps as _colormaps
from matplotlib.colors import Colormap
from matplotlib.patches import ConnectionPatch

from eval.plotting import add_geography, convert, variable_spec
from eval.plotting.maps_helpers import set_inner_extent


def plt_get_cmap(cmap) -> Colormap:
    """A Colormap object from a name or a Colormap."""
    return cmap if isinstance(cmap, Colormap) else _colormaps[cmap]

from .config import BBox
from .data import nearest_regrid


# ---------------------------------------------------------------------------
# Colour-map / label registries, built on the house variable table
# ---------------------------------------------------------------------------
# The projection rule is the shared one (eval.plotting.select_projection, via
# eval.plotting.maps_helpers.region_projection for southern-hemisphere boxes).
# Names, units and field colour maps come from eval.plotting.variables; the
# video frames show msl in hPa, z_500 in dam, temperatures in K (``to_display``).

# Scene variable name -> key of the house variable table ("wind" is the 10 m speed).
HOUSE_KEY: dict[str, str] = {"wind": "10ff"}


def _house(var: str):
    return variable_spec(HOUSE_KEY.get(var, var))


# Edit / extend the variable list to support new variables.
CMAP_BY_VAR: dict[str, Colormap] = {
    v: plt_get_cmap(_house(v).field_cmap()) for v in ("msl", "wind", "2t", "skt", "tcw", "z_500")
}

LABEL_BY_VAR: dict[str, str] = {
    v: _house(v).label for v in ("msl", "wind", "2t", "skt", "tcw", "z_500")
}


def cmap_for(var: str) -> Colormap:
    return CMAP_BY_VAR.get(var) or plt_get_cmap(_house(var).field_cmap())


def label_for(var: str) -> str:
    return LABEL_BY_VAR.get(var) or _house(var).label


def to_display(var: str, values):
    """Native values (Pa, m2 s-2, K, m s-1) in the display unit of ``label_for(var)``."""
    return np.asarray(convert(HOUSE_KEY.get(var, var), values), dtype=float)


def short_name(var: str) -> str:
    """Display name without unit, e.g. "Mean sea level pressure"."""
    return _house(var).name


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

    # Largest projected rectangle inside the box: no empty corners on a conic map.
    set_inner_extent(ax, bbox)
    gl = add_geography(ax, label_size=label_fontsize)
    if gl is not None:
        gl.left_labels = left_labels
        gl.bottom_labels = bottom_labels

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    if title is not None:
        ax.set_title(title, fontsize=11, pad=4)

    return im


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def add_bbox_polyline(ax, bbox: BBox, *, color: str = "black", linewidth: float = 1.8) -> None:
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
    color: str = "black",
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
