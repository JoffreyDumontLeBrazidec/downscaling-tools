"""Map helpers: one projection rule, one geography look, shared colour scales.

* ``select_projection`` is the only copy of the projection rule: Lambert conformal centred on
  the box, or plate carrée when the box crosses the dateline.
* ``add_geography`` gives every map coastlines, country borders and labelled grid lines.
* ``shared_norm`` / ``symmetric_norm`` build one colour scale for all the panels of a row, so
  panels that are compared are drawn on the same scale (errors always centred on zero).
* ``map_grid`` lays out rows of map panels with a shared colour bar per row.
"""
from __future__ import annotations

import numpy as np

COAST_LW = 0.6
BORDER_LW = 0.4


def select_projection(west: float, east: float, south: float, north: float):
    """Cartopy projection for a bounding box (degrees).

    Lambert conformal centred on the box for boxes that do not cross the dateline, plate
    carrée for those that do (``east < west``). Southern-hemisphere boxes get the mirrored cone.
    """
    from cartopy import crs

    if east < west:
        return crs.PlateCarree()
    central_lon = (west + east) / 2.0
    central_lat = (south + north) / 2.0
    if central_lat < 0:
        # Cartopy's default Lambert cone opens northwards and cuts off everything south of 30 S;
        # mirror it for boxes in the southern hemisphere.
        return crs.LambertConformal(central_longitude=central_lon, central_latitude=central_lat,
                                    standard_parallels=(-33.0, -45.0), cutoff=30)
    return crs.LambertConformal(central_longitude=central_lon, central_latitude=central_lat)


def select_projection_bbox(bbox):
    """``select_projection`` for a ``(west, east, south, north)`` tuple or a bbox object."""
    if hasattr(bbox, "west"):
        return select_projection(bbox.west, bbox.east, bbox.south, bbox.north)
    west, east, south, north = bbox
    return select_projection(west, east, south, north)


def add_geography(ax, *, coastlines: bool = True, borders: bool = True, gridlines: bool = True,
                  labels: bool = True, resolution: str = "50m", label_size: float = 7.0,
                  coast_lw: float = COAST_LW):
    """Coastlines, borders and (optionally labelled) grid lines on a Cartopy axes.

    Returns the ``Gridliner`` (or ``None``). Ticks are labelled on the left and bottom edges
    only. Axes that are not Cartopy axes get nothing.
    """
    if not hasattr(ax, "coastlines"):
        return None
    import cartopy.feature as cfeature

    if coastlines:
        ax.coastlines(resolution=resolution, linewidth=coast_lw, color="0.15", zorder=6)
    if borders:
        ax.add_feature(
            cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", resolution),
            edgecolor="0.35", facecolor="none", linewidth=BORDER_LW, linestyle="-", zorder=6,
        )
    gl = None
    if gridlines:
        gl = ax.gridlines(draw_labels=labels, linewidth=0.4, color="0.5", alpha=0.5,
                          linestyle=":", x_inline=False, y_inline=False)
        if labels:
            gl.top_labels = False
            gl.right_labels = False
            gl.rotate_labels = False
            gl.xlabel_style = {"size": label_size}
            gl.ylabel_style = {"size": label_size}
    return gl


def inscribed_extent(proj, extent, n: int = 101) -> tuple[float, float, float, float]:
    """Largest axis-aligned rectangle, in ``proj`` coordinates, inside the box ``extent``.

    ``extent`` is ``(west, east, south, north)`` in degrees. On a conic projection such as the
    Lambert conformal one the meridians converge and the parallels are arcs, so the four edges
    of a latitude-longitude box do not form a rectangle. ``set_extent`` with the box then shows
    the whole bounding rectangle of the fan-shaped box, and thin white wedges appear where the
    fan does not reach it. This rectangle lies wholly inside the box, so the axes are filled
    with data. It returns ``(x0, x1, y0, y1)`` for ``ax.set_extent(..., crs=proj)``.
    """
    import cartopy.crs as ccrs

    west, east, south, north = extent
    lons = np.linspace(west, east, n)
    lats = np.linspace(south, north, n)
    pc = ccrs.PlateCarree()

    def edge(lon, lat):
        pts = proj.transform_points(pc, np.asarray(lon, float), np.asarray(lat, float))
        return pts[:, 0], pts[:, 1]

    x_left = edge(np.full(n, west), lats)[0].max()
    x_right = edge(np.full(n, east), lats)[0].min()
    y_bottom = edge(lons, np.full(n, south))[1].max()
    y_top = edge(lons, np.full(n, north))[1].min()
    return float(x_left), float(x_right), float(y_bottom), float(y_top)


def new_map_axes(fig, spec, extent, *, projection=None, geography: bool = True, fill: bool = False,
                 **geo_kwargs):
    """Add a map panel to ``fig`` at grid position ``spec`` for ``extent=(west, east, south, north)``.

    With ``fill=True`` a Lambert (conic) panel is cropped to the largest rectangle inside the
    box (see ``inscribed_extent``), so that it has no white wedges at its corners. Plate carree
    panels are rectangles already and are left alone.
    """
    import cartopy.crs as ccrs

    west, east, south, north = extent
    proj = projection or select_projection(west, east, south, north)
    ax = fig.add_subplot(spec, projection=proj)
    if fill and not isinstance(proj, (ccrs.PlateCarree,)) and east >= west:
        ax.set_extent(inscribed_extent(proj, extent), crs=proj)
    else:
        ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
    if geography:
        add_geography(ax, **geo_kwargs)
    return ax


def finite_values(*arrays) -> np.ndarray:
    """All finite values of the given arrays as one flat array."""
    parts = [np.asarray(a, dtype=float).ravel() for a in arrays if a is not None]
    if not parts:
        return np.array([])
    v = np.concatenate(parts)
    return v[np.isfinite(v)]


def symmetric_limit(*arrays, q: float = 99.0, floor: float = 1e-12) -> float:
    """Half-width of a zero-centred colour scale covering the ``q`` percentile of ``|values|``."""
    v = finite_values(*arrays)
    if v.size == 0:
        return 1.0
    return max(float(np.percentile(np.abs(v), q)), floor)


def symmetric_norm(*arrays, q: float = 99.0, limit: float | None = None):
    """``(norm, limit)`` for errors: a linear scale centred on zero with equal limits either side."""
    from matplotlib.colors import Normalize

    lim = float(limit) if limit is not None else symmetric_limit(*arrays, q=q)
    return Normalize(vmin=-lim, vmax=lim), lim


def shared_norm(*arrays, q: tuple[float, float] = (1.0, 99.0), vmin: float | None = None,
                vmax: float | None = None, centered: bool = False):
    """One ``Normalize`` for every panel in a row or comparison.

    ``vmin``/``vmax`` take precedence over the percentile range ``q``. With
    ``centered=True`` the scale is symmetric about zero (use for signed fields and errors).
    """
    from matplotlib.colors import Normalize

    if centered:
        lim = max(abs(vmin), abs(vmax)) if vmin is not None and vmax is not None else \
            symmetric_limit(*arrays, q=q[1])
        return Normalize(vmin=-lim, vmax=lim)
    v = finite_values(*arrays)
    lo = vmin if vmin is not None else (float(np.percentile(v, q[0])) if v.size else 0.0)
    hi = vmax if vmax is not None else (float(np.percentile(v, q[1])) if v.size else 1.0)
    if hi <= lo:
        hi = lo + 1e-12
    return Normalize(vmin=lo, vmax=hi)


def extend_for(norm, *arrays) -> str:
    """Colour-bar ``extend`` value: arrows on the sides where data exceeds the scale."""
    v = finite_values(*arrays)
    if v.size == 0:
        return "neither"
    lo, hi = v.min() < norm.vmin, v.max() > norm.vmax
    return "both" if lo and hi else "min" if lo else "max" if hi else "neither"


def add_row_colorbar(fig, mappable, axes, label: str, *, extend: str = "neither",
                     orientation: str = "vertical", pad: float = 0.02, shrink: float = 0.9):
    """One colour bar shared by all ``axes`` of a row, with the unit in ``label``."""
    cb = fig.colorbar(mappable, ax=list(np.ravel(axes)), orientation=orientation, extend=extend,
                      pad=pad, shrink=shrink, fraction=0.04 if orientation == "vertical" else 0.06)
    cb.set_label(label)
    return cb


def map_grid(nrows: int, ncols: int, extent, *, panel_size=(3.6, 3.2), projection=None,
             fill: bool = False, **geo_kwargs):
    """Create ``(fig, axes)`` with ``nrows`` x ``ncols`` map panels (axes is a 2-D array).

    ``fill=True`` crops conic panels to the box's inscribed rectangle (see ``new_map_axes``)."""
    import matplotlib.pyplot as plt

    west, east, south, north = extent
    proj = projection or select_projection(west, east, south, north)
    fig = plt.figure(figsize=(panel_size[0] * ncols + 0.9, panel_size[1] * nrows + 0.6))
    gs = fig.add_gridspec(nrows, ncols, wspace=0.12, hspace=0.18)
    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = new_map_axes(fig, gs[i, j], extent, projection=proj, fill=fill, **geo_kwargs)
    return fig, axes
