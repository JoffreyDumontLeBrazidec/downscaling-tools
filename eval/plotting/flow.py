"""10 m wind streamlines over a shaded field: the house flow overlay of the map templates.

Owner, 26 Nov 2026 (review of a colleague's AIFS-ENS deck, slide 2, items 1 and 2 approved): the sparse quiver
arrows of the zoom maps are replaced by thin semi-transparent streamlines whose width grows with the wind speed,
drawn over the shaded field (10 m wind speed, tcw, 2t). Same white-background template, colour bars and titles;
only the arrows change.

Method, the same in every template:

* the wind of each panel's own native points is put on one regular grid in the axes' own coordinates (the map
  projection, or the km plane of a storm-centred panel): inverse-distance weights of the 4 nearest points, cells
  farther than 3 median spacings from any point are left empty (no streamline crosses a data gap);
* the direction is the true direction on the map: eastward wind is divided by cos(latitude) before it is turned
  into the projection, so a 45-degree wind is drawn at 45 degrees at any latitude;
* line width = ``LW_MIN`` + (``LW_MAX`` - ``LW_MIN``) * min(speed / ``SPEED_REF``, 1); the same fixed reference in
  every panel and every page, so widths compare across panels, members and runs;
* small arrow heads give the sense of the flow. Paths only, no score; one realisation per panel.
"""
from __future__ import annotations

import numpy as np

SPEED_REF = 30.0      # m s-1: lines reach their full width at this speed
LW_MIN, LW_MAX = 0.25, 1.5
ALPHA = 0.6
# white on the dark shades of wind speed and msl; near-black where the colour map is white at one end or in the
# middle (tcw davos_r is white when dry, 2t vik white at 0 degC, tp white when dry)
LINE_COLOR = {"wind10m": "white", "10ff": "white", "msl": "white"}
DARK_LINE = "0.08"
GRID_CELLS = 220      # streamline grid cells along the longer side of a panel


def line_color(variable: str) -> str:
    return LINE_COLOR.get(variable, DARK_LINE)


def width_from_speed(speed: np.ndarray) -> np.ndarray:
    s = np.nan_to_num(np.asarray(speed, dtype=np.float64), nan=0.0)
    return LW_MIN + (LW_MAX - LW_MIN) * np.clip(s / SPEED_REF, 0.0, 1.0)


def to_regular(x: np.ndarray, y: np.ndarray, values: list[np.ndarray], x_edges: tuple[float, float],
               y_edges: tuple[float, float], n_long: int = GRID_CELLS):
    """(gx, gy, [grids]) of scattered values on a regular grid; empty cells far from every point are NaN."""
    from scipy.spatial import cKDTree

    x0, x1 = map(float, x_edges)
    y0, y1 = map(float, y_edges)
    if (x1 - x0) >= (y1 - y0):
        nx, ny = n_long, max(8, int(round(n_long * (y1 - y0) / (x1 - x0))))
    else:
        ny, nx = n_long, max(8, int(round(n_long * (x1 - x0) / (y1 - y0))))
    gx, gy = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    ok = np.isfinite(x) & np.isfinite(y)
    for v in values:
        ok &= np.isfinite(v)
    pad_x, pad_y = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
    ok &= (x >= x0 - pad_x) & (x <= x1 + pad_x) & (y >= y0 - pad_y) & (y <= y1 + pad_y)
    if ok.sum() < 8:
        return gx, gy, [np.full((ny, nx), np.nan) for _ in values]
    pts = np.c_[x[ok], y[ok]]
    tree = cKDTree(pts)
    spacing = float(np.median(tree.query(pts[: min(len(pts), 20000)], k=2)[0][:, 1]))
    gxx, gyy = np.meshgrid(gx, gy)
    d, i = tree.query(np.c_[gxx.ravel(), gyy.ravel()], k=min(4, len(pts)))
    w = 1.0 / np.maximum(d, 1e-9 * max(spacing, 1e-12)) ** 2
    w /= w.sum(axis=1, keepdims=True)
    far = d[:, 0] > 3.0 * max(spacing, (gx[1] - gx[0]), (gy[1] - gy[0]))
    out = []
    for v in values:
        g = (w * np.asarray(v, dtype=np.float64)[ok][i]).sum(axis=1)
        g[far] = np.nan
        out.append(g.reshape(ny, nx))
    return gx, gy, out


def _plain_streamplot(ax, gx, gy, U, V, speed, *, color, density, alpha, zorder):
    import matplotlib.axes

    U = np.ma.masked_invalid(U)
    V = np.ma.masked_invalid(V)
    if U.count() < 16:
        return None
    lw = width_from_speed(speed)
    # base Axes.streamplot: the grid is already in the axes' own coordinates (cartopy's GeoAxes.streamplot would
    # regrid it again on a coarse 30 x density grid)
    sp = matplotlib.axes.Axes.streamplot(ax, gx, gy, U, V, density=density, color=color, linewidth=lw,
                                         arrowsize=0.55, arrowstyle="-|>", minlength=0.12, zorder=zorder,
                                         integration_direction="both")
    sp.lines.set_alpha(alpha)
    sp.arrows.set_alpha(alpha)
    return sp


def streamlines_on_map(ax, lon: np.ndarray, lat: np.ndarray, u: np.ndarray, v: np.ndarray, extent, *,
                       variable: str = "wind10m", density: float = 2.0, alpha: float = ALPHA,
                       n_long: int = GRID_CELLS, zorder: float = 6):
    """Streamlines of (u, v) in m s-1 at native (lon, lat) points on a cartopy map axes, inside ``extent``.

    ``extent`` = [lon_min, lon_max, lat_min, lat_max]; the axes' extent must already be set."""
    import cartopy.crs as ccrs

    lon = ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lat = np.asarray(lat, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w, e, s, n = extent
    if e > 180.0 or w < -180.0:                    # a box across the dateline: lon continuous with the box
        lon = np.where(lon < w, lon + 360.0, lon)
    pad = 0.1 * max(e - w, n - s)
    k = (lon >= w - pad) & (lon <= e + pad) & (lat >= s - pad) & (lat <= n + pad) & np.isfinite(u) & np.isfinite(v)
    if k.sum() < 16:
        return None
    lon, lat, u, v = lon[k], lat[k], u[k], v[k]
    speed = np.hypot(u, v)
    pc = ccrs.PlateCarree()
    proj = ax.projection
    c = np.maximum(np.cos(np.deg2rad(lat)), 0.05)
    if isinstance(proj, ccrs.PlateCarree):
        xy = np.c_[lon, lat]
        ue, ve = u / c, v                       # degrees per unit time in the plotted lon/lat plane
    else:
        xy = proj.transform_points(pc, lon, lat)[:, :2]
        ue, ve = proj.transform_vectors(pc, lon, lat, u / c, v)
    x0, x1, y0, y1 = ax.get_extent(crs=proj)
    gx, gy, (U, V, S) = to_regular(xy[:, 0], xy[:, 1], [ue, ve, speed], (x0, x1), (y0, y1), n_long=n_long)
    sp = _plain_streamplot(ax, gx, gy, U, V, S, color=line_color(variable), density=density, alpha=alpha,
                           zorder=zorder)
    ax.set_extent(extent, crs=pc)                # streamplot autoscales the view
    return sp


def streamlines_on_plane(ax, x: np.ndarray, y: np.ndarray, U: np.ndarray, V: np.ndarray, *,
                         variable: str = "wind10m", density: float = 1.8, alpha: float = ALPHA, zorder: float = 6):
    """Streamlines on a regular grid of a plain axes (e.g. the km plane of a storm-centred panel).

    ``x`` (nx,) and ``y`` (ny,) increasing and evenly spaced, ``U``, ``V`` (ny, nx) in m s-1."""
    return _plain_streamplot(ax, np.asarray(x, float), np.asarray(y, float), np.asarray(U, float),
                             np.asarray(V, float), np.hypot(U, V), color=line_color(variable), density=density,
                             alpha=alpha, zorder=zorder)
