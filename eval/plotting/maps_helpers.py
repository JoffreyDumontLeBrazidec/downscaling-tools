"""Helpers for the regional map figures (region_plot, zoom_maps, precip maps).

These complement ``eval.plotting.maps``; they were written for the map figures of the
``maps`` conversion group and are kept out of the shared modules on purpose.

* ``octahedral_grid_name`` names an octahedral reduced Gaussian grid from its point count
  (421 120 points -> "O320"), so a panel title can say which grid it shows.
* ``regrid_nearest`` / ``draw_unstructured`` put unstructured grid points on a regular
  longitude-latitude grid by nearest neighbour and draw it as one rasterised mesh. Nearest
  neighbour keeps the native resolution visible (a coarse input looks coarse) and leaves
  cells without a nearby source point empty instead of inventing values.
* ``region_projection`` is ``eval.plotting.select_projection`` made safe for the southern
  hemisphere: Cartopy's ``LambertConformal`` defaults (standard parallels 33 and 45 N,
  cut-off at 30 S) cannot show a box south of 30 S, so for boxes centred south of the
  equator the cone is mirrored (parallels 33 and 45 S, cut-off at 30 N).
* ``set_inner_extent`` shows the largest projected rectangle that lies inside a
  longitude-latitude box, so a Lambert map of a cropped region has no empty corners.
* ``colorbar_beside`` adds one colour bar to the right of a group of map panels, aligned
  with the drawn map area.
* ``region_panel_title`` turns the raw panel keys of the region figures (``x_0``,
  ``x_interp_0``, ``residuals_pred_0``, ...) into readable titles.
"""
from __future__ import annotations

import math
import re

import numpy as np

__all__ = [
    "octahedral_grid_name",
    "dataset_grid_names",
    "region_projection",
    "lambert_for",
    "regrid_nearest",
    "draw_unstructured",
    "set_inner_extent",
    "colorbar_beside",
    "set_grid_ticks",
    "region_panel_title",
    "is_difference_key",
]


def octahedral_grid_name(n_points: int) -> str | None:
    """``"O<N>"`` when ``n_points`` is the size of an octahedral grid (4 N (N + 9)), else None."""
    n = int(n_points)
    if n <= 0:
        return None
    big_n = int(round((-36.0 + math.sqrt(36.0 ** 2 + 16.0 * n)) / 8.0))
    return f"O{big_n}" if 4 * big_n * (big_n + 9) == n else None


def dataset_grid_names(ds) -> tuple[str | None, str | None]:
    """(input grid, target grid) of a predictions dataset, e.g. ("O320", "O1280").

    Uses the ``grid`` attribute for the target when present, else the point counts of
    ``grid_point_lres`` / ``grid_point_hres``. Call it before cropping to a region.
    """
    sizes = getattr(ds, "sizes", {})
    input_grid = octahedral_grid_name(int(sizes["grid_point_lres"])) if "grid_point_lres" in sizes else None
    target = str(getattr(ds, "attrs", {}).get("grid", "")).strip() or None
    if target is None and "grid_point_hres" in sizes:
        target = octahedral_grid_name(int(sizes["grid_point_hres"]))
    return input_grid, target


def lambert_for(central_longitude: float, central_latitude: float):
    """Lambert conformal projection whose cone opens towards the hemisphere of the centre."""
    import cartopy.crs as ccrs

    if central_latitude < 0:
        return ccrs.LambertConformal(central_longitude=central_longitude, central_latitude=central_latitude,
                                     standard_parallels=(-33.0, -45.0), cutoff=30)
    return ccrs.LambertConformal(central_longitude=central_longitude, central_latitude=central_latitude)


def region_projection(west: float, east: float, south: float, north: float):
    """``select_projection`` (the shared rule) with a southern-hemisphere cone when needed."""
    import cartopy.crs as ccrs

    from .maps import select_projection

    proj = select_projection(west, east, south, north)
    if isinstance(proj, ccrs.LambertConformal) and (south + north) / 2.0 < 0:
        return lambert_for((west + east) / 2.0, (south + north) / 2.0)
    return proj


def _lon_near(lon: np.ndarray, west: float, east: float) -> np.ndarray:
    """Shift longitudes by multiples of 360 degrees into the window that starts at ``west``."""
    lon = np.asarray(lon, dtype=float)
    centre = 0.5 * (west + east) if east >= west else 0.5 * (west + east + 360.0)
    return (lon - centre + 180.0) % 360.0 - 180.0 + centre


def regrid_nearest(lon, lat, values, extent, *, res: float | None = None,
                   max_cells: int = 700, max_gap: float = 1.6):
    """Nearest-neighbour resampling of unstructured points onto a regular grid.

    ``extent`` is ``(west, east, south, north)`` in degrees. ``res`` is the latitude
    spacing of the output grid in degrees; by default it is half the mean spacing of the
    source points inside the box, so the native resolution stays visible. Cells whose
    nearest source point is farther than ``max_gap`` source spacings stay NaN.

    Returns ``(lon_centres, lat_centres, grid)`` with ``grid.shape == (n_lat, n_lon)``.
    """
    from scipy.spatial import cKDTree

    west, east, south, north = (float(v) for v in extent)
    if east < west:
        east += 360.0
    lon = _lon_near(lon, west, east)
    lat = np.asarray(lat, dtype=float)
    val = np.asarray(values, dtype=float).reshape(-1)
    ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    lon, lat, val = lon[ok], lat[ok], val[ok]
    ny_empty = (np.array([west, east]), np.array([south, north]), np.full((2, 2), np.nan))
    if lon.size == 0:
        return ny_empty
    coslat = max(math.cos(math.radians(0.5 * (south + north))), 0.05)
    inside = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    n_inside = max(int(inside.sum()), 1)
    area = max((east - west) * coslat * (north - south), 1e-9)
    spacing = math.sqrt(area / n_inside)  # latitude degrees
    dlat = float(res) if res else 0.5 * spacing
    dlat = max(dlat, (north - south) / max_cells, (east - west) * coslat / max_cells)
    dlon = dlat / coslat
    gx = np.arange(west + 0.5 * dlon, east, dlon)
    gy = np.arange(south + 0.5 * dlat, north, dlat)
    if gx.size == 0 or gy.size == 0:
        return ny_empty
    tree = cKDTree(np.column_stack([lon * coslat, lat]))
    mx, my = np.meshgrid(gx, gy)
    dist, idx = tree.query(np.column_stack([mx.ravel() * coslat, my.ravel()]), workers=-1)
    grid = val[idx].reshape(my.shape)
    if max_gap:
        grid = np.where(dist.reshape(my.shape) <= max_gap * max(spacing, dlat), grid, np.nan)
    return gx, gy, grid


def draw_unstructured(ax, lon, lat, values, extent, *, cmap, norm, res: float | None = None, **kwargs):
    """Draw unstructured points on a Cartopy axes as one rasterised nearest-neighbour mesh."""
    import cartopy.crs as ccrs

    gx, gy, grid = regrid_nearest(lon, lat, values, extent, res=res)
    kwargs.setdefault("rasterized", True)
    kwargs.setdefault("shading", "nearest")
    return ax.pcolormesh(gx, gy, np.ma.masked_invalid(grid), transform=ccrs.PlateCarree(),
                         cmap=cmap, norm=norm, **kwargs)


def set_inner_extent(ax, extent, *, n: int = 60):
    """Show the largest projected rectangle inside the ``(west, east, south, north)`` box.

    For a plate carrée axes this is the box itself. For a conic projection the box edges
    are curved in projected coordinates; taking the inner rectangle avoids empty corners
    where the cropped data end.
    """
    import cartopy.crs as ccrs

    west, east, south, north = (float(v) for v in extent)
    if isinstance(ax.projection, ccrs.PlateCarree) or east < west:
        ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
        return
    lons = np.linspace(west, east, n)
    lats = np.linspace(south, north, n)
    proj = ax.projection

    def xy(lo, la):
        pts = proj.transform_points(ccrs.PlateCarree(), np.asarray(lo, float), np.asarray(la, float))
        return pts[:, 0], pts[:, 1]

    left_x, _ = xy(np.full(n, west), lats)
    right_x, _ = xy(np.full(n, east), lats)
    _, bottom_y = xy(lons, np.full(n, south))
    _, top_y = xy(lons, np.full(n, north))
    x0, x1, y0, y1 = left_x.max(), right_x.min(), bottom_y.max(), top_y.min()
    if not (np.isfinite([x0, x1, y0, y1]).all() and x1 > x0 and y1 > y0):
        ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
        return
    ax.set_extent([x0, x1, y0, y1], crs=proj)


def set_grid_ticks(gl, extent, *, nbins: int = 4) -> None:
    """Round, evenly spaced grid-line positions for a small box (also across the antimeridian).

    Cartopy's automatic locator can leave a regional map with a single longitude label;
    this picks about ``nbins`` round values inside ``(west, east, south, north)``.
    """
    if gl is None:
        return
    from matplotlib.ticker import FixedLocator, MaxNLocator

    west, east, south, north = (float(v) for v in extent)
    if east < west:
        east += 360.0
    lons = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(west, east)
    lats = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(south, north)
    lons = [((v + 180.0) % 360.0) - 180.0 for v in lons if west - 1e-9 <= v <= east + 1e-9]
    lats = [v for v in lats if south - 1e-9 <= v <= north + 1e-9]
    gl.xlocator = FixedLocator(lons)
    gl.ylocator = FixedLocator(lats)


def colorbar_beside(fig, axes, mappable, label: str, *, extend: str = "neither",
                    width: float = 0.010, pad: float = 0.006, orientation: str = "vertical"):
    """One colour bar next to a group of map axes, aligned with their drawn map area.

    Vertical bars sit right of the right-most axes and span the group's height; horizontal
    bars sit below the group. Call after every map extent is set.
    """
    axes = [a for a in np.ravel(axes) if a is not None and a.get_visible()]
    if not axes:
        return None
    for a in axes:
        a.apply_aspect()
    boxes = [a.get_position() for a in axes]
    x0 = min(b.x0 for b in boxes)
    x1 = max(b.x1 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    if orientation == "vertical":
        cax = fig.add_axes([x1 + pad, y0 + 0.05 * (y1 - y0), width, 0.9 * (y1 - y0)])
    else:
        cax = fig.add_axes([x0 + 0.1 * (x1 - x0), y0 - pad - width, 0.8 * (x1 - x0), width])
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation, extend=extend)
    cb.set_label(label)
    return cb


_MEMBER_SUFFIX = re.compile(r"^(?P<base>.+?)_(?P<idx>\d+)$")
_INTER_STEP = re.compile(r"^inter_step_(?P<step>\d+)(?:\s*\(sigma=(?P<sigma>[-+0-9.eE]+)\))?$")

_DIFFERENCE_TITLES = {
    "residuals": "Interpolated input minus truth",
    "x_interp_minus_y": "Interpolated input minus truth",
    "residuals_pred": "Interpolated input minus model",
    "x_interp_minus_y_pred": "Interpolated input minus model",
    "y_diff": "Model minus truth",
}


def is_difference_key(key: str) -> bool:
    """True for the difference panels of the region figures (always drawn centred on zero)."""
    k = str(key)
    m = _MEMBER_SUFFIX.match(k)
    base = m.group("base") if m and m.group("base") in _DIFFERENCE_TITLES else k
    return base in _DIFFERENCE_TITLES


def region_panel_title(key: str, *, input_grid: str | None = None, target_grid: str | None = None,
                       truth: str | None = None, input_name: str | None = None) -> str:
    """Readable panel title for a raw region-figure key.

    ``x_0`` -> "Input (O320)", ``x_interp_0`` -> "Input interpolated to O1280",
    ``y_0`` -> "Truth (O1280)", ``y_pred_0`` -> "Model (O1280)", ``residuals_0`` ->
    "Interpolated input minus truth". ``truth``/``input_name`` replace the grid-only
    description when the caller knows the source (for example "ENFO O1280").
    """
    raw = str(key).strip()
    m = _INTER_STEP.match(raw)
    if m:
        text = f"Intermediate state, step {int(m.group('step'))}"
        if m.group("sigma"):
            text += f" (σ = {float(m.group('sigma')):.3g})"
        return text
    base = raw
    mm = _MEMBER_SUFFIX.match(raw)
    if mm and mm.group("base") in ("x", "x_interp", "y", "y_pred", *tuple(_DIFFERENCE_TITLES)):
        base = mm.group("base")
    if base in _DIFFERENCE_TITLES:
        return _DIFFERENCE_TITLES[base]
    inp = input_name or input_grid
    tgt = truth or target_grid
    if base == "x":
        return f"Input ({inp})" if inp else "Input"
    if base == "x_interp":
        return f"Input interpolated to {target_grid}" if target_grid else "Input interpolated to the target grid"
    if base == "y":
        return f"Truth ({tgt})" if tgt else "Truth"
    if base == "y_pred":
        return f"Model ({target_grid})" if target_grid else "Model"
    from .labels import readable_label

    text = readable_label(raw)
    return text[:1].upper() + text[1:]
