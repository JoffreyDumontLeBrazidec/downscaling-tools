"""Simple one-row maps for the suite: input member k | target | model member k, at several zoom levels.

Owner, 23 Nov 2026 (thread "Draw input, truth and 4 km model maps"): the zoom_maps pages were "far too intense"
and "so complex". Owner 10:19 UTC: "I want several levels of zoom, the one you show in this plot is the highest
level, I dont want anything that looks more zoomed in the nodes than this (of course the level of zoom then
depends on the lane), we can also have less zoomed in maps. In general for my eval.cli suite I want all sorts of
zooms."

So the zoom is fixed in OUTPUT GRID POINTS, not degrees: the tightest box is ``CLOSE_POINTS`` output grid
spacings wide (about 1000 km, ~10 deg at mid latitudes, on the 4.4 km lane; about 2000 km on the 9 km lane).
Wider levels are multiples of it (default x1, x2, x4). Nothing tighter than the close level is drawn.

Drawing: every field is interpolated from its own native points (Clough-Tocher, C1; longitude scaled by
cos(latitude)) onto one regular grid, drawn with bilinear smoothing. No grid cells, no triangle facets.
Colour range = 2nd-98th percentile of the target in the box (model when there is no target); wind and rain
start at 0. One colour bar per row. Titles name the field plainly; a target with one field for every member
(e.g. iekm) carries no member number. Input, target and model are separate realisations, not paired.

Scalar + flow (owner 26 Nov 2026, items 1-2 of a colleague's AIFS-ENS deck review): ``flow=True`` overlays 10 m
wind streamlines on every panel, each panel from its own wind (eval/plotting/flow.py: thin, semi-transparent,
width by speed, full width at 30 m/s). Used for tcw + wind, 2t + wind and wind speed + wind.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

# Tightest box width in output grid spacings. 230 x 4.4 km = 1010 km: the 10-degree boxes the owner accepted
# (Alps 5-16 E at 46 N = 850 km, Adriatic 10-20 E at 43.5 N = 810 km, TCs 10 deg at 20 N = 1045 km).
CLOSE_POINTS = 230
DEFAULT_LEVELS = (1, 2, 4)
HEIGHT_RATIO = 0.7          # box height / width (km), a landscape panel
KM_PER_DEG = 111.2
_START_AT_ZERO = {"wind10m", "tp"}
TP_MIN_TOP_MM = 5.0         # smallest colour-bar top for rain (mm per accumulation window)


def spacing_km(lat: np.ndarray, lon: np.ndarray) -> float:
    """Mean grid spacing (km) of a point cloud: sqrt(area per point) over the points' own lat/lon box.

    O2560 gives about 4.4 km, O1280 about 8.8 km. Works on regional crops, unlike a global point count."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    la0, la1 = np.nanpercentile(lat, [1, 99])
    lo0, lo1 = np.nanpercentile(lon, [1, 99])
    k = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    r_km = KM_PER_DEG * 180.0 / np.pi
    area = r_km ** 2 * np.deg2rad(lo1 - lo0) * (np.sin(np.deg2rad(la1)) - np.sin(np.deg2rad(la0)))
    return float(np.sqrt(area / max(int(k.sum()), 1)))


def box_around(lon: float, lat: float, width_km: float) -> list[float]:
    """[lon_min, lon_max, lat_min, lat_max] of a box ``width_km`` wide, HEIGHT_RATIO as tall, centred on a point."""
    half_lat = 0.5 * HEIGHT_RATIO * width_km / KM_PER_DEG
    half_lon = 0.5 * width_km / (KM_PER_DEG * np.cos(np.deg2rad(lat)))
    return [lon - half_lon, lon + half_lon, max(-89.9, lat - half_lat), min(89.9, lat + half_lat)]


def width_km(extent: list[float]) -> float:
    w, e, s, n = extent
    return (e - w) * KM_PER_DEG * np.cos(np.deg2rad(0.5 * (s + n)))


def widen_to_close(extent: list[float], close_km: float) -> tuple[list[float], bool]:
    """The extent, widened around its centre to the close level when it is tighter (both axes)."""
    w, e, s, n = extent
    lon, lat = 0.5 * (w + e), 0.5 * (s + n)
    if width_km(extent) >= close_km - 1e-6 and (n - s) * KM_PER_DEG >= HEIGHT_RATIO * close_km - 1e-6:
        return list(extent), False
    b = box_around(lon, lat, close_km)
    return [min(w, b[0]), max(e, b[1]), min(s, b[2]), max(n, b[3])], True


def _wrap(lon):
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def load_panels(pred_file: Path, variable: str, member: int, model_label: str, flow: bool = False) -> list[dict]:
    """Input (native x), target (y) and model (y_pred) of one member, in display units; target left out if empty.

    ``flow``: each panel also carries its own 10 m wind (``u``, ``v`` in m/s) for the streamline overlay."""
    import xarray as xr

    from eval.evaluators.zoom_maps.core.plot_member_wind_maps import (
        VARIABLES, _field, _member_slice, is_single_field,
    )

    spec = VARIABLES[variable]
    with xr.open_dataset(pred_file) as ds:
        states = [str(s) for s in ds["weather_state"].values]
        lat_l, lon_l = np.asarray(ds["lat_lres"].values, float), _wrap(ds["lon_lres"].values)
        lat_h, lon_h = np.asarray(ds["lat_hres"].values, float), _wrap(ds["lon_hres"].values)
        if flow and not {"10u", "10v"} <= set(states):
            raise ValueError(f"flow needs 10u and 10v; {pred_file.name} carries {states}")
        arrs = {k: _member_slice(ds[k], member) for k in ("x", "y_pred", "y") if k in ds}
        x = _field(arrs["x"], states, spec)
        p = _field(arrs["y_pred"], states, spec)
        y = _field(arrs["y"], states, spec) if "y" in arrs else None
        single = "y" in ds and is_single_field(ds["y"])

    def wind(k):
        if not flow:
            return {}
        a = arrs[k]
        return dict(u=a[:, states.index("10u")], v=a[:, states.index("10v")])

    dl, dh = spacing_km(lat_l, lon_l), spacing_km(lat_h, lon_h)
    out = [dict(title=f"Input member {member} ({dl:.1f} km)", lat=lat_l, lon=lon_l, val=x, dx=dl, **wind("x"))]
    if y is not None and np.isfinite(y).any():
        t = f"Target ({dh:.1f} km)" if single else f"Target member {member} ({dh:.1f} km)"
        out.append(dict(title=t, lat=lat_h, lon=lon_h, val=y, dx=dh, target=True, **wind("y")))
    out.append(dict(title=f"{model_label} member {member} ({dh:.1f} km)", lat=lat_h, lon=lon_h, val=p, dx=dh,
                    **wind("y_pred")))
    return out


def _regrid(lon, lat, val, extent, n_x):
    from scipy.interpolate import CloughTocher2DInterpolator

    w, e, s, n = extent
    lat0 = 0.5 * (s + n)
    c = np.cos(np.deg2rad(lat0))
    pad = 0.05 * max(e - w, n - s)
    lon = np.where(lon < w - 180, lon + 360, np.where(lon > e + 180, lon - 360, lon))
    k = np.isfinite(val) & (lon >= w - pad) & (lon <= e + pad) & (lat >= s - pad) & (lat <= n + pad)
    n_y = max(2, int(round(n_x * (n - s) / ((e - w) * c))))
    gx, gy = np.linspace(w, e, n_x), np.linspace(s, n, n_y)
    if k.sum() < 4:
        return np.full((n_y, n_x), np.nan)
    f = CloughTocher2DInterpolator(np.c_[lon[k] * c, lat[k]], val[k])
    glon, glat = np.meshgrid(gx, gy)
    return f(glon * c, glat)


def draw_row(panels: list[dict], extent: list[float], out: Path, *, variable: str, head: str,
             pct=(2.0, 98.0), n_x: int = 900, dpi: int = 250) -> Path:
    """One figure: the panels side by side on the same box and colour range, one colour bar."""
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    from eval.evaluators.zoom_maps.core.plot_member_wind_maps import VARIABLES
    from eval.plotting.variables import _cmap

    spec = VARIABLES[variable]
    w, e, s, n = extent
    lat0 = 0.5 * (s + n)
    grids = [_regrid(p["lon"], p["lat"], p["val"], extent, n_x) for p in panels]
    ref = next((g for g, p in zip(grids, panels) if p.get("target")), grids[-1])
    if variable in _START_AT_ZERO:              # no negative overshoot of the interpolation
        grids = [np.clip(g, 0.0, None) for g in grids]
        ref = next((g for g, p in zip(grids, panels) if p.get("target")), grids[-1])
    vmin, vmax = (float(v) for v in np.nanpercentile(ref, pct))
    if variable in _START_AT_ZERO:
        vmin = 0.0
    if variable == "tp":
        # a dry box has a 98th percentile near 0 mm, which blows drizzle up to full colour (Iberia 16 Aug 2025):
        # top = 99.5th percentile of the target, never below TP_MIN_TOP_MM
        vmax = max(float(np.nanpercentile(ref, 99.5)), TP_MIN_TOP_MM)
    if not vmax > vmin:
        vmax = vmin + 1.0
    aspect = (e - w) * np.cos(np.deg2rad(lat0)) / (n - s)
    pw = 5.2
    fig, axs = plt.subplots(1, len(panels), figsize=(len(panels) * pw, pw / aspect + 1.2), layout="constrained",
                            subplot_kw=dict(projection=ccrs.PlateCarree()), squeeze=False)
    axs = axs[0]
    cmap = _cmap(spec["cmap"], "viridis")
    for ax, p, g in zip(axs, panels, grids):
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        im = ax.imshow(g, origin="lower", extent=extent, transform=ccrs.PlateCarree(), cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="bilinear")
        if "u" in p:
            from eval.plotting.flow import streamlines_on_map

            streamlines_on_map(ax, p["lon"], p["lat"], p["u"], p["v"], extent, variable=variable, density=1.7)
        ax.coastlines("50m" if width_km(extent) > 3000 else "10m", linewidth=0.6, color="k")
        ax.add_feature(cfeature.BORDERS.with_scale("50m" if width_km(extent) > 3000 else "10m"),
                       linewidth=0.4, edgecolor="0.25")
        ax.set_aspect(1 / np.cos(np.deg2rad(lat0)))
        ax.set_title(p["title"], fontsize=13)
    fig.text(0.5, 1.01, head, ha="center", va="bottom", fontsize=15)
    cb = fig.colorbar(im, ax=list(axs), orientation="horizontal", shrink=0.6, aspect=50,
                      extend="max" if variable in _START_AT_ZERO else "both")
    cb.set_label(spec.get("cbar_label", variable))
    if any("u" in p for p in panels):
        from eval.plotting.flow import SPEED_REF

        fig.text(0.5, -0.01, f"Lines: 10 m wind streamlines of each panel, width grows with speed (full width at "
                 f"{SPEED_REF:g} m s⁻¹)", ha="center", va="top", fontsize=11)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def clip_to_data(extent: list[float], lat: np.ndarray, lon: np.ndarray) -> list[float]:
    """The extent clipped to the lat/lon box covered by the data (a regional crop or the globe)."""
    w, e, s, n = extent
    return [max(w, float(np.nanmin(lon))), min(e, float(np.nanmax(lon))),
            max(s, float(np.nanmin(lat))), min(n, float(np.nanmax(lat)))]


def _fmt_box(extent) -> str:
    w, e, s, n = extent

    def lo(v):
        return f"{abs(v):.1f}°{'E' if v >= 0 else 'W'}"

    def la(v):
        return f"{abs(v):.1f}°{'N' if v >= 0 else 'S'}"
    return f"{lo(w)}-{lo(e)}, {la(s)}-{la(n)}"


def zoom_ladder(pred_file: Path, case, date: str, out_dir: Path, *, model_label: str, member: int = 1,
                levels=DEFAULT_LEVELS, close_points: int = CLOSE_POINTS) -> tuple[list[Path], str]:
    """One row per zoom level around the centre of ``case.extent``; the close level = ``close_points`` spacings.

    ``case.flow`` (case list ``flow: true``) overlays the 10 m wind streamlines on every panel."""
    flow = bool(getattr(case, "flow", False))
    panels = load_panels(pred_file, case.variable, member, model_label, flow=flow)
    dx = panels[-1]["dx"]
    close_km = close_points * dx
    w, e, s, n = case.extent
    lon, lat = 0.5 * (w + e), 0.5 * (s + n)
    from eval.evaluators.zoom_maps.core.plot_member_wind_maps import VARIABLES

    name = VARIABLES[case.variable].get("subtitle", case.variable)
    if flow:
        name += " + 10 m wind"
    when = f"{date[:4]}-{date[4:6]}-{date[6:]} 00 UTC +{case.step} h"
    rows, seen = [], []
    for f in levels:
        ext = clip_to_data(box_around(lon, lat, close_km * f), panels[-1]["lat"], panels[-1]["lon"])
        if seen and np.allclose(ext, seen[-1], atol=0.05):
            break                                  # the data's whole domain is already drawn
        seen.append(ext)
        head = f"{name}, {case.name}, {when}, {_fmt_box(ext)} (about {width_km(ext):.0f} km wide)"
        tag = f"{case.name}_{case.variable}{'_flow' if flow else ''}_x{f}.png"
        rows.append(draw_row(panels, ext, Path(out_dir) / tag, variable=case.variable, head=head))
    return rows, f"close level {close_km:.0f} km = {close_points} x {dx:.1f} km, levels x{', x'.join(map(str, levels))}"
