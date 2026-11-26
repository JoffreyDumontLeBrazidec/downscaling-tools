"""eval.plotting.flow: the streamline overlay of the map templates (owner 26 Nov 2026)."""
from __future__ import annotations

import numpy as np
import pytest

from eval.plotting import flow


def test_width_grows_with_speed_and_saturates():
    w = flow.width_from_speed(np.array([0.0, flow.SPEED_REF / 2, flow.SPEED_REF, 3 * flow.SPEED_REF, np.nan]))
    assert w[0] == pytest.approx(flow.LW_MIN) and w[4] == pytest.approx(flow.LW_MIN)
    assert w[1] == pytest.approx(0.5 * (flow.LW_MIN + flow.LW_MAX))
    assert w[2] == pytest.approx(flow.LW_MAX) and w[3] == pytest.approx(flow.LW_MAX)


def test_line_colour_dark_on_maps_white_at_one_end():
    assert flow.line_color("wind10m") == "white"
    assert flow.line_color("tcw") == flow.DARK_LINE and flow.line_color("2t") == flow.DARK_LINE


def test_regular_grid_reproduces_a_linear_field_and_leaves_gaps_empty():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 10, 20000), rng.uniform(0, 10, 20000)
    keep = ~((x > 6) & (y > 6))                       # a hole in the data
    gx, gy, (g,) = flow.to_regular(x[keep], y[keep], [2 * x[keep] + y[keep]], (0, 10), (0, 10), n_long=50)
    gxx, gyy = np.meshgrid(gx, gy)
    inside = (gxx > 0.5) & (gxx < 5.5) & (gyy > 0.5) & (gyy < 9.5)
    assert np.nanmax(np.abs(g - (2 * gxx + gyy))[inside]) < 0.3
    assert np.isnan(g[(gxx > 7.5) & (gyy > 7.5)]).all()


def test_streamlines_follow_a_diagonal_wind_at_high_latitude():
    """A south-westerly (u = v) at 60 N is drawn at 45 degrees on a map with the 1/cos(lat) aspect."""
    pytest.importorskip("cartopy")
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    lat, lon = np.meshgrid(np.arange(55, 65, 0.1), np.arange(-10, 10, 0.1), indexing="ij")
    lat, lon = lat.ravel(), lon.ravel()
    fig = plt.figure()
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    extent = [-5, 5, 58, 62]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    sp = flow.streamlines_on_map(ax, lon, lat, np.full(lat.size, 10.0), np.full(lat.size, 10.0), extent)
    seg = np.concatenate(sp.lines.get_segments())
    d = np.diff(seg, axis=0)
    d = d[np.hypot(*d.T) > 0]
    # in degrees: dlon / dlat = 1 / cos(60) = 2; in km on the map it is 45 degrees
    ratio = np.median(d[:, 0] / d[:, 1])
    assert ratio == pytest.approx(1 / np.cos(np.deg2rad(60.0)), rel=0.1)
    plt.close(fig)
