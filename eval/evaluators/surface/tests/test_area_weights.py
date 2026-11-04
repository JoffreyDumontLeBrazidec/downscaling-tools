import numpy as np
import xarray as xr

from eval.evaluators.surface.core.compute import _area_weights, _point_area_weights


def octahedral(n_rings_half: int):
    """Octahedral reduced Gaussian-like grid: ring i from the pole holds 4i + 16 points (latitudes equally spaced)."""
    lats, lons = [], []
    edges = np.linspace(90, 0, n_rings_half + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for i, la in enumerate(centres, start=1):
        n = 4 * i + 16
        for sgn in (1, -1):
            lats.append(np.full(n, sgn * la)); lons.append(np.arange(n) * 360.0 / n)
    return np.concatenate(lats), np.concatenate(lons)


def test_ring_sums_follow_cos_lat_on_reduced_grid():
    lat, lon = octahedral(80)
    w = _point_area_weights(lat, lon)
    for la in np.unique(lat)[::17]:
        ring = w[lat == la].sum()
        assert np.isclose(ring / (360.0 * np.cos(np.deg2rad(la))), 1.0, rtol=1e-9)


def test_reduced_grid_area_fraction_is_exact():
    lat, lon = octahedral(80)
    w = _point_area_weights(lat, lon); w = w / w.sum()
    north = w[lat >= 60].sum()
    exact = (1 - np.sin(np.deg2rad(60))) / 2
    assert abs(north - exact) < 2e-3
    naive = np.cos(np.deg2rad(lat)); naive = naive / naive.sum()
    assert abs(naive[lat >= 60].sum() - exact) > 5 * abs(north - exact)


def test_regional_cut_and_regular_grid():
    lat, lon = octahedral(80)
    lon180 = np.where(lon > 180, lon - 360, lon)
    box = (lat > 10) & (lat < 70) & (lon180 > -80) & (lon180 < 40)
    w = _point_area_weights(lat[box], lon[box]); w = w / w.sum()
    la = lat[box]
    exact = (np.sin(np.deg2rad(70)) - np.sin(np.deg2rad(60))) / (np.sin(np.deg2rad(70)) - np.sin(np.deg2rad(10)))
    assert abs(w[la >= 60].sum() - exact) < 0.01
    glat, glon = np.meshgrid(np.arange(-89.5, 90, 1.0), np.arange(0, 360, 1.0), indexing="ij")
    wr = _point_area_weights(glat.ravel(), glon.ravel())
    c = np.cos(np.deg2rad(glat.ravel()))
    assert np.allclose(wr / wr.sum(), c / c.sum())


def test_area_weights_uses_lon_from_dataset():
    lat, lon = octahedral(40)
    n = lat.size
    ds = xr.Dataset({"lat_hres": ("grid_point_hres", lat), "lon_hres": ("grid_point_hres", lon)})
    w = _area_weights(ds, n)
    assert np.isclose(w.sum(), 1.0)
    ref = _point_area_weights(lat, lon)
    assert np.allclose(w, ref / ref.sum())
    ds2 = xr.Dataset({"lat_hres": ("grid_point_hres", lat)})
    c = np.cos(np.deg2rad(lat))
    assert np.allclose(_area_weights(ds2, n), c / c.sum())
