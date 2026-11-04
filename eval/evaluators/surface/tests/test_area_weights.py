"""Area weights on grids stored as latitude rings (eval.evaluators.surface.core.compute.ring_area_weights and
_area_weights): an octahedral grid's area mean of a latitude function matches the sphere's, a box cut keeps the
global per-point weights, and a regular grid keeps cos(lat)."""
import numpy as np
import pytest
import xarray as xr

from eval.evaluators.surface.core.compute import _area_weights, ring_area_weights


def octahedral(n):
    lat, lon = [], []
    for hemi in (1, -1):
        for i in range(1, n + 1):
            k = 4 * i + 16
            lat += [hemi * (90.0 - (i - 0.5) * 90.0 / n)] * k
            lon += list(np.arange(k) * 360.0 / k)
    return np.array(lat), np.array(lon)


def test_octahedral_ring_sums_follow_cos_lat():
    lat, lon = octahedral(48)
    w = ring_area_weights(lat, lon)
    for la in np.unique(lat):
        np.testing.assert_allclose(w[lat == la].sum(), 360.0 * np.cos(np.deg2rad(la)), rtol=1e-9)   # k points x 360/k each


def test_octahedral_area_mean_matches_the_sphere():
    lat, lon = octahedral(96)
    f = np.sin(np.deg2rad(lat)) ** 2          # area mean on the sphere: 1/3
    w = ring_area_weights(lat, lon)
    assert abs((w * f).sum() / w.sum() - 1 / 3) < 2e-3
    c = np.cos(np.deg2rad(lat))               # cos(lat) per point: biased towards the poles on this grid
    assert abs((c * f).sum() / c.sum() - 1 / 3) > 0.02


def test_box_cut_keeps_the_global_weights():
    lat, lon = octahedral(96)
    w = ring_area_weights(lat, lon)
    l180 = (lon + 180) % 360 - 180
    m = (lat > 10) & (lat < 70) & (l180 > -80) & (l180 < 40)   # crosses longitude 0
    np.testing.assert_allclose(ring_area_weights(lat[m], lon[m]), w[m], rtol=1e-9)


def test_regular_grid_keeps_cos_lat():
    la, lo = np.meshgrid(np.arange(-80.0, 80.1, 2.0), np.arange(0.0, 360.0, 2.5), indexing="ij")
    w = ring_area_weights(la.ravel(), lo.ravel())
    c = np.cos(np.deg2rad(la.ravel()))
    np.testing.assert_allclose(w / w.sum(), c / c.sum(), rtol=1e-12)


def test_single_point_ring_takes_a_neighbour_spacing():
    lat = np.array([10.0, 10.0, 10.0, 20.0, 30.0, 30.0, 30.0]); lon = np.array([0.0, 1.0, 2.0, 5.0, 0.0, 2.0, 4.0])
    w = ring_area_weights(lat, lon)
    assert np.isfinite(w).all() and w[3] > 0


def test_area_weights_on_a_prediction_dataset_uses_the_rings():
    lat, lon = octahedral(32)
    ds = xr.Dataset({"lat_hres": ("grid_point_hres", lat), "lon_hres": ("grid_point_hres", lon)})
    w = _area_weights(ds, lat.size)
    np.testing.assert_allclose(w.sum(), 1.0)
    ref = ring_area_weights(lat, lon)
    np.testing.assert_allclose(w, ref / ref.sum(), rtol=1e-12)
