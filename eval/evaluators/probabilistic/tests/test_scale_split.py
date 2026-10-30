"""The low-pass P_L: transfer function in km, constants, edges, poles and dateline."""
from __future__ import annotations

import numpy as np
import pytest

from eval.evaluators.probabilistic.core.scale_split import (
    KM_PER_DEG,
    LowPassOperator,
    box_edge_mask,
    gaussian_power_transfer,
    inner_mask,
    is_global,
    sigma_km_for_cut,
)


def _box_points(lat0=30.0, lat1=60.0, lon0=-40.0, lon1=20.0, spacing_km=9.0, seed=0):
    """A 9 km point set over a mid-latitude box, jittered so it is not a lattice."""
    rng = np.random.default_rng(seed)
    dlat = spacing_km / KM_PER_DEG
    lats, lons = [], []
    for la in np.arange(lat0, lat1, dlat):
        lo = np.arange(lon0, lon1, spacing_km / (KM_PER_DEG * np.cos(np.deg2rad(la))))
        lats.append(np.full(lo.size, la))
        lons.append(lo)
    lat = np.concatenate(lats) + rng.uniform(-0.1, 0.1, sum(a.size for a in lats)) * dlat
    return lat, np.concatenate(lons)


@pytest.fixture(scope="module")
def box():
    lat, lon = _box_points()
    op = LowPassOperator(lat, lon, np.cos(np.deg2rad(lat)), cut_km=400.0)
    inner = inner_mask(lat, lon, 400.0)
    x_km = (lon - lon.mean()) * KM_PER_DEG * np.cos(np.deg2rad(lat))
    y_km = (lat - lat.mean()) * KM_PER_DEG
    return op, inner, x_km, y_km


def test_sigma_of_the_cut():
    assert sigma_km_for_cut(400.0) == pytest.approx(53.0, abs=0.1)
    assert sigma_km_for_cut(250.0) == pytest.approx(33.1, abs=0.1)
    assert sigma_km_for_cut(600.0) == pytest.approx(79.5, abs=0.1)
    assert gaussian_power_transfer(400.0, sigma_km_for_cut(400.0)) == pytest.approx(0.5)


@pytest.mark.parametrize("wavelength_km,lo,hi", [
    (100.0, 0.0, 0.002),      # little leak below the cut
    (250.0, 0.14, 0.19),      # Gaussian 0.17
    (400.0, 0.47, 0.51),      # half power at the cut (Gaussian 0.50)
    (600.0, 0.71, 0.75),      # Gaussian 0.73
    (1000.0, 0.87, 0.91),     # Gaussian 0.90
    (2000.0, 0.96, 0.99),     # near 1
])
def test_transfer_function_in_km(box, wavelength_km, lo, hi):
    op, inner, x_km, y_km = box
    for coord in (x_km, y_km, (x_km + y_km) / np.sqrt(2.0)):
        f = np.sin(2 * np.pi * coord / wavelength_km)
        kept = np.mean(op.apply(f)[inner] ** 2) / np.mean(f[inner] ** 2)
        assert lo <= kept <= hi, (wavelength_km, kept)


def test_near_idempotent_away_from_the_cut(box):
    op, inner, x_km, _ = box
    f = np.sin(2 * np.pi * x_km / 2000.0)
    once = op.apply(f)
    twice = op.apply(once)
    assert np.mean((twice - once)[inner] ** 2) / np.mean(once[inner] ** 2) < 0.03


def test_constant_is_kept_and_parts_add_up(box):
    op, inner, x_km, _ = box
    const = op.apply(np.full(x_km.size, 3.0))
    assert np.allclose(const[inner], 3.0, atol=1e-6)
    assert np.allclose(const, 3.0, atol=1e-2)          # one-sided averages at the box edge stay exact for a constant
    f = np.sin(2 * np.pi * x_km / 300.0) + 0.1 * np.cos(2 * np.pi * x_km / 50.0)
    large, fine = op.split(f)
    np.testing.assert_allclose(large + fine, f)


def test_nan_points_carry_no_weight(box):
    op, inner, x_km, _ = box
    f = np.sin(2 * np.pi * x_km / 1000.0)
    g = f.copy()
    g[::7] = np.nan
    out = op.apply(g)
    assert np.all(np.isnan(out[::7])) and np.all(np.isfinite(np.delete(out, np.arange(0, f.size, 7))))
    ok = np.isfinite(g) & inner
    assert np.sqrt(np.mean((out[ok] - op.apply(f)[ok]) ** 2)) < 0.01


def test_inner_mask_flags_a_margin_only_on_regional_boxes():
    lat, lon = _box_points()
    inner = inner_mask(lat, lon, 400.0)
    assert 0.3 < inner.mean() < 0.8
    assert not inner[np.argmin(lat)] and not inner[np.argmax(lon)]
    assert inner[np.argmin(np.hypot(lat - 45.0, lon + 10.0))]
    glat = np.linspace(-89.9, 89.9, 400)
    glon = np.linspace(-180.0, 179.0, 400)
    assert is_global(glat, glon) and inner_mask(glat, glon, 400.0).all()
    assert inner_mask(lat, lon, 0.0).all()



def test_box_edge_mask_drops_a_one_degree_strip():
    lat, lon = _box_points()
    keep = box_edge_mask(lat, lon, 1.0)
    assert keep[np.argmin(np.hypot(lat - 45.0, lon + 10.0))]
    assert not keep[lat < lat.min() + 1.0].any() and not keep[lon > lon.max() - 1.0].any()
    assert keep[(lat > lat.min() + 1.01) & (lat < lat.max() - 1.01)
                & (lon > lon.min() + 1.01) & (lon < lon.max() - 1.01)].all()
    glat = np.linspace(-89.9, 89.9, 400)
    glon = np.linspace(-180.0, 179.0, 400)
    assert box_edge_mask(glat, glon, 1.0).all() and box_edge_mask(lat, lon, 0.0).all()

def test_global_coarse_set_handles_poles_and_dateline():
    # a coarse global reduced-grid-like set (about 110 km): empty 25 km cells everywhere,
    # which the normalised convolution must shrug off; a zonal wavenumber-4 wave is large scale
    rows = np.arange(-89.5, 90.0, 1.0)
    lat = np.concatenate([np.full(max(int(round(360 * np.cos(np.deg2rad(r)))), 4), r) for r in rows])
    lon = np.concatenate([np.linspace(-180.0, 180.0, max(int(round(360 * np.cos(np.deg2rad(r)))), 4), endpoint=False)
                          for r in rows])
    op = LowPassOperator(lat, lon, np.cos(np.deg2rad(lat)), cut_km=400.0)
    assert op.global_coverage
    const = op.apply(np.ones(lat.size))
    assert np.allclose(const, 1.0, atol=1e-6)
    f = np.cos(4 * np.deg2rad(lon)) * np.cos(np.deg2rad(lat)) ** 2
    out = op.apply(f)
    assert np.all(np.isfinite(out))
    mid = np.abs(lat) < 60.0
    assert np.mean(out[mid] ** 2) / np.mean(f[mid] ** 2) > 0.9
    # the dateline is not an edge: the field is continuous across it
    near = np.abs(np.abs(lon) - 180.0) < 1.0
    assert np.sqrt(np.mean((out[near] - f[near]) ** 2)) < 0.05


def test_describe_and_mismatch():
    lat, lon = _box_points(spacing_km=50.0)
    op = LowPassOperator(lat, lon, cut_km=250.0, grid_km=50.0)
    d = op.describe()
    assert d["cut_km"] == 250.0 and d["sigma_km"] == pytest.approx(33.1, abs=0.1) and d["grid_km"] == 50.0
    with pytest.raises(ValueError):
        op.apply(np.zeros(lat.size + 1))
