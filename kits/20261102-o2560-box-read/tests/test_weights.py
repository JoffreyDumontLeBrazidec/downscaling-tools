import numpy as np
import pytest
from conftest import STORE, need
from o2560_common import area_weights, octahedral_ring_counts


def full_octahedral(n):
    """A full synthetic octahedral grid with n rings per hemisphere (ring i at about 90 - (i - 0.5) * 90 / n)."""
    lat, lon = [], []
    for hemi in (1, -1):
        for i in range(1, n + 1):
            k = 4 * i + 16
            lat += [hemi * (90.0 - (i - 0.5) * 90.0 / n)] * k
            lon += list(np.arange(k) * 360.0 / k)
    return np.array(lat), np.array(lon)


def test_ring_counts_and_weights_on_a_full_grid():
    lat, lon = full_octahedral(32)
    cnt = octahedral_ring_counts(lat, lon, n=32)
    i = np.rint((90 - np.abs(lat)) * 32 / 90 + 0.5)
    np.testing.assert_array_equal(cnt, 4 * i + 16)
    w = area_weights(lat, lon, n=32)
    for la in np.unique(lat):   # summed over a full ring the weight is cos(lat)
        np.testing.assert_allclose(w[lat == la].sum(), np.cos(np.deg2rad(la)), rtol=1e-12)
    c = np.full(lat.size, 3.7)
    assert abs((w * c).sum() / w.sum() - 3.7) < 1e-12
    # the unweighted-by-ring cos(lat) per point over-weights high latitudes: different from the ring weights
    f = np.abs(lat)
    assert abs((w * f).sum() / w.sum() - (np.cos(np.deg2rad(lat)) * f).sum() / np.cos(np.deg2rad(lat)).sum()) > 0.1


def test_wrong_ring_index_fails():
    lat, lon = full_octahedral(32)
    with pytest.raises(ValueError):
        octahedral_ring_counts(lat, lon, n=33)


def test_ring_counts_on_the_o2560_box_store():
    zarr = pytest.importorskip("zarr")
    z = zarr.open(need(f"{STORE}/hres_o2560.zarr"), "r")
    lat = np.asarray(z["latitudes"][:], dtype=np.float64); lon = np.asarray(z["longitudes"][:], dtype=np.float64)
    cnt = octahedral_ring_counts(lat, lon)   # checks the longitude spacing of every ring of the box: 360 / (4 i + 16)
    assert cnt.min() >= 4 * 1 + 16 and cnt.max() <= 4 * 2560 + 16
    w = area_weights(lat, lon)
    assert np.isfinite(w).all() and (w > 0).all()
