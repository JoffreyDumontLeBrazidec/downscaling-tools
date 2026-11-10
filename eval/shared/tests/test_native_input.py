"""The native input reader of rule 41: nearest native point per output point, disk cache, fallback."""
from __future__ import annotations

import numpy as np
import pytest

from eval.shared import native_input as ni


def _grid(lat0, lat1, lon0, lon1, step):
    la, lo = np.meshgrid(np.arange(lat0, lat1 + 1e-9, step), np.arange(lon0, lon1 + 1e-9, step), indexing="ij")
    return la.ravel(), lo.ravel()


def _write(path, *, native=True, members=3):
    netCDF4 = pytest.importorskip("netCDF4")
    la_o, lo_o = _grid(40.1, 50.1, -4.9, 5.1, 0.25)  # off the half-way points: no ties
    la_i, lo_i = _grid(39.0, 51.0, -6.0, 6.0, 1.0)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("sample", 1)
        ds.createDimension("ensemble_member", members)
        ds.createDimension("grid_point_hres", la_o.size)
        ds.createDimension("weather_state", 2)
        ds.createVariable("lat_hres", "f8", ("grid_point_hres",))[:] = la_o
        ds.createVariable("lon_hres", "f8", ("grid_point_hres",))[:] = np.mod(lo_o, 360.0)
        ws = ds.createVariable("weather_state", str, ("weather_state",))
        ws[0], ws[1] = "2t", "tp"
        full = ("sample", "ensemble_member", "grid_point_hres", "weather_state")
        ds.createVariable("y_pred", "f4", full)[:] = 0.0
        ds.createVariable("x_interp", "f4", ("sample", "grid_point_hres", "weather_state"))[:] = -1.0
        if native:
            ds.createDimension("grid_point_lres", la_i.size)
            ds.createVariable("lat_lres", "f8", ("grid_point_lres",))[:] = la_i
            ds.createVariable("lon_lres", "f8", ("grid_point_lres",))[:] = np.mod(lo_i, 360.0)
            x = ds.createVariable("x", "f4", ("sample", "ensemble_member", "grid_point_lres", "weather_state"))
            v = np.zeros((1, members, la_i.size, 2), np.float32)
            v[0, :, :, 0] = np.arange(la_i.size)[None, :] + 1000.0 * np.arange(members)[:, None]
            v[0, :, :, 1] = 7.0
            x[:] = v
    return (la_o, lo_o), (la_i, lo_i)


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv(ni.CACHE_ENV, str(tmp_path / "idxcache"))
    ni._MEMO.clear()


def test_nearest_index_matches_brute_force():
    rng = np.random.default_rng(0)
    la_i, lo_i = _grid(-10.0, 10.0, 170.0, 190.0, 1.0)  # crosses the dateline
    la_o, lo_o = rng.uniform(-9, 9, 500), rng.uniform(171, 189, 500)
    idx = ni.nearest_native_index(la_i, np.mod(lo_i, 360.0), la_o, lo_o - 360.0 * (lo_o > 180))
    d = ni._xyz(la_o, lo_o) @ ni._xyz(la_i, lo_i).T  # largest dot product = smallest great-circle distance
    np.testing.assert_array_equal(idx, d.argmax(1))


def test_index_is_cached_on_disk_and_memoised(tmp_path):
    la_i, lo_i = _grid(0.0, 5.0, 0.0, 5.0, 1.0)
    la_o, lo_o = _grid(0.0, 5.0, 0.0, 5.0, 0.3)
    idx = ni.nearest_native_index(la_i, lo_i, la_o, lo_o)
    files = list((tmp_path / "idxcache").glob("*.npz"))
    assert len(files) == 1
    assert ni.nearest_native_index(la_i, lo_i, la_o, lo_o) is idx  # memo
    ni._MEMO.clear()
    np.testing.assert_array_equal(ni.nearest_native_index(la_i, lo_i, la_o, lo_o), idx)  # from disk
    # another output grid: another key
    ni.nearest_native_index(la_i, lo_i, la_o[:10], lo_o[:10])
    assert len(list((tmp_path / "idxcache").glob("*.npz"))) == 2


def test_unwritable_cache_never_fails(tmp_path, monkeypatch):
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    monkeypatch.setenv(ni.CACHE_ENV, str(blocker / "sub"))  # a directory under a file: mkdir fails
    la_i, lo_i = _grid(0.0, 5.0, 0.0, 5.0, 1.0)
    idx = ni.nearest_native_index(la_i, lo_i, [0.1, 4.9], [0.1, 4.9])
    assert idx.tolist() == [0, la_i.size - 1]
    corrupt = tmp_path / "c"
    corrupt.mkdir()
    monkeypatch.setenv(ni.CACHE_ENV, str(corrupt))
    ni._MEMO.clear()
    key = ni._key(*(np.asarray(a, np.float64).reshape(-1) for a in (la_i, lo_i, [0.1, 4.9], [0.1, 4.9])))
    (corrupt / f"{key}.npz").write_bytes(b"not an npz")
    assert ni.nearest_native_index(la_i, lo_i, [0.1, 4.9], [0.1, 4.9]).tolist() == [0, la_i.size - 1]


@pytest.mark.parametrize("opener", ["netcdf4", "xarray"])
def test_reader_returns_native_values_at_the_nearest_native_point(tmp_path, opener):
    path = tmp_path / "p.nc"
    (la_o, lo_o), (la_i, lo_i) = _write(path)
    if opener == "netcdf4":
        import netCDF4

        ds = netCDF4.Dataset(path)
    else:
        import xarray as xr

        ds = xr.open_dataset(path)
    try:
        assert ni.has_native_input(ds)
        r = ni.InputReader(ds)
        assert r.source == ni.SOURCE_NATIVE and r.native and r.n_members == 3
        a = r.channel("2t")
        assert a.shape == (3, la_o.size)
        want = ni.nearest_native_index(la_i, lo_i, la_o, lo_o)
        np.testing.assert_array_equal(a[0], want.astype(float))     # value = native index: no averaging
        np.testing.assert_array_equal(a[2], want + 2000.0)
        assert set(np.unique(a[0])) <= set(np.arange(la_i.size, dtype=float))
        np.testing.assert_array_equal(r.channel(1), 7.0)
        sub = ni.InputReader(ds, points=np.arange(5)).channel("2t")
        np.testing.assert_array_equal(sub, a[:, :5])
        old, src = ni.read_input(ds, "2t", native_input=False)
        assert ni.is_interpolated(src) and "native_input: false" in src
        assert old.shape == (1, la_o.size) and (old == -1.0).all()
    finally:
        ds.close()


def test_reader_falls_back_to_x_interp_without_native_x(tmp_path, caplog):
    import netCDF4
    import logging

    path = tmp_path / "p.nc"
    (la_o, _), _ = _write(path, native=False)
    with netCDF4.Dataset(path) as ds:
        assert not ni.has_native_input(ds)
        a, src = ni.read_input(ds, "tp")
        assert src == ni.SOURCE_INTERP and a.shape == (1, la_o.size)
        assert not ni.InputReader(ds, fallback=("x_interp_members",)).available
    with caplog.at_level(logging.WARNING):
        line = ni.warn_fallback(logging.getLogger("t"), "demo", src)
    assert "rule 41" in line and "rule 41" in caplog.text
