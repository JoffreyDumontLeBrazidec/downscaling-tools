import sys
import types

import numpy as np
import pytest
import xarray as xr

from manual_inference.input_data_construction import bundle


def _make_sfc_dataset(point_count: int, *, prefix: float) -> xr.Dataset:
    coords = {"values": np.arange(point_count, dtype=np.int32)}
    data_vars = {
        "u10": ("values", np.full(point_count, prefix + 1, dtype=np.float32)),
        "v10": ("values", np.full(point_count, prefix + 2, dtype=np.float32)),
        "d2m": ("values", np.full(point_count, prefix + 3, dtype=np.float32)),
        "t2m": ("values", np.full(point_count, prefix + 4, dtype=np.float32)),
        "msl": ("values", np.full(point_count, prefix + 5, dtype=np.float32)),
        "skt": ("values", np.full(point_count, prefix + 6, dtype=np.float32)),
        "sp": ("values", np.full(point_count, prefix + 7, dtype=np.float32)),
        "tcw": ("values", np.full(point_count, prefix + 8, dtype=np.float32)),
        "latitude": ("values", np.linspace(10, 20, point_count, dtype=np.float32)),
        "longitude": ("values", np.linspace(30, 40, point_count, dtype=np.float32)),
        "valid_time": ((), np.datetime64("2023-08-21T00:00:00")),
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)


def _make_pl_dataset(point_count: int) -> xr.Dataset:
    levels = np.asarray([50, 100, 200], dtype=np.int32)
    shape = (levels.size, point_count)
    coords = {
        "level": levels,
        "values": np.arange(point_count, dtype=np.int32),
    }
    data_vars = {
        "q": (("level", "values"), np.full(shape, 1.0, dtype=np.float32)),
        "t": (("level", "values"), np.full(shape, 2.0, dtype=np.float32)),
        "u": (("level", "values"), np.full(shape, 3.0, dtype=np.float32)),
        "v": (("level", "values"), np.full(shape, 4.0, dtype=np.float32)),
        "w": (("level", "values"), np.full(shape, 5.0, dtype=np.float32)),
        "z": (("level", "values"), np.full(shape, 6.0, dtype=np.float32)),
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)


def _make_hres_dataset(point_count: int) -> xr.Dataset:
    coords = {"values": np.arange(point_count, dtype=np.int32)}
    data_vars = {
        "z": ("values", np.full(point_count, 7.0, dtype=np.float32)),
        "lsm": ("values", np.full(point_count, 8.0, dtype=np.float32)),
        "latitude": ("values", np.linspace(50, 60, point_count, dtype=np.float32)),
        "longitude": ("values", np.linspace(70, 80, point_count, dtype=np.float32)),
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)


def _install_fake_earthkit(monkeypatch, datasets: dict[str, xr.Dataset]) -> None:
    class _FakeSource:
        def __init__(self, ds: xr.Dataset):
            self._ds = ds

        def to_xarray(self, engine: str = "cfgrib") -> xr.Dataset:
            assert engine == "cfgrib"
            return self._ds

    def _from_source(kind: str, path: str) -> _FakeSource:
        assert kind == "file"
        return _FakeSource(datasets[path])

    fake_data_module = types.ModuleType("earthkit.data")
    fake_data_module.from_source = _from_source
    fake_earthkit = types.ModuleType("earthkit")
    fake_earthkit.data = fake_data_module
    monkeypatch.setitem(sys.modules, "earthkit", fake_earthkit)
    monkeypatch.setitem(sys.modules, "earthkit.data", fake_data_module)


def test_cleanup_empty_cfgrib_indexes_removes_zero_byte_sidecars(tmp_path):
    grib = tmp_path / "sample.grib"
    grib.write_text("stub", encoding="utf-8")
    empty_idx = tmp_path / "sample.grib.5b7b6.idx"
    empty_idx.write_bytes(b"")
    good_idx = tmp_path / "sample.grib.abcde.idx"
    good_idx.write_bytes(b"not-empty")

    bundle._cleanup_empty_cfgrib_indexes(grib)

    assert not empty_idx.exists()
    assert good_idx.exists()


def test_build_input_bundle_allows_missing_target_with_explicit_override(tmp_path, monkeypatch):
    datasets = {
        "lres_sfc.grib": _make_sfc_dataset(3, prefix=0.0),
        "lres_pl.grib": _make_pl_dataset(3),
        "hres.grib": _make_hres_dataset(4),
    }
    _install_fake_earthkit(monkeypatch, datasets)

    out_path = tmp_path / "bundle.nc"
    bundle.build_input_bundle_from_grib(
        lres_sfc_grib="lres_sfc.grib",
        lres_pl_grib="lres_pl.grib",
        hres_grib="hres.grib",
        out_nc=out_path,
        require_target_fields=False,
    )

    ds = xr.open_dataset(out_path)
    try:
        assert ds.attrs["has_target_hres_fields"] == "no"
        assert (
            ds.attrs["missing_target_policy"]
            == "bundle_without_target_hres_due_to_allow_missing_target_unsafe"
        )
        assert "Prediction-only" in ds.attrs["description"]
    finally:
        ds.close()


def test_build_input_bundle_requires_target_by_default(tmp_path, monkeypatch):
    datasets = {
        "lres_sfc.grib": _make_sfc_dataset(3, prefix=0.0),
        "lres_pl.grib": _make_pl_dataset(3),
        "hres.grib": _make_hres_dataset(4),
    }
    _install_fake_earthkit(monkeypatch, datasets)

    with pytest.raises(ValueError, match="No target_hres_\\* fields were added to bundle"):
        bundle.build_input_bundle_from_grib(
            lres_sfc_grib="lres_sfc.grib",
            lres_pl_grib="lres_pl.grib",
            hres_grib="hres.grib",
            out_nc=tmp_path / "bundle.nc",
            require_target_fields=True,
        )


def test_build_input_bundle_unsafe_skips_auto_inferred_target_gribs(tmp_path, monkeypatch):
    work_dir = tmp_path / "gribs"
    work_dir.mkdir()
    hres_path = work_dir / "enfo_o320_0001_date20230821_time0000_step24to120_sfc.grib"
    sfc_y_path = work_dir / "enfo_o320_0001_date20230821_time0000_mem1to10_step24to120_sfc_y.grib"
    pl_y_path = work_dir / "enfo_o320_0001_date20230821_time0000_mem1to10_step24to120_pl_y.grib"
    hres_path.touch()
    sfc_y_path.touch()
    pl_y_path.touch()

    datasets = {
        str(work_dir / "lres_sfc.grib"): _make_sfc_dataset(3, prefix=0.0),
        str(work_dir / "lres_pl.grib"): _make_pl_dataset(3),
        str(hres_path): _make_hres_dataset(4),
    }
    _install_fake_earthkit(monkeypatch, datasets)

    out_path = tmp_path / "bundle.nc"
    bundle.build_input_bundle_from_grib(
        lres_sfc_grib=work_dir / "lres_sfc.grib",
        lres_pl_grib=work_dir / "lres_pl.grib",
        hres_grib=hres_path,
        out_nc=out_path,
        require_target_fields=False,
    )

    ds = xr.open_dataset(out_path)
    try:
        assert ds.attrs["has_target_hres_fields"] == "no"
    finally:
        ds.close()


def test_build_input_bundle_auto_infers_humberto_target_gribs(tmp_path, monkeypatch):
    work_dir = tmp_path / "gribs"
    work_dir.mkdir()
    hres_path = work_dir / "enfo_o96_0001_date20250926_time0000_step24to120_sfc.grib"
    sfc_y_path = work_dir / "iekm_o96_iekm_date20250926_time0000_step24to120_sfc_y.grib"
    pl_y_path = work_dir / "iekm_o96_iekm_date20250926_time0000_step24to120_pl_y.grib"
    hres_path.touch()
    sfc_y_path.touch()
    pl_y_path.touch()

    datasets = {
        str(work_dir / "lres_sfc.grib"): _make_sfc_dataset(3, prefix=0.0),
        str(work_dir / "lres_pl.grib"): _make_pl_dataset(3),
        str(hres_path): _make_hres_dataset(4),
        str(sfc_y_path): _make_sfc_dataset(4, prefix=100.0),
        str(pl_y_path): _make_pl_dataset(4),
    }
    _install_fake_earthkit(monkeypatch, datasets)

    out_path = tmp_path / "bundle.nc"
    bundle.build_input_bundle_from_grib(
        lres_sfc_grib=work_dir / "lres_sfc.grib",
        lres_pl_grib=work_dir / "lres_pl.grib",
        hres_grib=hres_path,
        out_nc=out_path,
        step_hours=24,
    )

    ds = xr.open_dataset(out_path)
    try:
        assert ds.attrs["has_target_hres_fields"] == "yes"
        assert ds.attrs["source_target_sfc"] == str(sfc_y_path)
        assert ds.attrs["source_target_pl"] == str(pl_y_path)
        assert "target_hres_10u" in ds
        assert "target_hres_q" in ds
    finally:
        ds.close()


def test_build_input_bundle_backfills_missing_lres_precip(tmp_path, monkeypatch):
    datasets = {
        "lres_sfc.grib": _make_sfc_dataset(3, prefix=0.0),
        "lres_pl.grib": _make_pl_dataset(3),
        "hres.grib": _make_hres_dataset(4),
    }
    _install_fake_earthkit(monkeypatch, datasets)

    out_path = tmp_path / "bundle.nc"
    bundle.build_input_bundle_from_grib(
        lres_sfc_grib="lres_sfc.grib",
        lres_pl_grib="lres_pl.grib",
        hres_grib="hres.grib",
        out_nc=out_path,
        require_target_fields=False,
    )

    ds = xr.open_dataset(out_path)
    try:
        expected_zero_vars = [
            "in_lres_cp",
            "in_lres_hcc",
            "in_lres_lcc",
            "in_lres_mcc",
            "in_lres_ssrd",
            "in_lres_strd",
            "in_lres_tcc",
            "in_lres_tp",
        ]
        for name in expected_zero_vars:
            assert name in ds
            assert np.allclose(ds[name].values, 0.0)
    finally:
        ds.close()


def test_load_inputs_from_bundle_numpy_backfills_legacy_lres_precip(monkeypatch):
    ds = xr.Dataset(
        data_vars={
            "in_lres_10u": ("point_lres", np.array([1.0, 2.0], dtype=np.float32)),
            "in_hres_z": ("point_hres", np.array([5.0, 6.0, 7.0], dtype=np.float32)),
        },
        coords={
            "point_lres": np.arange(2, dtype=np.int32),
            "point_hres": np.arange(3, dtype=np.int32),
            "lat_lres": ("point_lres", np.array([10.0, 11.0], dtype=np.float32)),
            "lon_lres": ("point_lres", np.array([20.0, 21.0], dtype=np.float32)),
            "lat_hres": ("point_hres", np.array([30.0, 31.0, 32.0], dtype=np.float32)),
            "lon_hres": ("point_hres", np.array([40.0, 41.0, 42.0], dtype=np.float32)),
        },
        attrs={"case_valid_time": "2023-08-21T00:00:00"},
    )
    monkeypatch.setattr(bundle, "fill_hres_features", lambda *args, **kwargs: None)

    x_lres, x_hres, *_ = bundle.load_inputs_from_bundle_numpy(
        ds,
        {"10u": 0, "cp": 1, "hcc": 2, "tp": 3},
        {},
    )

    assert x_lres.shape == (2, 4)
    assert np.allclose(x_lres[:, 0], [1.0, 2.0])
    assert np.allclose(x_lres[:, 1], 0.0)
    assert np.allclose(x_lres[:, 2], 0.0)
    assert np.allclose(x_lres[:, 3], 0.0)
    assert x_hres.shape == (3, 0)


def test_load_inputs_from_bundle_numpy_interpolates_missing_pl_level(monkeypatch):
    ds = xr.Dataset(
        data_vars={
            "in_lres_q": (
                ("level", "point_lres"),
                np.array(
                    [
                        [1.0, 3.0],
                        [5.0, 7.0],
                    ],
                    dtype=np.float32,
                ),
            ),
        },
        coords={
            "level": np.array([100, 200], dtype=np.int32),
            "point_lres": np.arange(2, dtype=np.int32),
            "point_hres": np.arange(1, dtype=np.int32),
            "lat_lres": ("point_lres", np.array([10.0, 11.0], dtype=np.float32)),
            "lon_lres": ("point_lres", np.array([20.0, 21.0], dtype=np.float32)),
            "lat_hres": ("point_hres", np.array([30.0], dtype=np.float32)),
            "lon_hres": ("point_hres", np.array([40.0], dtype=np.float32)),
        },
        attrs={"case_valid_time": "2023-08-21T00:00:00"},
    )
    monkeypatch.setattr(bundle, "fill_hres_features", lambda *args, **kwargs: None)

    x_lres, *_ = bundle.load_inputs_from_bundle_numpy(
        ds,
        {"q_150": 0},
        {},
    )

    assert x_lres.shape == (2, 1)
    assert np.allclose(x_lres[:, 0], [3.0, 5.0])


def test_load_inputs_from_bundle_numpy_uses_lres_constant_for_single_level_z(monkeypatch):
    ds = xr.Dataset(
        data_vars={
            "in_lres_z": (
                ("level", "point_lres"),
                np.array(
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ],
                    dtype=np.float32,
                ),
            ),
        },
        coords={
            "level": np.array([100, 200], dtype=np.int32),
            "point_lres": np.arange(2, dtype=np.int32),
            "point_hres": np.arange(1, dtype=np.int32),
            "lat_lres": ("point_lres", np.array([10.0, 11.0], dtype=np.float32)),
            "lon_lres": ("point_lres", np.array([20.0, 21.0], dtype=np.float32)),
            "lat_hres": ("point_hres", np.array([30.0], dtype=np.float32)),
            "lon_hres": ("point_hres", np.array([40.0], dtype=np.float32)),
        },
        attrs={"case_valid_time": "2023-08-21T00:00:00"},
    )
    monkeypatch.setattr(bundle, "fill_hres_features", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bundle,
        "_load_constant_forcings_for_size",
        lambda *args, **kwargs: ({"z": np.array([9.0, 10.0], dtype=np.float32)}, "fake", []),
    )

    x_lres, *_ = bundle.load_inputs_from_bundle_numpy(
        ds,
        {"z": 0},
        {},
    )

    assert x_lres.shape == (2, 1)
    assert np.allclose(x_lres[:, 0], [9.0, 10.0])


def test_bundle_main_forwards_allow_missing_target_unsafe(monkeypatch, tmp_path, capsys):
    captured: dict[str, object] = {}

    def _fake_build_input_bundle_from_grib(**kwargs):
        captured.update(kwargs)
        return tmp_path / "bundle.nc"

    monkeypatch.setattr(bundle, "build_input_bundle_from_grib", _fake_build_input_bundle_from_grib)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bundle.py",
            "--lres-sfc-grib",
            "lres_sfc.grib",
            "--lres-pl-grib",
            "lres_pl.grib",
            "--hres-grib",
            "hres.grib",
            "--allow-missing-target-unsafe",
            "--out",
            str(tmp_path / "bundle.nc"),
        ],
    )

    bundle.main()

    assert captured["require_target_fields"] is False
    assert f"Saved bundle: {tmp_path / 'bundle.nc'}" in capsys.readouterr().out
