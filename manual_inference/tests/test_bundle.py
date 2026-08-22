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


def _make_tp_only_sfc_dataset(point_count: int, *, value: float) -> xr.Dataset:
    coords = {"values": np.arange(point_count, dtype=np.int32)}
    return xr.Dataset(
        data_vars={
            "tp": ("values", np.full(point_count, value, dtype=np.float32)),
            "latitude": ("values", np.linspace(10, 20, point_count, dtype=np.float32)),
            "longitude": ("values", np.linspace(30, 40, point_count, dtype=np.float32)),
            "valid_time": ((), np.datetime64("2023-08-21T00:00:00")),
        },
        coords=coords,
    )


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
    original_open_dataset = xr.open_dataset

    def _open_dataset(path, engine=None, backend_kwargs=None, **kwargs):
        if engine == "cfgrib":
            return datasets[str(path)]
        return original_open_dataset(path, engine=engine, backend_kwargs=backend_kwargs, **kwargs)

    monkeypatch.setattr(xr, "open_dataset", _open_dataset)


def test_open_cfgrib_dataset_forwards_filter_by_keys(monkeypatch):
    captured = {}
    original_open_dataset = xr.open_dataset

    def _open_dataset(path, engine=None, backend_kwargs=None, **kwargs):
        captured["path"] = str(path)
        captured["engine"] = engine
        captured["backend_kwargs"] = backend_kwargs
        return xr.Dataset() if engine == "cfgrib" else original_open_dataset(
            path, engine=engine, backend_kwargs=backend_kwargs, **kwargs
        )

    monkeypatch.setattr(xr, "open_dataset", _open_dataset)

    bundle._open_cfgrib_dataset(
        "/tmp/mixed_input.grib",
        filter_by_keys={"typeOfLevel": "surface"},
    )

    assert captured["path"] == "/tmp/mixed_input.grib"
    assert captured["engine"] == "cfgrib"
    assert captured["backend_kwargs"]["filter_by_keys"] == {"typeOfLevel": "surface"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", None),
        ("10u,10v,2t,msl", ["10u", "10v", "2t", "msl"]),
        ("NONE", []),
        ("-", []),
        ("[]", []),
        ("empty", []),
    ],
)
def test_parse_channel_subset_csv_supports_explicit_empty_override(raw, expected):
    assert bundle._parse_channel_subset_csv(raw) == expected


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


def test_select_member_accepts_requested_member_one_for_singleton_zero_coord():
    ds = xr.Dataset(
        data_vars={"u10": (("number", "values"), np.array([[1.0, 2.0]], dtype=np.float32))},
        coords={
            "number": np.array([0], dtype=np.int32),
            "values": np.arange(2, dtype=np.int32),
        },
    )

    selected = bundle._select_member(ds, 1)

    assert "number" in selected.coords
    assert selected.sizes["values"] == 2
    assert np.asarray(selected["u10"].values).shape == (2,)


def test_select_member_accepts_matching_scalar_number_coord():
    ds = xr.Dataset(
        data_vars={"z": ("values", np.array([7.0, 8.0], dtype=np.float32))},
        coords={
            "values": np.arange(2, dtype=np.int32),
            "number": np.array(1, dtype=np.int32),
        },
    )

    selected = bundle._select_member(ds, 1)

    assert selected.identical(ds)


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


def test_build_input_bundle_uses_explicit_hres_static_override(tmp_path, monkeypatch):
    work_dir = tmp_path / "gribs"
    work_dir.mkdir()
    hres_path = work_dir / "enfo_o96_0001_date20250926_time0000_step24to120_sfc.grib"
    hres_static_path = work_dir / "iekm_o96_static_date20250926_time0000_sfc.grib"
    sfc_y_path = work_dir / "iekm_o96_iekm_date20250926_time0000_step24to120_sfc_y.grib"
    pl_y_path = work_dir / "iekm_o96_iekm_date20250926_time0000_step24to120_pl_y.grib"
    hres_path.touch()
    hres_static_path.touch()
    sfc_y_path.touch()
    pl_y_path.touch()

    datasets = {
        str(work_dir / "lres_sfc.grib"): _make_sfc_dataset(3, prefix=0.0),
        str(work_dir / "lres_pl.grib"): _make_pl_dataset(3),
        str(hres_path): _make_hres_dataset(4),
        str(hres_static_path): xr.Dataset(
            data_vars={
                "z": (("step", "values"), np.full((1, 4), 70.0, dtype=np.float32)),
                "lsm": (("step", "values"), np.full((1, 4), 80.0, dtype=np.float32)),
                "latitude": (("step", "values"), np.linspace(50, 60, 4, dtype=np.float32)[None, :]),
                "longitude": (("step", "values"), np.linspace(70, 80, 4, dtype=np.float32)[None, :]),
            },
            coords={
                "step": np.array([np.timedelta64(24, "h")]),
                "values": np.arange(4, dtype=np.int32),
            },
        ),
        str(sfc_y_path): _make_sfc_dataset(4, prefix=100.0),
        str(pl_y_path): _make_pl_dataset(4),
    }
    _install_fake_earthkit(monkeypatch, datasets)

    out_path = tmp_path / "bundle.nc"
    bundle.build_input_bundle_from_grib(
        lres_sfc_grib=work_dir / "lres_sfc.grib",
        lres_pl_grib=work_dir / "lres_pl.grib",
        hres_grib=hres_path,
        hres_static_grib=hres_static_path,
        out_nc=out_path,
        step_hours=48,
        target_sfc_grib=sfc_y_path,
        target_pl_grib=pl_y_path,
    )

    ds = xr.open_dataset(out_path)
    try:
        assert ds.attrs["source_hres"] == str(hres_static_path)
        assert ds.attrs["source_hres_static_override"] == str(hres_static_path)
        assert np.allclose(ds["in_hres_z"].values, 70.0)
        assert np.allclose(ds["in_hres_lsm"].values, 80.0)
        assert ds.attrs["source_target_sfc"] == str(sfc_y_path)
        assert ds.attrs["source_target_pl"] == str(pl_y_path)
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


def test_build_input_bundle_merges_surface_tp_sidecars(tmp_path, monkeypatch):
    datasets = {
        "lres_sfc.grib": _make_sfc_dataset(3, prefix=0.0),
        "lres_sfc_tp.grib": _make_tp_only_sfc_dataset(3, value=9.0),
        "lres_pl.grib": _make_pl_dataset(3),
        "hres.grib": _make_hres_dataset(4),
        "target_sfc.grib": _make_sfc_dataset(4, prefix=100.0),
        "target_sfc_tp.grib": _make_tp_only_sfc_dataset(4, value=109.0),
        "target_pl.grib": _make_pl_dataset(4),
    }
    _install_fake_earthkit(monkeypatch, datasets)

    out_path = tmp_path / "bundle.nc"
    bundle.build_input_bundle_from_grib(
        lres_sfc_grib="lres_sfc.grib",
        lres_sfc_extra_gribs=["lres_sfc_tp.grib"],
        lres_pl_grib="lres_pl.grib",
        hres_grib="hres.grib",
        out_nc=out_path,
        target_sfc_grib="target_sfc.grib",
        target_sfc_extra_gribs=["target_sfc_tp.grib"],
        target_pl_grib="target_pl.grib",
    )

    ds = xr.open_dataset(out_path)
    try:
        assert "in_lres_tp" in ds
        assert "target_hres_tp" in ds
        assert np.allclose(ds["in_lres_tp"].values, 9.0)
        assert np.allclose(ds["target_hres_tp"].values, 109.0)
        assert ds.attrs["source_lres_sfc_extra"] == "lres_sfc_tp.grib"
        assert ds.attrs["source_target_sfc_extra"] == "target_sfc_tp.grib"
    finally:
        ds.close()


def test_build_input_bundle_supports_surface_only_channel_subset_without_sp(tmp_path, monkeypatch):
    work_dir = tmp_path / "gribs"
    work_dir.mkdir()
    lres_path = work_dir / "destine_input.grib"
    hres_path = work_dir / "destine_forcing_sfc.grib"
    target_path = work_dir / "destine_target_y.grib"
    lres_path.touch()
    hres_path.touch()
    target_path.touch()

    lres_surface = _make_sfc_dataset(3, prefix=0.0).drop_vars("sp")
    target_surface = _make_sfc_dataset(4, prefix=100.0).drop_vars(["sp", "d2m", "skt", "tcw"])
    datasets = {
        str(lres_path): lres_surface,
        str(hres_path): _make_hres_dataset(4),
        str(target_path): target_surface,
    }
    _install_fake_earthkit(monkeypatch, datasets)

    out_path = tmp_path / "bundle.nc"
    bundle.build_input_bundle_from_grib(
        lres_sfc_grib=lres_path,
        lres_pl_grib=lres_path,
        hres_grib=hres_path,
        out_nc=out_path,
        step_hours=24,
        member=1,
        target_sfc_grib=target_path,
        target_pl_grib=None,
        lres_sfc_channels=["10u", "10v", "2t", "msl"],
        lres_pl_channels=[],
        target_sfc_channels=["10u", "10v", "2t", "msl"],
        target_pl_channels=[],
    )

    ds = xr.open_dataset(out_path)
    try:
        assert ds.attrs["selected_lres_sfc_channels"] == "10u,10v,2t,msl"
        assert ds.attrs["selected_lres_pl_channels"] == ""
        assert ds.attrs["selected_target_sfc_channels"] == "10u,10v,2t,msl"
        assert ds.attrs["selected_target_pl_channels"] == ""
        assert "in_lres_10u" in ds
        assert "in_lres_10v" in ds
        assert "in_lres_2t" in ds
        assert "in_lres_msl" in ds
        assert "in_lres_sp" not in ds
        assert "in_lres_q" not in ds
        assert "target_hres_10u" in ds
        assert "target_hres_10v" in ds
        assert "target_hres_2t" in ds
        assert "target_hres_msl" in ds
        assert "target_hres_sp" not in ds
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


def test_load_inputs_from_bundle_numpy_reads_explicit_hres_input(monkeypatch):
    ds = xr.Dataset(
        data_vars={
            "in_lres_10u": ("point_lres", np.array([1.0, 2.0], dtype=np.float32)),
            "in_hres_2t": ("point_hres", np.array([7.0, 8.0, 9.0], dtype=np.float32)),
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
        {"10u": 0},
        {"2t": 0},
        constant_forcings_npz=None,
    )

    assert np.allclose(x_lres[:, 0], [1.0, 2.0])
    assert np.allclose(x_hres[:, 0], [7.0, 8.0, 9.0])


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


def test_bundle_main_forwards_surface_extra_gribs(monkeypatch, tmp_path):
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
            "--lres-sfc-extra-grib",
            "lres_sfc_tp.grib",
            "--lres-pl-grib",
            "lres_pl.grib",
            "--hres-grib",
            "hres.grib",
            "--target-sfc-grib",
            "target_sfc.grib",
            "--target-sfc-extra-grib",
            "target_sfc_tp.grib",
            "--out",
            str(tmp_path / "bundle.nc"),
        ],
    )

    bundle.main()

    assert captured["lres_sfc_extra_gribs"] == ["lres_sfc_tp.grib"]
    assert captured["target_sfc_extra_gribs"] == ["target_sfc_tp.grib"]


def test_bundle_main_forwards_channel_subset_overrides(monkeypatch, tmp_path):
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
            "--target-sfc-grib",
            "target_sfc.grib",
            "--lres-sfc-channels",
            "10u,10v,2t,msl",
            "--lres-pl-channels",
            "",
            "--target-sfc-channels",
            "10u,10v,2t,msl",
            "--target-pl-channels",
            "",
            "--out",
            str(tmp_path / "bundle.nc"),
        ],
    )

    bundle.main()

    assert captured["lres_sfc_channels"] == ["10u", "10v", "2t", "msl"]
    assert captured["lres_pl_channels"] is None
    assert captured["target_sfc_channels"] == ["10u", "10v", "2t", "msl"]
    assert captured["target_pl_channels"] is None


# --- de-accumulation of accumulated low-res surface inputs -------------------
# Regression cover for the defect that invalidated the o1280->o2560 Humberto
# campaign: ssrd/strd arrive accumulated since forecast start, and feeding the
# raw running total put the model about 15 sigma out of distribution from the
# second lead time onward. The correction now decides per field from the data,
# so these tests pin both halves of that decision.


def test_deaccumulation_defaults_to_automatic_detection(monkeypatch):
    monkeypatch.delenv("MI_DEACCUMULATE_LRES", raising=False)
    assert bundle.deaccumulate_mode_from_env() == ("auto", None)


@pytest.mark.parametrize(
    "value, expected_mode",
    [("auto", "auto"), ("off", "off"), ("none", "off"), ("ssrd,strd", "forced")],
)
def test_deaccumulation_escape_hatch_is_honoured(monkeypatch, value, expected_mode):
    monkeypatch.setenv("MI_DEACCUMULATE_LRES", value)
    mode, names = bundle.deaccumulate_mode_from_env()
    assert mode == expected_mode
    if expected_mode == "forced":
        assert names == ("ssrd", "strd")


def test_running_total_is_recognised_as_accumulated():
    previous = np.linspace(0.0, 100.0, 500)
    current = previous + np.linspace(1.0, 9.0, 500)  # every point grew
    accumulated, fraction = bundle.looks_accumulated(current, previous)
    assert accumulated
    assert fraction == pytest.approx(1.0)


def test_field_constant_in_time_is_not_zeroed_out():
    """A constant field is non-decreasing everywhere, so monotonicity alone
    would wrongly mark it accumulated and subtracting would zero a real input."""
    constant = np.full(500, 3.75)
    accumulated, _ = bundle.looks_accumulated(constant, constant)
    assert not accumulated


def test_per_step_field_is_left_alone():
    rng = np.random.default_rng(0)
    previous = rng.normal(size=5000)
    current = rng.normal(size=5000)
    accumulated, fraction = bundle.looks_accumulated(current, previous)
    assert not accumulated
    assert 0.3 < fraction < 0.7


def test_mismatched_or_non_finite_input_is_never_deaccumulated():
    assert bundle.looks_accumulated(np.zeros(10), np.zeros(11))[0] is False
    with_nan = np.arange(10, dtype=float)
    with_nan[3] = np.nan
    assert bundle.looks_accumulated(with_nan, np.zeros(10))[0] is False


def test_first_step_has_no_previous_bundle(tmp_path):
    """At the first step the accumulation window already equals one increment,
    so nothing may be subtracted."""
    first = tmp_path / "case_step006h_input_bundle.nc"
    first.write_bytes(b"")
    assert bundle.previous_step_bundle_path(first) is None


def test_later_step_resolves_its_predecessor(tmp_path):
    earlier = tmp_path / "case_step018h_input_bundle.nc"
    later = tmp_path / "case_step024h_input_bundle.nc"
    earlier.write_bytes(b"")
    later.write_bytes(b"")
    assert bundle.previous_step_bundle_path(later) == earlier


def test_predecessor_is_found_at_a_non_six_hour_cadence(tmp_path):
    """The o320->o1280 regional bundles are staged 24-hourly. Assuming six hours
    would find no file and silently skip the correction."""
    for step in (24, 48, 72):
        (tmp_path / ("case_step%03dh_input_bundle.nc" % step)).write_bytes(b"")
    later = tmp_path / "case_step072h_input_bundle.nc"
    assert bundle.previous_step_bundle_path(later).name == "case_step048h_input_bundle.nc"


def test_earliest_step_has_no_predecessor_at_any_cadence(tmp_path):
    for step in (24, 48):
        (tmp_path / ("case_step%03dh_input_bundle.nc" % step)).write_bytes(b"")
    first = tmp_path / "case_step024h_input_bundle.nc"
    assert bundle.previous_step_bundle_path(first) is None


def test_nearest_earlier_step_wins_when_the_cadence_is_irregular(tmp_path):
    for step in (6, 12, 36):
        (tmp_path / ("case_step%03dh_input_bundle.nc" % step)).write_bytes(b"")
    later = tmp_path / "case_step036h_input_bundle.nc"
    assert bundle.previous_step_bundle_path(later).name == "case_step012h_input_bundle.nc"


def test_forced_cadence_overrides_detection(tmp_path, monkeypatch):
    for step in (6, 12, 18, 24):
        (tmp_path / ("case_step%03dh_input_bundle.nc" % step)).write_bytes(b"")
    later = tmp_path / "case_step024h_input_bundle.nc"
    monkeypatch.setenv("MI_DEACCUM_STEP_HOURS", "12")
    assert bundle.previous_step_bundle_path(later).name == "case_step012h_input_bundle.nc"
    monkeypatch.delenv("MI_DEACCUM_STEP_HOURS")
    assert bundle.previous_step_bundle_path(later).name == "case_step018h_input_bundle.nc"
