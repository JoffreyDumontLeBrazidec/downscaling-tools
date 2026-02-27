import tempfile
from datetime import datetime

import numpy as np
import torch
import xarray as xr

from manual_inference.input_data_construction import bundle as bundle_mod


def _make_bundle_dataset(point_lres=3, point_hres=4, levels=(850,)):
    coords = {
        "point_lres": np.arange(point_lres, dtype=np.int32),
        "point_hres": np.arange(point_hres, dtype=np.int32),
        "level": np.array(levels, dtype=np.int32),
        "lat_hres": ("point_hres", np.linspace(10, 13, point_hres, dtype=np.float32)),
        "lon_hres": ("point_hres", np.linspace(20, 23, point_hres, dtype=np.float32)),
    }

    data_vars = {
        "in_lres_10u": ("point_lres", np.linspace(0, 2, point_lres, dtype=np.float32)),
        "in_lres_t": (("level", "point_lres"), np.ones((len(levels), point_lres), dtype=np.float32)),
        "in_hres_z": ("point_hres", np.linspace(1, 4, point_hres, dtype=np.float32)),
        "in_hres_lsm": ("point_hres", np.linspace(5, 8, point_hres, dtype=np.float32)),
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["case_valid_time"] = "2024-01-01T00:00:00"
    return ds


def test_split_level_channel():
    assert bundle_mod.split_level_channel("t_850") == ("t", 850)
    assert bundle_mod.split_level_channel("10u") == ("10u", None)


def test_parse_valid_time():
    dt = bundle_mod.parse_valid_time("2024-02-03T01:02:03", None)
    assert dt == datetime(2024, 2, 3, 1, 2, 3)

    try:
        bundle_mod.parse_valid_time(None, "unknown")
    except ValueError as exc:
        assert "No valid time available" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown valid_time")


def test_fill_inputs_from_bundle(monkeypatch):
    ds = _make_bundle_dataset()
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        ds.to_netcdf(tmp.name, engine="scipy")

        called = {}

        def _fake_fill_hres(x_hres, name_to_idx_hres, lat_hres, lon_hres, dt, device, **kwargs):
            called["dt"] = dt
            called["lat_hres"] = np.asarray(lat_hres)
            called["lon_hres"] = np.asarray(lon_hres)
            called["z"] = kwargs.get("z")
            called["lsm"] = kwargs.get("lsm")

        monkeypatch.setattr(bundle_mod, "fill_hres_features", _fake_fill_hres)

        x_lres = torch.zeros((1, 1, 1, 3, 2), dtype=torch.float32)
        x_hres = torch.zeros((1, 1, 1, 4, 2), dtype=torch.float32)
        name_to_idx_lres = {"10u": 0, "t_850": 1}
        name_to_idx_hres = {"z": 0, "lsm": 1}

        bundle_mod.fill_inputs_from_bundle(
            tmp.name,
            x_lres,
            x_hres,
            name_to_idx_lres,
            name_to_idx_hres,
            device="cpu",
        )

        assert np.allclose(x_lres[0, 0, 0, :, 0].numpy(), np.linspace(0, 2, 3))
        assert np.allclose(x_lres[0, 0, 0, :, 1].numpy(), np.ones(3))
        assert called["dt"] == datetime(2024, 1, 1, 0, 0)
        assert called["z"].shape[0] == 4
        assert called["lsm"].shape[0] == 4
        assert called["lat_hres"].shape[0] == 4
        assert called["lon_hres"].shape[0] == 4


def test_fill_inputs_from_bundle_zarr(monkeypatch):
    zarr = None
    try:
        import zarr as _zarr  # noqa: F401

        zarr = _zarr
    except Exception:
        pass
    if zarr is None:
        return

    ds = _make_bundle_dataset()
    with tempfile.TemporaryDirectory(suffix=".zarr") as tmpdir:
        ds.to_zarr(tmpdir, mode="w", consolidated=True)

        monkeypatch.setattr(bundle_mod, "fill_hres_features", lambda *args, **kwargs: None)

        x_lres = torch.zeros((1, 1, 1, 3, 1), dtype=torch.float32)
        x_hres = torch.zeros((1, 1, 1, 4, 1), dtype=torch.float32)
        name_to_idx_lres = {"10u": 0}
        name_to_idx_hres = {"z": 0}

        bundle_mod.fill_inputs_from_bundle(
            tmpdir,
            x_lres,
            x_hres,
            name_to_idx_lres,
            name_to_idx_hres,
            device="cpu",
        )

        assert np.allclose(x_lres[0, 0, 0, :, 0].numpy(), np.linspace(0, 2, 3))


def test_fill_inputs_from_bundle_grid_mismatch(monkeypatch):
    ds = _make_bundle_dataset(point_lres=2)
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        ds.to_netcdf(tmp.name, engine="scipy")

        monkeypatch.setattr(bundle_mod, "fill_hres_features", lambda *args, **kwargs: None)

        x_lres = torch.zeros((1, 1, 1, 3, 2), dtype=torch.float32)
        x_hres = torch.zeros((1, 1, 1, 4, 2), dtype=torch.float32)
        name_to_idx_lres = {"10u": 0}
        name_to_idx_hres = {"z": 0}

        try:
            bundle_mod.fill_inputs_from_bundle(
                tmp.name,
                x_lres,
                x_hres,
                name_to_idx_lres,
                name_to_idx_hres,
                device="cpu",
            )
        except RuntimeError as exc:
            assert "LRES grid-size mismatch" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError for grid mismatch")
