from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


MODULE_PATH = Path(__file__).resolve().parents[1] / "templates" / "predictions_dir_spectra.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_predictions_dir_spectra_module", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_spectra_artifacts_writes_full_and_residual_outputs(tmp_path: Path, monkeypatch):
    module = _load_module()

    predictions_dir = tmp_path / "predictions"
    out_dir = tmp_path / "spectra"
    predictions_dir.mkdir()
    out_dir.mkdir()

    checkpoint_path = tmp_path / "model.ckpt"
    inference_checkpoint_path = tmp_path / "inference-model.ckpt"
    checkpoint_path.write_text("base\n", encoding="utf-8")
    inference_checkpoint_path.write_text("inference\n", encoding="utf-8")

    ds = xr.Dataset(
        data_vars={
            "x": (
                ["sample", "ensemble_member", "grid_point_lres", "weather_state"],
                np.array([[[[1.0], [1.0]]]], dtype=np.float32),
            ),
            "y": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                np.array([[[[4.0], [4.0]]]], dtype=np.float32),
            ),
            "y_pred": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                np.array([[[[5.0], [5.0]]]], dtype=np.float32),
            ),
            "lon_hres": (["grid_point_hres"], np.array([0.0, 1.0], dtype=np.float32)),
            "lat_hres": (["grid_point_hres"], np.array([10.0, 11.0], dtype=np.float32)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [0],
            "grid_point_lres": [0, 1],
            "grid_point_hres": [0, 1],
            "weather_state": ["2t"],
        },
        attrs={"checkpoint_path": str(checkpoint_path)},
    )
    ds.to_netcdf(predictions_dir / "predictions_20230826_step024.nc")

    class _FakeInterface:
        def __init__(self):
            self.model = self

        def eval(self):
            return self

        def apply_interpolate_to_high_res(self, x_tensor):
            return x_tensor + 1.0

    monkeypatch.setattr("eval.checkpoint_interpolation.torch.load", lambda *args, **kwargs: _FakeInterface())
    monkeypatch.setattr(
        module,
        "resolve_spectra_method",
        lambda method: (lambda lat, lon, val, nside, lmax: np.asarray(val, dtype=np.float64), "fake"),
    )

    summary = module.build_spectra_artifacts(
        pred_dir=predictions_dir,
        out_dir=out_dir,
        run_label="demo",
        states=["2t"],
        nside=8,
        lmax=8,
        spectra_method="auto",
        member_aggregation="per-file-mean",
        show_individual_curves=False,
        score_wavenumber_min_exclusive=0.0,
    )

    state_summary = summary["weather_states"]["2t"]
    assert state_summary["relative_l2_mean_curve"] == pytest.approx(0.25)
    assert state_summary["scopes"]["residual"]["relative_l2_mean_curve"] == pytest.approx(0.5)
    assert Path(state_summary["pdf"]).exists()
    assert Path(state_summary["scopes"]["residual"]["pdf"]).exists()
    assert summary["checkpoint_path"] == str(checkpoint_path)


def test_build_spectra_artifacts_uses_predictions_x_interp_without_checkpoint(tmp_path: Path, monkeypatch):
    module = _load_module()

    predictions_dir = tmp_path / "predictions"
    out_dir = tmp_path / "spectra"
    predictions_dir.mkdir()
    out_dir.mkdir()

    ds = xr.Dataset(
        data_vars={
            "x": (
                ["sample", "ensemble_member", "grid_point_lres", "weather_state"],
                np.array([[[[1.0], [1.0]]]], dtype=np.float32),
            ),
            "x_interp": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                np.array([[[[2.0], [2.0]]]], dtype=np.float32),
            ),
            "y": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                np.array([[[[4.0], [4.0]]]], dtype=np.float32),
            ),
            "y_pred": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                np.array([[[[5.0], [5.0]]]], dtype=np.float32),
            ),
            "lon_hres": (["grid_point_hres"], np.array([0.0, 1.0], dtype=np.float32)),
            "lat_hres": (["grid_point_hres"], np.array([10.0, 11.0], dtype=np.float32)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [0],
            "grid_point_lres": [0, 1],
            "grid_point_hres": [0, 1],
            "weather_state": ["2t"],
        },
    )
    ds.to_netcdf(predictions_dir / "predictions_20230826_step024.nc")

    monkeypatch.setattr(
        module.CheckpointResidualInterpolator,
        "_load_interface",
        lambda self: (_ for _ in ()).throw(AssertionError("checkpoint interpolation should not be used when x_interp is present")),
    )
    monkeypatch.setattr(
        module,
        "resolve_spectra_method",
        lambda method: (lambda lat, lon, val, nside, lmax: np.asarray(val, dtype=np.float64), "fake"),
    )

    summary = module.build_spectra_artifacts(
        pred_dir=predictions_dir,
        out_dir=out_dir,
        run_label="demo",
        states=["2t"],
        nside=8,
        lmax=8,
        spectra_method="auto",
        member_aggregation="per-file-mean",
        show_individual_curves=False,
        score_wavenumber_min_exclusive=0.0,
    )

    state_summary = summary["weather_states"]["2t"]
    assert summary["checkpoint_path"] is None
    assert summary["residualization"]["method"] == "predictions_x_interp"
    assert state_summary["relative_l2_mean_curve"] == pytest.approx(0.25)
    assert state_summary["scopes"]["residual"]["relative_l2_mean_curve"] == pytest.approx(0.5)
