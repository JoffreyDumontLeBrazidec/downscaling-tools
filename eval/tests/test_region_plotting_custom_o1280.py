from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from eval.region_plotting import plot_regions as mod


def test_sample_meta_title_includes_indices_and_dates():
    ds = xr.Dataset(
        data_vars={
            "date": ("sample", np.array(["2026-02-27T00:00:00"], dtype="datetime64[ns]")),
            "init_date": ("sample", np.array(["2026-02-26T00:00:00"], dtype="datetime64[ns]")),
            "lead_step_hours": ("sample", np.array([24], dtype=np.int32)),
        },
        coords={"sample": [0]},
    )
    title = mod._sample_meta_title(ds, "tibet_karakoram", 0)
    assert "tibet_karakoram" in title
    assert "sample_pos=0" in title
    assert "sample_id=0" in title
    assert "date=2026-02-27 00:00" in title
    assert "init=2026-02-26 00:00" in title
    assert "lead_h=24" in title


def test_run_region_plots_uses_custom_path_for_o1280(tmp_path: Path, monkeypatch):
    run_parent = tmp_path / "eval"
    run_name = "myrun"
    run_dir = run_parent / run_name
    run_dir.mkdir(parents=True)
    pred_path = run_dir / "predictions.nc"

    ds = xr.Dataset(
        data_vars={
            "x_0": (("sample", "grid_point_lres", "weather_state"), np.zeros((1, 1, 1), dtype=np.float32)),
            "y_0": (("sample", "grid_point_hres", "weather_state"), np.zeros((1, 1, 1), dtype=np.float32)),
            "y_pred_0": (("sample", "grid_point_hres", "weather_state"), np.zeros((1, 1, 1), dtype=np.float32)),
            "lon_hres": (("grid_point_hres",), np.array([0.0], dtype=np.float32)),
            "lat_hres": (("grid_point_hres",), np.array([0.0], dtype=np.float32)),
            "lon_lres": (("grid_point_lres",), np.array([0.0], dtype=np.float32)),
            "lat_lres": (("grid_point_lres",), np.array([0.0], dtype=np.float32)),
        },
        coords={
            "sample": [0],
            "grid_point_hres": [0],
            "grid_point_lres": [0],
            "weather_state": ["10u"],
        },
        attrs={"grid": "O1280"},
    )
    ds.to_netcdf(pred_path)

    called = {"custom": False, "default": False}

    def _fake_custom(**kwargs):
        called["custom"] = True

    class _FakeLIP:
        def __init__(self, *args, **kwargs):
            called["default"] = True
            self.regions = []

        def save_plot(self, *args, **kwargs):
            called["default"] = True

    monkeypatch.setattr(mod, "_save_custom_o1280_plots", _fake_custom)
    monkeypatch.setattr(mod, "LocalInferencePlotter", _FakeLIP)

    mod.run_region_plots_from_predictions(
        run_parent_dir=run_parent,
        run_name=run_name,
        predictions_filename="predictions.nc",
    )

    assert called["custom"] is True
    assert called["default"] is False


def test_run_region_plots_accepts_slim_prediction_variables(tmp_path: Path, monkeypatch):
    run_parent = tmp_path / "eval"
    run_name = "slimrun"
    run_dir = run_parent / run_name
    run_dir.mkdir(parents=True)
    pred_path = run_dir / "predictions.nc"

    ds = xr.Dataset(
        data_vars={
            "x": (("sample", "grid_point_lres", "weather_state"), np.zeros((1, 1, 1), dtype=np.float32)),
            "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), np.zeros((1, 1, 1, 1), dtype=np.float32)),
            "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), np.zeros((1, 1, 1, 1), dtype=np.float32)),
            "lon_hres": (("grid_point_hres",), np.array([0.0], dtype=np.float32)),
            "lat_hres": (("grid_point_hres",), np.array([0.0], dtype=np.float32)),
            "lon_lres": (("grid_point_lres",), np.array([0.0], dtype=np.float32)),
            "lat_lres": (("grid_point_lres",), np.array([0.0], dtype=np.float32)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [0],
            "grid_point_hres": [0],
            "grid_point_lres": [0],
            "weather_state": ["10u"],
        },
        attrs={"grid": "O320"},
    )
    ds.to_netcdf(pred_path)

    captured: dict[str, list[str]] = {}

    class _FakeLIP:
        def __init__(self, *args, **kwargs):
            self.regions = ["amazon_forest"]

        def save_plot(self, list_regions, list_model_variables, weather_states, **kwargs):
            captured["regions"] = list_regions
            captured["model_variables"] = list_model_variables
            captured["weather_states"] = weather_states

    monkeypatch.setattr(mod, "LocalInferencePlotter", _FakeLIP)

    mod.run_region_plots_from_predictions(
        run_parent_dir=run_parent,
        run_name=run_name,
        predictions_filename="predictions.nc",
    )

    assert captured["regions"] == ["amazon_forest"]
    assert captured["model_variables"] == ["x", "y", "y_pred"]
    assert captured["weather_states"] == ["10u"]
