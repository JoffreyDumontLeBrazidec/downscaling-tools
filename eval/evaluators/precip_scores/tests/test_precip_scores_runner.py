"""End-to-end test of the precip_scores runner on synthetic predictions.

Uses the embedded-y truth path and the embedded-x_interp baseline path so no
GRIB files are needed; the GRIB-backed sources have their own unit tests.
"""
from __future__ import annotations

import json

import numpy as np
import xarray as xr

from eval.evaluators.precip_scores.runner import run


def _write_predictions(path, *, truth_m, pred_m, base_m):
    n = truth_m.shape[0]
    n_members = pred_m.shape[0]
    y = np.broadcast_to(truth_m[None, None, :, None],
                        (1, n_members, n, 1)).astype(np.float32)
    ds = xr.Dataset(
        data_vars={
            "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y),
            "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                       pred_m[None, :, :, None].astype(np.float32)),
            "x_interp": (("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                         base_m[None, :, :, None].astype(np.float32)),
            "lat_hres": (("grid_point_hres",), np.linspace(-10, 10, n)),
            "lon_hres": (("grid_point_hres",), np.linspace(0, 20, n)),
        },
        coords={
            "sample": [0],
            "ensemble_member": list(range(n_members)),
            "weather_state": ["tp"],
        },
        attrs={"member_ids": ",".join(str(i + 1) for i in range(n_members)),
               "checkpoint_id": "test-ckpt"},
    )
    ds.to_netcdf(path)


def test_scores_embedded_truth_and_baseline(tmp_path):
    rng = np.random.default_rng(11)
    n = 400
    truth_m = rng.gamma(0.5, 0.002, size=n)  # metres per 6h window
    pred_m = np.stack([truth_m + 0.001, truth_m - 0.001])   # +-1 mm bias
    base_m = np.stack([truth_m + 0.002, truth_m + 0.002])   # +2 mm bias
    for step in (6, 12):
        _write_predictions(tmp_path / f"predictions_20250926_step{step:03d}.nc",
                           truth_m=truth_m, pred_m=pred_m, base_m=base_m)

    out = run(tmp_path, lane_config={}, eval_config={}, overwrite=True)
    payload = json.loads((out / "scores.json").read_text())

    assert payload["meta"]["truth_source"] == "embedded-y"
    assert payload["meta"]["baseline_source"] == "x_interp"
    assert payload["meta"]["n_slices"] == 2
    assert payload["meta"]["checkpoint_id"] == "test-ckpt"

    # float32 storage limits agreement to ~1e-4 mm
    row = payload["rows"][0]
    m1, m2 = row["members"]
    assert abs(m1["model"]["rmse_mm"] - 1.0) < 1e-3
    assert abs(m1["model"]["bias_mm"] - 1.0) < 1e-3
    assert abs(m2["model"]["bias_mm"] + 1.0) < 1e-3
    assert abs(m1["baseline"]["bias_mm"] - 2.0) < 1e-3
    # members are +1mm and -1mm around truth -> ensemble mean ~= truth
    assert row["model_ens_mean"]["rmse_mm"] < 1e-2

    summary = payload["summary"]
    assert abs(summary["model_rmse_mm"] - 1.0) < 1e-3
    assert abs(summary["baseline_rmse_mm"] - 2.0) < 1e-3
    assert abs(summary["model_over_baseline_rmse_ratio"] - 0.5) < 1e-3

    assert (out / "plots" / "precip_scores.pdf").stat().st_size > 1024
    assert (out / "scores_rows.csv").exists()


def test_missing_truth_without_config_raises(tmp_path):
    n = 50
    nanfield = np.full(n, np.nan)
    pred_m = np.stack([np.full(n, 0.001)])
    _write_predictions(tmp_path / "predictions_20250926_step006.nc",
                       truth_m=nanfield, pred_m=pred_m, base_m=pred_m)
    import pytest
    with pytest.raises(RuntimeError, match="truth_grib_tpl"):
        run(tmp_path, lane_config={}, eval_config={}, overwrite=True)


def test_baseline_from_the_native_input_grid(tmp_path):
    """Rule 41: a file with a native x (lat_lres/lon_lres) gives the baseline from x at the nearest native point,
    not from x_interp; native_input: false reads x_interp as before."""
    rng = np.random.default_rng(3)
    n = 400
    truth_m = rng.gamma(0.5, 0.002, size=n)
    pred_m = np.stack([truth_m + 0.001, truth_m - 0.001])
    base_m = np.stack([truth_m + 0.002, truth_m + 0.002])
    path = tmp_path / "predictions_20250926_step006.nc"
    _write_predictions(path, truth_m=truth_m, pred_m=pred_m, base_m=base_m)
    ds = xr.load_dataset(path)
    lat_l = np.linspace(-10, 10, 40)
    ds["lat_lres"] = (("grid_point_lres",), lat_l)
    ds["lon_lres"] = (("grid_point_lres",), np.linspace(0, 20, 40))
    ds["x"] = (("sample", "ensemble_member", "grid_point_lres", "weather_state"),
               np.full((1, 2, 40, 1), 0.005, np.float32))   # 5 mm everywhere on the native grid
    ds.to_netcdf(path)
    # a toy line of points reads as a regional box: keep its edge strip so the means below hold
    out = run(tmp_path, lane_config={}, eval_config={"box_edge_deg": 0}, output_dir=tmp_path / "nat", overwrite=True)
    meta = json.loads((out / "scores.json").read_text())["meta"]
    assert meta["baseline_source"] == "x-native" and meta["input_source"] == "x (native input grid)"
    row = json.loads((out / "scores.json").read_text())["rows"][0]
    assert abs(row["members"][0]["baseline"]["bias_mm"] - (5.0 - truth_m.mean() * 1000)) < 1e-2
    old = run(tmp_path, lane_config={}, eval_config={"native_input": False, "box_edge_deg": 0}, output_dir=tmp_path / "old",
              overwrite=True)
    meta_old = json.loads((old / "scores.json").read_text())["meta"]
    assert meta_old["baseline_source"] == "x_interp" and "interpolated" in meta_old["input_source"]
