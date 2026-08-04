from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from eval.evaluators.local_global.runner import run


def _write_prediction(path: Path, lon: list[float], lat: list[float], values: np.ndarray) -> None:
    ds = xr.Dataset(
        {
            "y_pred": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                values,
            ),
            "y": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                values + 1.0,
            ),
            "lon_hres": (["grid_point_hres"], np.asarray(lon, dtype=np.float64)),
            "lat_hres": (["grid_point_hres"], np.asarray(lat, dtype=np.float64)),
            "lon_lres": (["grid_point_lres"], np.asarray(lon, dtype=np.float64)),
            "lat_lres": (["grid_point_lres"], np.asarray(lat, dtype=np.float64)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [1],
            "grid_point_hres": range(len(lon)),
            "grid_point_lres": range(len(lon)),
            "weather_state": ["msl", "10u", "10v"],
        },
    )
    ds.to_netcdf(path)
    ds.close()


def test_local_global_runner_compares_cropped_support(tmp_path: Path):
    global_dir = tmp_path / "global"
    local_dir = tmp_path / "local"
    out_dir = tmp_path / "out"
    global_dir.mkdir()
    local_dir.mkdir()

    global_values = np.array([[[[100000.0, 1.0, 2.0], [99000.0, 3.0, 4.0], [98000.0, 5.0, 12.0]]]])
    _write_prediction(
        global_dir / "predictions_20230829_step024.nc",
        [-86.0, -84.0, -10.0],
        [27.0, 29.0, 0.0],
        global_values,
    )
    _write_prediction(
        local_dir / "predictions_20230829_step024.nc",
        [-86.0, -84.0],
        [27.0, 29.0],
        global_values[:, :, :2, :],
    )

    run(
        local_dir,
        {},
        {"global_predictions_dir": str(global_dir)},
        output_dir=out_dir,
    )

    payload = json.loads((out_dir / "local_global_parity.json").read_text())
    assert payload["headline"]["files"] == 1
    assert payload["headline"]["y_pred_max_abs"] == 0.0
    assert payload["headline"]["tc_max_wind_ms_abs_diff_max"] == 0.0
