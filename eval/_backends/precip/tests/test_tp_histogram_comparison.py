from __future__ import annotations

import numpy as np
import xarray as xr

from eval._backends.precip.tp_histogram_comparison import load_tp_by_step


def test_load_tp_by_step_uses_selected_ensemble_member(tmp_path):
    path = tmp_path / "predictions_20250101_step024.nc"
    ds = xr.Dataset(
        data_vars={
            "y": (
                ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                np.array([[[[1.0], [2.0], [3.0]], [[91.0], [92.0], [93.0]]]], dtype=np.float32),
            ),
            "y_pred": (
                ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                np.array([[[[4.0], [5.0], [6.0]], [[94.0], [95.0], [96.0]]]], dtype=np.float32),
            ),
            "x_interp": (
                ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                np.array([[[[7.0], [8.0], [9.0]], [[97.0], [98.0], [99.0]]]], dtype=np.float32),
            ),
        },
        coords={
            "sample": [0],
            "ensemble_member": [0, 1],
            "grid_point_hres": [0, 1, 2],
            "weather_state": ["tp"],
        },
    )
    ds.to_netcdf(path)

    step_data = load_tp_by_step(tmp_path, ensemble_member_index=0)

    np.testing.assert_array_equal(step_data[24]["truth"], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(step_data[24]["pred"], np.array([4.0, 5.0, 6.0], dtype=np.float32))
    np.testing.assert_array_equal(step_data[24]["input"], np.array([7.0, 8.0, 9.0], dtype=np.float32))
