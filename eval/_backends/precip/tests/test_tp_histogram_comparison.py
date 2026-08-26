from __future__ import annotations

import numpy as np
import xarray as xr

from eval._backends.precip.tp_histogram_comparison import (
    StreamingDist,
    accumulate_tp_by_step,
)


def _write_nc(path, *, y, y_pred, x_interp):
    ds = xr.Dataset(
        data_vars={
            "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y),
            "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y_pred),
            "x_interp": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), x_interp),
            "lat_hres": (("grid_point_hres",), np.linspace(-10, 10, y.shape[2])),
            "lon_hres": (("grid_point_hres",), np.linspace(0, 20, y.shape[2])),
        },
        coords={
            "sample": [0],
            "ensemble_member": list(range(y.shape[1])),
            "weather_state": ["tp"],
        },
    )
    ds.to_netcdf(path)


def test_accumulate_uses_selected_member_and_mm_units(tmp_path):
    shape = (1, 2, 3, 1)
    y = np.array([[[[0.001], [0.002], [0.003]],
                   [[0.091], [0.092], [0.093]]]], dtype=np.float32)
    y_pred = y + 0.001
    x_interp = y + 0.002
    _write_nc(tmp_path / "predictions_20250101_step024.nc",
              y=y, y_pred=y_pred, x_interp=x_interp)

    step_data = accumulate_tp_by_step(tmp_path, ensemble_member_index=0)
    bucket = step_data[24]
    # member 0 only: 3 points per series, in mm
    assert bucket["truth"].n == 3
    assert bucket["pred"].n == 3
    assert bucket["input"].n == 3
    assert np.isclose(bucket["truth"].max, 3.0, atol=0.01)
    assert np.isclose(bucket["pred"].max, 4.0, atol=0.01)
    assert np.isclose(bucket["input"].max, 5.0, atol=0.01)


def test_degenerate_x_interp_is_not_used_as_input(tmp_path):
    y = np.full((1, 1, 4, 1), 0.001, dtype=np.float32)
    _write_nc(tmp_path / "predictions_20250101_step006.nc",
              y=y, y_pred=y, x_interp=np.zeros_like(y))
    step_data = accumulate_tp_by_step(tmp_path)
    assert step_data[6]["input"].empty
    assert not step_data[6]["pred"].empty


def test_streaming_dist_quantiles_and_density():
    rng = np.random.default_rng(3)
    vals = rng.gamma(0.6, 3.0, size=100_000)
    d = StreamingDist()
    d.update(vals[:50_000])
    d.update(vals[50_000:])
    assert d.n == vals.size
    q = d.quantile(99.0)
    exact = float(np.percentile(vals, 99.0))
    assert abs(q - exact) <= max(0.05 * exact, 0.05)
    widths = np.diff(d.EDGES)
    assert np.isclose(float((d.density() * widths).sum()), 1.0, atol=1e-9)


def test_streaming_dist_counts_negatives_in_zero_bin():
    d = StreamingDist()
    d.update(np.array([-1.0, -0.5, 0.5, 2.0]))
    assert d.neg == 2
    assert d.n == 4
    assert d.counts.sum() == 4
