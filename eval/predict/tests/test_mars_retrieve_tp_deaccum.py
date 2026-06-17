"""Tests for _deaccumulate_tp_xarray in eval/predict/mars_retrieve.py."""
from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import pytest

from eval.predict.mars_retrieve import _deaccumulate_tp_xarray


def _make_ds(steps_h=(6, 12, 18, 24), members=(1, 2), gridpoints=4):
    n_step = len(steps_h)
    n_mem = len(members)
    tp = np.zeros((n_mem, n_step, gridpoints), dtype=np.float64)
    for i, s in enumerate(steps_h):
        tp[:, i, :] = s * 1e-4
    t2m = np.full((n_mem, n_step, gridpoints), 280.0, dtype=np.float64)
    return xr.Dataset(
        {
            "tp": (("number", "step", "values"), tp),
            "2t": (("number", "step", "values"), t2m),
        },
        coords={
            "number": np.array(members, dtype=np.int64),
            "step": pd.to_timedelta(np.array(steps_h, dtype="int64"), unit="h"),
        },
    )


def test_step6_unchanged():
    ds = _make_ds()
    out = _deaccumulate_tp_xarray(ds, window_hours=6)
    s0 = out.sel(step=pd.Timedelta(hours=6))["tp"].values
    expected = ds.sel(step=pd.Timedelta(hours=6))["tp"].values
    np.testing.assert_array_equal(s0, expected)


def test_step12_is_difference():
    ds = _make_ds()
    out = _deaccumulate_tp_xarray(ds, window_hours=6)
    s12_out = out.sel(step=pd.Timedelta(hours=12))["tp"].values
    expected = (
        ds.sel(step=pd.Timedelta(hours=12))["tp"].values
        - ds.sel(step=pd.Timedelta(hours=6))["tp"].values
    )
    np.testing.assert_allclose(s12_out, expected)


def test_other_vars_untouched():
    ds = _make_ds()
    out = _deaccumulate_tp_xarray(ds, window_hours=6)
    np.testing.assert_array_equal(out["2t"].values, ds["2t"].values)


def test_unsorted_steps_handled():
    ds = _make_ds(steps_h=(24, 6, 18, 12))
    out = _deaccumulate_tp_xarray(ds, window_hours=6)
    s12 = out.sel(step=pd.Timedelta(hours=12))["tp"].values
    expected = (
        ds.sel(step=pd.Timedelta(hours=12))["tp"].values
        - ds.sel(step=pd.Timedelta(hours=6))["tp"].values
    )
    np.testing.assert_allclose(s12, expected)


def test_no_tp_returns_unchanged():
    ds = xr.Dataset({"2t": (("step",), np.array([1.0, 2.0]))},
                    coords={"step": pd.to_timedelta([6, 12], unit="h")})
    out = _deaccumulate_tp_xarray(ds, window_hours=6)
    np.testing.assert_array_equal(out["2t"].values, ds["2t"].values)
