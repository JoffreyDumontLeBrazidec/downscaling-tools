"""Unit tests for precip truth/baseline sources (no real GRIBs needed)."""
from __future__ import annotations

import numpy as np
import pytest

from eval._backends.precip.sources import (
    LresInterpBaseline,
    PrecipTruthSource,
    build_nn_index,
    check_grid_match,
    is_degenerate_channel,
    maybe_deaccumulate,
)


# ---------------------------------------------------------------------------
# maybe_deaccumulate
# ---------------------------------------------------------------------------

def test_deaccumulates_running_total():
    per_window = {6: np.array([1.0, 2.0]), 12: np.array([3.0, 1.0]),
                  18: np.array([0.5, 0.5]), 24: np.array([2.0, 2.0])}
    accumulated, run = {}, np.zeros(2)
    for s in sorted(per_window):
        run = run + per_window[s]
        accumulated[s] = run.copy()
    out, was_acc = maybe_deaccumulate(accumulated, context="test")
    assert was_acc
    for s in per_window:
        np.testing.assert_allclose(out[s], per_window[s])


def test_windowed_series_untouched():
    windowed = {6: np.array([1.0, 2.0]), 12: np.array([3.0, 1.0]),
                18: np.array([0.5, 0.5]), 24: np.array([2.0, 2.0])}
    out, was_acc = maybe_deaccumulate(dict(windowed), context="test")
    assert not was_acc
    for s in windowed:
        np.testing.assert_array_equal(out[s], windowed[s])


def test_constant_series_untouched():
    # Non-decreasing but not growing: a legitimate constant field must never
    # be silently differenced to zero.
    const = {s: np.array([2.0, 2.0]) for s in (6, 12, 18, 24)}
    out, was_acc = maybe_deaccumulate(dict(const), context="test")
    assert not was_acc
    np.testing.assert_array_equal(out[24], const[24])


# ---------------------------------------------------------------------------
# grid check / degenerate probe
# ---------------------------------------------------------------------------

def test_grid_match_accepts_identical_and_wrapped_lons():
    lat = np.linspace(-80, 80, 500)
    lon = np.linspace(0, 359, 500)
    check_grid_match(lat, lon, lat, lon, context="t")
    lon_wrapped = np.where(lon > 180, lon - 360.0, lon)
    check_grid_match(lat, lon_wrapped, lat, lon, context="t")


def test_grid_match_rejects_size_and_order_mismatch():
    lat = np.linspace(-80, 80, 500)
    lon = np.linspace(0, 359, 500)
    with pytest.raises(ValueError, match="size mismatch"):
        check_grid_match(lat, lon, lat[:-1], lon[:-1], context="t")
    with pytest.raises(ValueError, match="ordering mismatch"):
        check_grid_match(lat, lon, lat[::-1], lon, context="t")


def test_is_degenerate_channel():
    assert is_degenerate_channel(np.zeros(100))
    assert is_degenerate_channel(np.full(100, np.nan))
    assert is_degenerate_channel(np.array([]))
    assert not is_degenerate_channel(np.array([0.0, 0.1, 0.0]))


# ---------------------------------------------------------------------------
# PrecipTruthSource with an injected reader
# ---------------------------------------------------------------------------

def _fake_truth_reader(date):
    lats = np.linspace(-10, 10, 8)
    lons = np.linspace(100, 120, 8)
    vals = {(0, s): np.full(8, 0.001 * (i + 1) * ((-1) ** i + 1.5))
            for i, s in enumerate((6, 12, 18, 24))}
    return vals, lats, lons


def test_truth_source_load_and_grid():
    src = PrecipTruthSource("/nowhere/{date}.grib", _reader=_fake_truth_reader)
    assert src.steps("20250926") == [6, 12, 18, 24]
    v = src.load("20250926", 12)
    assert v.shape == (8,)
    lats = np.linspace(-10, 10, 8)
    lons = np.linspace(100, 120, 8)
    src.verify_grid(lats, lons)
    with pytest.raises(KeyError):
        src.load("20250926", 30)


# ---------------------------------------------------------------------------
# LresInterpBaseline: nearest-neighbour index + member selection
# ---------------------------------------------------------------------------

def test_build_nn_index_picks_true_nearest():
    src_lat = np.array([0.0, 0.0, 10.0])
    src_lon = np.array([0.0, 5.0, 0.0])
    dst_lat = np.array([0.1, 9.0, 0.0])
    dst_lon = np.array([4.4, 0.5, 0.1])
    idx = build_nn_index(src_lat, src_lon, dst_lat, dst_lon)
    np.testing.assert_array_equal(idx, [1, 2, 0])


def _fake_baseline_reader(date):
    src_lat = np.array([0.0, 0.0, 10.0])
    src_lon = np.array([0.0, 5.0, 0.0])
    vals = {}
    for member in (1, 2):
        for step in (6, 12):
            vals[(member, step)] = np.array([member + step, member * 10.0, step * 1.0])
    return vals, src_lat, src_lon


def test_baseline_interp_load(tmp_path):
    cache = tmp_path / "nn.npz"
    src = LresInterpBaseline("/nowhere/{date}.grib", cache,
                             _reader=_fake_baseline_reader)
    dst_lat = np.array([0.1, 9.0])
    dst_lon = np.array([4.4, 0.5])
    src.ensure_index(dst_lat, dst_lon, probe_date="20250926")
    out = src.load("20250926", 6, 2)
    # dst point 0 -> src 1 (value member*10=20), dst point 1 -> src 2 (step=6)
    np.testing.assert_allclose(out, [20.0, 6.0])
    assert cache.exists()
    # cache round-trip: a new instance loads the same index without scipy work
    src2 = LresInterpBaseline("/nowhere/{date}.grib", cache,
                              _reader=_fake_baseline_reader)
    src2.ensure_index(dst_lat, dst_lon, probe_date="20250926")
    np.testing.assert_array_equal(src2._nn_index, src._nn_index)
    with pytest.raises(KeyError):
        src.load("20250926", 6, 99)
