"""Tests for plot_member_wind_maps pure helpers (no rendering)."""
from __future__ import annotations

import numpy as np
import pytest

from eval._backends.region_plotting.plot_member_wind_maps import (
    _parse_kv,
    build_arg_parser,
    nearest_grid,
)


def test_parse_kv_order_and_values():
    parsed = _parse_kv(["guided=/a/b", "control=/c/d"], "run")
    assert list(parsed.items()) == [("guided", "/a/b"), ("control", "/c/d")]


def test_parse_kv_rejects_bad_spec():
    with pytest.raises(SystemExit):
        _parse_kv(["nodelimiter"], "run")


def test_build_arg_parser_defaults():
    args = build_arg_parser().parse_args(
        ["--date", "20250926", "--step", "24", "--member", "2", "--output-dir", "/tmp/x"]
    )
    assert args.extent == [-45.0, 55.0, 27.0, 72.0]
    assert args.vmax == 25.0
    assert args.proj_lon == 5.0 and args.proj_lat == 50.0
    assert args.region_tag == "europe-cutout"


def test_nearest_grid_fills_every_cell_with_nearest_value():
    # Two point clusters with distinct values; every grid cell must carry the
    # value of its nearest cluster — no gaps, no averaging.
    extent = (0.0, 10.0, 0.0, 10.0)
    lat = np.array([2.0, 2.0, 8.0, 8.0])
    lon = np.array([2.0, 2.5, 8.0, 8.5])
    val = np.array([1.0, 1.0, 5.0, 5.0])
    gx, gy, grid = nearest_grid(lat, lon, val, extent=extent, margin=1.0, res=1.0)
    assert grid.shape == (len(gy), len(gx))
    assert not np.isnan(grid).any()
    assert set(np.unique(grid)) == {1.0, 5.0}
    # Cell nearest the (2,2) cluster gets 1.0; nearest the (8,8) cluster gets 5.0.
    assert grid[np.searchsorted(gy, 2.0), np.searchsorted(gx, 2.0)] == 1.0
    assert grid[np.searchsorted(gy, 8.0), np.searchsorted(gx, 8.0)] == 5.0


def test_nearest_grid_raises_outside_extent():
    with pytest.raises(ValueError):
        nearest_grid(
            np.array([50.0]), np.array([120.0]), np.array([1.0]),
            extent=(0.0, 10.0, 0.0, 10.0), margin=1.0, res=1.0,
        )
