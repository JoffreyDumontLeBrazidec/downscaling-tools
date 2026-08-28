"""Tests for plot_member_wind_maps pure helpers (no rendering)."""
from __future__ import annotations

import numpy as np
import pytest

from eval._backends.region_plotting.plot_member_wind_maps import (
    VARIABLES,
    _field,
    _parse_kv,
    build_arg_parser,
    nearest_grid,
    resolve_scale,
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
    assert args.proj_lon == 5.0 and args.proj_lat == 50.0
    assert args.region_tag == "europe-cutout"
    # The default variable and its resolved colour scale must stay exactly what
    # the tool did before --variable existed.
    assert args.variable == "wind10m"
    spec, vmin, vmax = resolve_scale(args)
    assert (spec["token"], vmin, vmax) == ("10mwind", 0.0, 25.0)


def test_msl_variable_resolves_its_own_scale_and_token():
    args = build_arg_parser().parse_args(
        ["--date", "20250926", "--step", "24", "--member", "2", "--output-dir", "/tmp/x",
         "--variable", "msl"]
    )
    spec, vmin, vmax = resolve_scale(args)
    assert (spec["token"], vmin, vmax) == ("msl", 960.0, 1040.0)
    assert spec["states"] == ("msl",)


def test_explicit_scale_overrides_the_variable_default():
    args = build_arg_parser().parse_args(
        ["--date", "20250926", "--step", "24", "--member", "2", "--output-dir", "/tmp/x",
         "--variable", "msl", "--vmin", "980", "--vmax", "1020"]
    )
    _, vmin, vmax = resolve_scale(args)
    assert (vmin, vmax) == (980.0, 1020.0)


def test_field_rejects_a_missing_weather_state():
    spec = VARIABLES["msl"]
    arr = np.zeros((4, 2))
    with pytest.raises(SystemExit):
        _field(arr, ["10u", "10v"], spec)


def test_field_converts_pressure_to_hectopascals():
    arr = np.array([[1.0, 2.0, 3.0, 101325.0]])
    val = _field(arr, ["10u", "10v", "2t", "msl"], VARIABLES["msl"])
    assert val[0] == pytest.approx(1013.25)


def test_field_wind_speed_matches_the_hypotenuse():
    arr = np.array([[3.0, 4.0, 0.0, 0.0]])
    val = _field(arr, ["10u", "10v", "2t", "msl"], VARIABLES["wind10m"])
    assert val[0] == pytest.approx(5.0)


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
