"""Tests for plot_member_wind_maps pure helpers (no rendering)."""
from __future__ import annotations

import numpy as np
import pytest

from eval.evaluators.zoom_maps.core.plot_member_wind_maps import (
    MIN_SPAN_DEG,
    VARIABLES,
    _field,
    _parse_kv,
    box_text,
    build_arg_parser,
    highpass_at_points,
    nearest_grid,
    resolve_extent,
    resolve_scale,
    widen_extent,
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
    # The projection centre defaults to the centre of the box drawn.
    assert args.proj_lon is None and args.proj_lat is None
    extent, _ = resolve_extent(args)
    assert extent == (-45.0, 55.0, 27.0, 72.0)
    assert (args.proj_lon, args.proj_lat) == (5.0, 49.5)
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


def test_2t_variable_resolves_its_own_scale_and_token():
    args = build_arg_parser().parse_args(
        ["--date", "20250926", "--step", "24", "--member", "2", "--output-dir", "/tmp/x",
         "--variable", "2t"]
    )
    spec, vmin, vmax = resolve_scale(args)
    # Temperatures are shown in K (house rule); same physical range as the former -20..35 degC.
    assert (spec["token"], vmin, vmax) == ("2t", 253.15, 308.15)
    assert spec["states"] == ("2t",)


def test_field_keeps_temperature_in_kelvin():
    arr = np.array([[0.0, 0.0, 273.15, 0.0]])
    val = _field(arr, ["10u", "10v", "2t", "msl"], VARIABLES["2t"])
    assert val[0] == pytest.approx(273.15)


def test_every_variable_declares_a_complete_spec():
    keys = {"states", "combine", "token", "scale", "offset", "cmap",
            "vmin", "vmax", "extend", "subtitle", "cbar_label", "fine_vmax", "house_key"}
    for name, spec in VARIABLES.items():
        assert set(spec) == keys, name
        assert spec["combine"] in ("hypot", "single"), name
        assert spec["vmin"] < spec["vmax"], name


def test_names_units_and_colour_maps_come_from_the_house_table():
    from eval.plotting import variable_spec

    assert VARIABLES["msl"]["cbar_label"] == "Mean sea level pressure (hPa)"
    assert VARIABLES["z_500"]["cbar_label"] == "500 hPa geopotential height (dam)"
    assert VARIABLES["2t"]["cbar_label"] == "2 m temperature (K)"
    for name, spec in VARIABLES.items():
        assert spec["cmap"] == variable_spec(spec["house_key"]).cmap, name
        assert spec["cmap"] not in ("RdBu_r", "RdYlBu_r", "jet", "rainbow"), name


def test_z500_conversion_matches_the_house_table():
    arr = np.array([[5500.0 * 9.80665]])
    val = _field(arr, ["z_500"], VARIABLES["z_500"])
    assert val[0] == pytest.approx(550.0)


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


def test_widen_extent_keeps_a_large_box_and_widens_a_small_one_around_its_centre():
    wta = (-70.0, -45.0, 10.0, 25.0)
    assert widen_extent(wta) == wta
    alps = widen_extent((4.0, 16.0, 43.0, 49.0))
    lon_min, lon_max, lat_min, lat_max = alps
    assert (lat_min, lat_max) == (40.0, 52.0)
    assert 0.5 * (lon_min + lon_max) == pytest.approx(10.0, abs=0.01)
    # east-west span measured in degrees of latitude at the centre (46 N)
    assert (lon_max - lon_min) * np.cos(np.radians(46.0)) == pytest.approx(MIN_SPAN_DEG, abs=0.02)
    # a box that crosses the dateline is left alone
    assert widen_extent((170.0, -170.0, 0.0, 5.0)) == (170.0, -170.0, 0.0, 5.0)


def test_resolve_extent_centres_the_projection_on_the_box_drawn(capsys):
    args = build_arg_parser().parse_args(
        ["--date", "20230828", "--step", "24", "--output-dir", "/tmp/x",
         "--extent", "-76", "-64", "22", "34"]
    )
    extent, line = resolve_extent(args)
    assert extent[2:] == (22.0, 34.0)
    assert args.proj_lon == pytest.approx(-70.0) and args.proj_lat == pytest.approx(28.0)
    assert line.startswith("map box ") and "widened from 76.0–64.0°W, 22.0–34.0°N" in line
    assert "extent used" in capsys.readouterr().out


def test_box_text_names_hemispheres():
    assert box_text((-76.0, -64.0, 22.0, 34.0)) == "76.0–64.0°W, 22.0–34.0°N"
    assert box_text((-10.0, 25.0, -5.0, 5.0)) == "10.0°W–25.0°E, 5.0°S–5.0°N"


def test_highpass_at_points_keeps_grid_scale_detail_and_removes_the_large_scale():
    # Points on a 0.1 degree lattice: a smooth large-scale ramp plus a
    # checkerboard at the grid scale. The high-pass at the points must return the
    # checkerboard with its full amplitude and no trace of the ramp.
    lon1 = np.arange(0.0, 20.0, 0.1)
    lat1 = np.arange(40.0, 55.0, 0.1)
    lon, lat = (a.ravel() for a in np.meshgrid(lon1, lat1))
    i, j = (a.ravel() for a in np.meshgrid(np.arange(lon1.size), np.arange(lat1.size)))
    checker = np.where((i + j) % 2 == 0, 1.0, -1.0)
    ramp = 0.5 * lon + 0.2 * lat
    fine = highpass_at_points(lat, lon, ramp + checker, extent=(6.0, 14.0, 44.0, 51.0),
                              res=0.1, fine_cut_deg=0.6, margin=3.0)
    inner = (lon > 6) & (lon < 14) & (lat > 44) & (lat < 51)
    assert np.isfinite(fine[inner]).all()
    assert np.corrcoef(fine[inner], checker[inner])[0, 1] > 0.99
    assert np.abs(fine[inner] - checker[inner]).max() < 0.05

