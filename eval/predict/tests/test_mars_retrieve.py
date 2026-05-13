"""Tests for MARS retrieval and weather state mapping."""
from __future__ import annotations

import pytest


def test_map_surface_state():
    from eval.predict.mars_retrieve import weather_state_to_mars
    result = weather_state_to_mars("2t")
    assert result == {"param": "2t", "levtype": "sfc"}


def test_map_pressure_level_state():
    from eval.predict.mars_retrieve import weather_state_to_mars
    result = weather_state_to_mars("z_500")
    assert result == {"param": "z", "levtype": "pl", "level": 500}


def test_map_pressure_level_t850():
    from eval.predict.mars_retrieve import weather_state_to_mars
    result = weather_state_to_mars("t_850")
    assert result == {"param": "t", "levtype": "pl", "level": 850}


def test_map_unknown_state_raises():
    from eval.predict.mars_retrieve import weather_state_to_mars
    with pytest.raises(ValueError, match="Unknown weather state"):
        weather_state_to_mars("unknown_var")


def test_group_states_by_levtype():
    from eval.predict.mars_retrieve import group_weather_states_for_mars
    states = ["2t", "10u", "10v", "sp", "z_500", "t_850"]
    result = group_weather_states_for_mars(states)
    assert result["sfc"]["params"] == ["2t", "10u", "10v", "sp"]
    assert result["pl"]["params"] == ["z", "t"]
    assert sorted(result["pl"]["levels"]) == [500, 850]


def test_build_mars_prediction_request():
    from eval.predict.mars_retrieve import build_prediction_request
    request = build_prediction_request(
        expver="j2pw",
        date="20230826",
        step=24,
        members=[1, 2, 3],
        output_mars={"class": "rd", "stream": "enfo", "type": "pf"},
        weather_states=["2t", "10u", "z_500"],
    )
    assert len(request) == 2
    sfc_req = request[0]
    assert sfc_req["expver"] == "j2pw"
    assert sfc_req["date"] == "20230826"
    assert sfc_req["step"] == 24
    assert sfc_req["levtype"] == "sfc"
    assert "2t" in sfc_req["param"]
    assert "10u" in sfc_req["param"]
    pl_req = request[1]
    assert pl_req["levtype"] == "pl"
    assert "z" in pl_req["param"]
    assert 500 in pl_req["levelist"]


def test_build_mars_prediction_request_sfc_only():
    from eval.predict.mars_retrieve import build_prediction_request
    request = build_prediction_request(
        expver="j2pw",
        date="20230826",
        step=24,
        members=[1],
        output_mars={"class": "rd", "stream": "enfo", "type": "pf"},
        weather_states=["2t", "10u", "sp"],
    )
    assert len(request) == 1
    assert request[0]["levtype"] == "sfc"


def _write_bundle(path, *, surface_params, pl_bases=(), pl_levels=()):
    """Helper: write a minimal netCDF bundle exposing target_hres_* variables."""
    import netCDF4 as nc

    ds = nc.Dataset(str(path), "w")
    try:
        ds.createDimension("point_hres", 4)
        for p in surface_params:
            v = ds.createVariable(f"target_hres_{p}", "f4", ("point_hres",))
            v[:] = 0
        if pl_bases and pl_levels:
            ds.createDimension("target_level", len(pl_levels))
            tl = ds.createVariable("target_level", "i4", ("target_level",))
            tl[:] = pl_levels
            for base in pl_bases:
                v = ds.createVariable(
                    f"target_hres_{base}", "f4", ("target_level", "point_hres"),
                )
                v[:] = 0
    finally:
        ds.close()


def test_discover_weather_states_surface_only(tmp_path):
    from eval.predict.mars_retrieve import discover_weather_states_from_bundle

    bundle = tmp_path / "minimal.nc"
    _write_bundle(bundle, surface_params=("10u", "10v", "2t", "msl", "2d", "skt", "sp", "tcw"))

    states = discover_weather_states_from_bundle(bundle, mode="all-surface-only")
    assert set(states) == {"10u", "10v", "2t", "msl", "2d", "skt", "sp", "tcw"}


def test_discover_weather_states_surface_plus_core_pl(tmp_path):
    """surface-plus-core-pl includes t_850 and z_500 when bundle PL stack supports it."""
    from eval.predict.mars_retrieve import discover_weather_states_from_bundle

    bundle = tmp_path / "with_pl.nc"
    _write_bundle(
        bundle,
        surface_params=("10u", "10v", "2t", "msl", "2d", "skt", "sp", "tcw"),
        pl_bases=("t", "z", "q", "u", "v", "w"),
        pl_levels=(1000, 925, 850, 700, 500, 400, 300, 200, 100, 50),
    )

    states = discover_weather_states_from_bundle(bundle)
    assert set(states) == {
        "10u", "10v", "2t", "msl", "2d", "skt", "sp", "tcw",
        "t_850", "z_500",
    }


def test_discover_weather_states_skips_unknown_surface_params(tmp_path):
    """Surface params not in the canonical _SURFACE_PARAMS list are silently dropped."""
    from eval.predict.mars_retrieve import discover_weather_states_from_bundle

    bundle = tmp_path / "with_unknown.nc"
    _write_bundle(bundle, surface_params=("10u", "fancy_param", "2t"))

    states = discover_weather_states_from_bundle(bundle, mode="all-surface-only")
    assert set(states) == {"10u", "2t"}


def test_discover_weather_states_pl_excluded_when_level_missing(tmp_path):
    """If target_level doesn't contain 500/850, the PL states are excluded."""
    from eval.predict.mars_retrieve import discover_weather_states_from_bundle

    bundle = tmp_path / "pl_levels_partial.nc"
    _write_bundle(
        bundle,
        surface_params=("10u", "2t"),
        pl_bases=("z", "t"),
        pl_levels=(1000, 925),  # no 500 or 850
    )

    states = discover_weather_states_from_bundle(bundle)
    assert set(states) == {"10u", "2t"}


def test_discover_weather_states_unsupported_mode_raises(tmp_path):
    from eval.predict.mars_retrieve import discover_weather_states_from_bundle

    bundle = tmp_path / "x.nc"
    _write_bundle(bundle, surface_params=("2t",))

    import pytest as _pytest
    with _pytest.raises(ValueError, match="Unsupported weather_states discovery mode"):
        discover_weather_states_from_bundle(bundle, mode="silly")


def test_discover_weather_states_missing_bundle_raises(tmp_path):
    from eval.predict.mars_retrieve import discover_weather_states_from_bundle

    import pytest as _pytest
    with _pytest.raises(FileNotFoundError):
        discover_weather_states_from_bundle(tmp_path / "nope.nc")
