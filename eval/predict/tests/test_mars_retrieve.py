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
