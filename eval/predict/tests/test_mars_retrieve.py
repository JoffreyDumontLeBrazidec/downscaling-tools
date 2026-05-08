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
