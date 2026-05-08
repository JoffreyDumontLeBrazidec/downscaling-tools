"""Tests for PrepML orchestration."""
from __future__ import annotations

import pytest


def _lane_config_with_prepml() -> dict:
    return {
        "predict": {
            "dates": ["20230826", "20230827"],
            "members": [1, 2],
            "steps": [24, 48],
            "sampler": {"num_steps": 30},
        },
        "prepml": {
            "debug_expvers": ["dbg_test_1", "dbg_test_2"],
            "runner": "anemoi-dev",
            "venv": "/path/to/venv",
            "input": {"class": "od", "stream": "eefo", "type": "pf", "grid": "O96"},
            "output": {"class": "rd", "stream": "enfo", "type": "pf"},
            "output_template": "/data/template.grib",
            "forcings_npz": "/data/forcings.npz",
            "constant_high_res_forcings": ["z"],
            "high_res_input": ["z"],
            "truth": {"root": "/perm/reference/o96_o320"},
            "time_step": "24h",
            "lead_time": "240h",
            "platform": {"gpu": {"time": "0-12:18"}},
        },
        "evaluator_groups": {"default": []},
    }


def test_resolve_expver_explicit():
    from eval.predict.prepml import resolve_expver
    result = resolve_expver("j2pw", _lane_config_with_prepml())
    assert result == "j2pw"


def test_resolve_expver_debug_default():
    from eval.predict.prepml import resolve_expver
    result = resolve_expver(None, _lane_config_with_prepml())
    assert result == "dbg_test_1"


def test_resolve_expver_no_debug_pool_raises():
    from eval.predict.prepml import resolve_expver
    config = _lane_config_with_prepml()
    config["prepml"]["debug_expvers"] = []
    with pytest.raises(ValueError, match="No debug expvers"):
        resolve_expver(None, config)


def test_resolve_expver_no_prepml_section_raises():
    from eval.predict.prepml import resolve_expver
    config = {"predict": {}, "evaluator_groups": {"default": []}}
    with pytest.raises(ValueError, match="No 'prepml' section"):
        resolve_expver(None, config)
