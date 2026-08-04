"""Tests for eval.jobs.resources."""

from __future__ import annotations

from eval.config.loader import load_host, load_lane
from eval.jobs.resources import resolve_resources


def test_full_contract_predict_gets_safety_time_floor():
    lane_config = {
        "predict": {
            "members": list(range(1, 11)),
            "dates": ["20230826", "20230827", "20230828", "20230829", "20230830"],
            "steps": [24, 48, 72, 96, 120],
        },
        "resource_profiles": {
            "predict": {
                "gpus": 2,
                "time": "02:00:00",
                "mem": "0",
                "cpus": 8,
            },
        },
    }
    host_config = {
        "scheduler": {
            "qos": "nf",
            "default_time": "04:00:00",
            "default_mem": "64G",
            "default_cpus": 16,
        },
    }

    resources = resolve_resources(lane_config, host_config, stage="predict")

    assert resources["time"] == "48:00:00"
    assert resources["qos"] == "ng"


def test_predict_stage_defaults_to_48_hours_without_profile():
    lane_config = {
        "predict": {
            "members": [1],
            "dates": ["20230826"],
            "steps": [24],
        },
    }
    host_config = {
        "scheduler": {
            "qos": "nf",
            "default_time": "04:00:00",
            "default_mem": "64G",
            "default_cpus": 16,
        },
    }

    resources = resolve_resources(lane_config, host_config, stage="predict")

    assert resources["time"] == "48:00:00"
    assert resources["qos"] == "nf"


def test_pristine_o96_o320_predict_inherits_safe_base_time():
    lane_config = load_lane("o96_o320_pristine_adamw_aug2630_10m")
    host_config = load_host("atos_ac_pristine")

    resources = resolve_resources(lane_config, host_config, stage="predict")

    assert resources["time"] == "48:00:00"


def test_evaluate_stage_defaults_to_48_hours():
    """Long evaluate jobs (quaver/tc/spectra) must not inherit a short wall.

    Regression: 2026-07-06 the j7xn quaver month job was TIME_LIMIT-killed at an
    8h wall; the harness default for the evaluate stage was the host default_time
    (4h).  Evaluate is a long stage and must default to 48h like predict.
    """
    lane_config = {"predict": {"members": [1], "dates": ["20230801"], "steps": [24]}}
    host_config = {
        "scheduler": {
            "qos": "nf",
            "default_time": "04:00:00",
            "default_mem": "64G",
            "default_cpus": 16,
        },
    }

    resources = resolve_resources(
        lane_config, host_config, stage="evaluate", evaluator="quaver"
    )

    assert resources["time"] == "48:00:00"


def test_evaluate_stage_respects_host_override():
    host_config = {
        "scheduler": {
            "qos": "nf",
            "default_time": "04:00:00",
            "default_evaluate_time": "12:00:00",
            "default_mem": "64G",
            "default_cpus": 16,
        },
    }

    resources = resolve_resources({}, host_config, stage="evaluate")

    assert resources["time"] == "12:00:00"


def test_scoreboard_stage_stays_short():
    """Scoreboard is a pure renderer; it must NOT be bumped to the 48h long-wall."""
    host_config = {
        "scheduler": {
            "qos": "nf",
            "default_time": "04:00:00",
            "default_mem": "64G",
            "default_cpus": 16,
        },
    }

    resources = resolve_resources({}, host_config, stage="scoreboard")

    assert resources["time"] == "04:00:00"
