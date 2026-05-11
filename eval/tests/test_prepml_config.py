"""Tests for PrepML config generation."""
from __future__ import annotations


def _lane_config(steps: list[int]) -> dict:
    return {
        "predict": {
            "dates": ["20250928"],
            "members": [1, 2],
            "steps": steps,
            "sampler": {"num_steps": 21},
        },
        "prepml": {
            "runner": "anemoi-dev",
            "venv": "/venv",
            "input": {
                "class": "od",
                "stream": "enfo",
                "type": "pf",
                "grid": "O48",
                "step_offset_hours": 6,
            },
            "output": {
                "class": "rd",
                "stream": "enfo",
                "type": "pf",
            },
            "lead_time": "240h",
            "time_step": "24h",
            "output_template": "/tmp/o96-template.grib",
            "forcings_npz": "/tmp/o96-forcings.npz",
            "constant_high_res_forcings": ["lsm", "z"],
            "high_res_input": ["lsm", "z"],
            "platform": {"gpu": {"time": "0-12:18"}},
        },
    }


def test_prepml_lead_time_uses_max_requested_step_for_smoke_scope():
    from eval.predict.prepml_config import generate_prepml_config

    config = generate_prepml_config(
        lane_config=_lane_config([24]),
        checkpoint_path="/ckpt/inference-last.ckpt",
    )

    assert config["model"]["lead_time"] == "24h"
    assert config["input"]["step"] == "6/to/30/by/24"


def test_prepml_lead_time_uses_max_requested_step_for_multi_step_scope():
    from eval.predict.prepml_config import generate_prepml_config

    config = generate_prepml_config(
        lane_config=_lane_config([24, 72, 120]),
        checkpoint_path="/ckpt/inference-last.ckpt",
    )

    assert config["model"]["lead_time"] == "120h"
    assert config["input"]["step"] == "6/to/126/by/24"
