"""Tests for PrepML YAML config generation."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _sample_lane_config(
    *,
    members: list[int] | None = None,
    num_gpus_per_model: int | None = None,
) -> dict:
    predict = {
        "dates": ["20230826", "20230827", "20230828", "20230829", "20230830"],
        "members": members or [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "steps": [24, 48, 72, 96, 120],
        "sampler": {
            "schedule_type": "experimental_piecewise",
            "num_steps": 30,
            "sigma_max": 1000.0,
            "sigma_min": 0.03,
            "sampler": "heun",
        },
    }
    if num_gpus_per_model is not None:
        predict["num_gpus_per_model"] = num_gpus_per_model
    return {
        "predict": predict,
        "prepml": {
            "runner": "anemoi-dev",
            "venv": "/home/ecm5702/dev/.ds-dyn-wrap",
            "input": {"class": "od", "stream": "eefo", "type": "pf", "grid": "O96"},
            "output": {"class": "rd", "stream": "enfo", "type": "pf"},
            "output_template": "/data/o320-template.grib",
            "forcings_npz": "/data/o320-forcings.npz",
            "constant_high_res_forcings": ["cos_latitude", "lsm", "z"],
            "high_res_input": ["cos_julian_day", "lsm", "z"],
            "time_step": "24h",
            "lead_time": "240h",
            "platform": {"gpu": {"time": "0-12:18"}},
        },
    }


def test_generate_prepml_yaml_structure():
    from eval.predict.prepml_config import generate_prepml_config
    config = generate_prepml_config(
        lane_config=_sample_lane_config(),
        checkpoint_path="/path/to/checkpoint.ckpt",
    )
    assert config["model"]["checkpoint"] == "/path/to/checkpoint.ckpt"
    assert config["model"]["name"] == "anemoi"
    assert config["model"]["runner"] == "downscaling"
    assert config["model"]["lead_time"] == "120h"
    assert config["runner"]["name"] == "anemoi-dev"
    assert config["runner"]["venv"] == "/home/ecm5702/dev/.ds-dyn-wrap"
    assert config["input"]["grid"] == "O96"
    assert config["output"]["class"] == "rd"
    assert config["evaluation"] is False


def test_generate_prepml_yaml_dates():
    from eval.predict.prepml_config import generate_prepml_config
    config = generate_prepml_config(
        lane_config=_sample_lane_config(),
        checkpoint_path="/path/to/checkpoint.ckpt",
    )
    assert config["dates"]["start"] == "2023-08-26"
    assert config["dates"]["end"] == "2023-08-30"
    assert config["dates"]["frequency"] == 24


def test_generate_prepml_yaml_sampler():
    from eval.predict.prepml_config import generate_prepml_config
    config = generate_prepml_config(
        lane_config=_sample_lane_config(),
        checkpoint_path="/path/to/checkpoint.ckpt",
    )
    extra_args = config["model"]["development_hacks"]["extra_args"]
    assert extra_args["schedule_type"] == "experimental_piecewise"
    assert extra_args["num_steps"] == 30


def test_generate_prepml_yaml_members():
    from eval.predict.prepml_config import generate_prepml_config
    config = generate_prepml_config(
        lane_config=_sample_lane_config(),
        checkpoint_path="/path/to/checkpoint.ckpt",
    )
    assert config["ensemble"]["loop"]["number"] == "1/to/10"


def test_generate_prepml_yaml_single_member_loop():
    from eval.predict.prepml_config import generate_prepml_config
    config = generate_prepml_config(
        lane_config=_sample_lane_config(members=[1]),
        checkpoint_path="/path/to/checkpoint.ckpt",
    )
    assert config["ensemble"]["loop"]["number"] == "1"


def test_write_prepml_yaml(tmp_path):
    from eval.predict.prepml_config import generate_prepml_config, write_prepml_config
    config = generate_prepml_config(
        lane_config=_sample_lane_config(),
        checkpoint_path="/path/to/checkpoint.ckpt",
    )
    out_path = write_prepml_config(config, tmp_path / "prepml_config.yaml")
    assert out_path.exists()
    loaded = yaml.safe_load(out_path.read_text())
    assert loaded["model"]["checkpoint"] == "/path/to/checkpoint.ckpt"


def test_runner_override():
    from eval.predict.prepml_config import generate_prepml_config
    config = generate_prepml_config(
        lane_config=_sample_lane_config(),
        checkpoint_path="/path/to/checkpoint.ckpt",
        runner_override="/custom/venv/path",
    )
    assert config["runner"]["venv"] == "/custom/venv/path"


def test_generate_prepml_yaml_multi_gpu_model_parallel_runner():
    from eval.predict.prepml_config import generate_prepml_config
    config = generate_prepml_config(
        lane_config=_sample_lane_config(num_gpus_per_model=4),
        checkpoint_path="/path/to/checkpoint.ckpt",
    )
    assert config["model"]["runner"] == {"parallel": {"base_runner": "downscaling"}}
    assert config["model"]["world_size"] == 4
    submit_args = config["platform"]["flavours"]["gpu"]["submit_arguments"]
    assert submit_args["gpus_per_node"] == 4
    assert "#SBATCH --gres=gpu:4" in submit_args["RAW_PRAGMA"]
