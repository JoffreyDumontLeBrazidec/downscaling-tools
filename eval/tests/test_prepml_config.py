"""Tests for PrepML config generation."""
from __future__ import annotations


def _lane_config(
    steps: list[int],
    *,
    members: list[int] | None = None,
    num_gpus_per_model: int | None = None,
) -> dict:
    predict = {
        "dates": ["20250928"],
        "members": members or [1, 2],
        "steps": steps,
        "sampler": {"num_steps": 21},
    }
    if num_gpus_per_model is not None:
        predict["num_gpus_per_model"] = num_gpus_per_model
    return {
        "predict": predict,
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


def test_prepml_forwards_predict_env_to_anemoi_model_env():
    from eval.predict.prepml_config import generate_prepml_config

    lane_config = _lane_config([24], num_gpus_per_model=4)
    lane_config["predict"]["env"] = {
        "ANEMOI_INFERENCE_NUM_CHUNKS": 32,
        "ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR": 32,
        "ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER": 32,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    config = generate_prepml_config(
        lane_config=lane_config,
        checkpoint_path="/ckpt/inference-last.ckpt",
    )

    assert config["model"]["env"] == lane_config["predict"]["env"]


def test_prepml_uses_parallel_runner_for_multi_gpu_model():
    from eval.predict.prepml_config import generate_prepml_config

    config = generate_prepml_config(
        lane_config=_lane_config([24], num_gpus_per_model=4),
        checkpoint_path="/ckpt/inference-last.ckpt",
    )

    assert config["model"]["runner"] == {"parallel": {"base_runner": "downscaling"}}
    assert config["model"]["world_size"] == 4
    submit_args = config["platform"]["flavours"]["gpu"]["submit_arguments"]
    assert submit_args["gpus_per_node"] == 4
    assert "#SBATCH --gres=gpu:4" in submit_args["RAW_PRAGMA"]


def test_prepml_single_member_loop_uses_scalar_value():
    from eval.predict.prepml_config import generate_prepml_config

    config = generate_prepml_config(
        lane_config=_lane_config([24], members=[1]),
        checkpoint_path="/ckpt/inference-last.ckpt",
    )

    assert config["ensemble"]["loop"]["number"] == "1"
