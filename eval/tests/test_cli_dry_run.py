"""Tests for CLI --dry-run mode."""

from __future__ import annotations

import os
import subprocess
import sys


CODE_ROOT = "/home/ecm5702/dev/downscaling-tools"


def _cli_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = CODE_ROOT + ":" + env.get("PYTHONPATH", "")
    return env


def test_cli_evaluate_dry_run(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "eval.cli", "evaluate",
            "--dry-run",
            "--lane", "o96_o320",
            "--predictions-dir", str(tmp_path),
        ],
        capture_output=True, text=True, env=_cli_env(), cwd=CODE_ROOT,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert '"lane": "o96_o320"' in result.stdout


def test_cli_run_dry_run():
    result = subprocess.run(
        [
            sys.executable, "-m", "eval.cli", "run",
            "--dry-run",
            "--lane", "o96_o320",
            "--checkpoint", "/tmp/test.ckpt",
        ],
        capture_output=True, text=True, env=_cli_env(), cwd=CODE_ROOT,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert '"lane": "o96_o320"' in result.stdout


def test_cli_evaluate_rejects_quaver_only(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "eval.cli", "evaluate",
            "--dry-run",
            "--lane", "o96_o320",
            "--predictions-dir", str(tmp_path),
            "--only", "quaver",
        ],
        capture_output=True, text=True, env=_cli_env(), cwd=CODE_ROOT,
    )
    assert result.returncode != 0
    assert "Unknown evaluator(s) in --only: ['quaver']" in result.stderr


def test_cli_predict_num_gpus_per_model_override():
    """--num-gpus-per-model on CLI surfaces in resolved predict config."""
    result = subprocess.run(
        [
            sys.executable, "-m", "eval.cli", "predict",
            "--dry-run",
            "--lane", "o1280_o2560",
            "--checkpoint", "/tmp/test.ckpt",
            "--num-gpus-per-model", "2",
        ],
        capture_output=True, text=True, env=_cli_env(), cwd=CODE_ROOT,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert '"num_gpus_per_model": 2' in result.stdout


def test_cli_predict_weather_states_override():
    """--weather-states on CLI surfaces in resolved predict config as a list."""
    result = subprocess.run(
        [
            sys.executable, "-m", "eval.cli", "predict",
            "--dry-run",
            "--lane", "o96_o320",
            "--checkpoint", "/tmp/test.ckpt",
            "--weather-states", "10u,2t,z_500",
        ],
        capture_output=True, text=True, env=_cli_env(), cwd=CODE_ROOT,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    # The override propagates through _build_lane_overrides -> predict.weather_states.
    assert '"weather_states": [' in result.stdout
    assert '"10u"' in result.stdout
    assert '"2t"' in result.stdout
    assert '"z_500"' in result.stdout


def test_cli_exports_host_env_vars(tmp_path, monkeypatch):
    """Host config environment_setup.exports must be applied to os.environ."""
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("GRID_DIR", raising=False)
    # Invoke a dry-run as a subprocess that prints DATA_DIR after eval.cli main has run.
    # Easier: import main directly and verify os.environ side effect.
    from eval import cli as eval_cli
    eval_cli.main([
        "evaluate",
        "--dry-run",
        "--lane", "o1280_o2560",
        "--host", "atos_ag",
        "--predictions-dir", str(tmp_path),
    ])
    # atos_ag.yaml declares DATA_DIR=/home/mlx/ai-ml/datasets/ in environment_setup.exports
    assert os.environ.get("DATA_DIR") == "/home/mlx/ai-ml/datasets/"
    assert os.environ.get("GRID_DIR") == "/home/mlx/ai-ml/grids/"


def test_cli_exports_dont_overwrite_existing(tmp_path, monkeypatch):
    """A pre-existing env var must not be clobbered by host_config exports."""
    monkeypatch.setenv("DATA_DIR", "/user/override")
    from eval import cli as eval_cli
    eval_cli.main([
        "evaluate",
        "--dry-run",
        "--lane", "o1280_o2560",
        "--host", "atos_ag",
        "--predictions-dir", str(tmp_path),
    ])
    assert os.environ.get("DATA_DIR") == "/user/override"


def test_cli_predict_output_dir_override(tmp_path):
    """--output-dir on predict surfaces in resolved config."""
    out = tmp_path / "my_run"
    result = subprocess.run(
        [
            sys.executable, "-m", "eval.cli", "predict",
            "--dry-run",
            "--lane", "o1280_o2560",
            "--checkpoint", "/tmp/test.ckpt",
            "--output-dir", str(out),
        ],
        capture_output=True, text=True, env=_cli_env(), cwd=CODE_ROOT,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert f'"output_dir": "{out}"' in result.stdout


def test_cli_predict_num_gpus_per_model_lane_default():
    """Lane YAML predict.num_gpus_per_model is surfaced without CLI override."""
    result = subprocess.run(
        [
            sys.executable, "-m", "eval.cli", "predict",
            "--dry-run",
            "--lane", "o1280_o2560",
            "--checkpoint", "/tmp/test.ckpt",
        ],
        capture_output=True, text=True, env=_cli_env(), cwd=CODE_ROOT,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert '"num_gpus_per_model": 4' in result.stdout


def test_cli_predict_bundle_dir_without_source_grib_root(tmp_path, monkeypatch):
    """--bundle-dir without --source-grib-root is used as input_root (no rebuild)."""
    from unittest.mock import patch
    from eval import cli as eval_cli

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    bundles = tmp_path / "bundles_with_y"
    bundles.mkdir()
    captured = {}

    def fake_run(cmd, check):
        captured["cmd"] = cmd
        return None

    lane_config = {
        "predict": {
            "members": [1],
            "steps": [24],
            "dates": ["20250926"],
            "num_gpus_per_model": 4,
        },
        "prepare": {"foo": "bar"},
    }
    host_config = {"environment_setup": {"exports": {}}}

    args = type("Args", (), {})()
    args.checkpoint = "/tmp/fake.ckpt"
    args.source_grib_root = None
    args.bundle_dir = str(bundles)
    args.mode = "manual"

    with patch.object(eval_cli.subprocess, "run", side_effect=fake_run):
        eval_cli.cmd_predict(args, lane_config, host_config, tmp_path)

    cmd = captured["cmd"]
    assert "--input-root" in cmd
    input_root_idx = cmd.index("--input-root")
    assert cmd[input_root_idx + 1] == str(bundles)
    # num_gpus_per_model from lane config must be forwarded
    assert "--num-gpus-per-model" in cmd
    ng_idx = cmd.index("--num-gpus-per-model")
    assert cmd[ng_idx + 1] == "4"
    # No SLURM_JOB_ID set → no srun wrap
    assert cmd[0] != "srun"


def test_cli_predict_wraps_in_srun_under_slurm(tmp_path, monkeypatch):
    """num_gpus_per_model > 1 within an sbatch allocation wraps predict.main in srun."""
    from unittest.mock import patch
    from eval import cli as eval_cli

    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    bundles = tmp_path / "bundles_with_y"
    bundles.mkdir()
    captured = {}

    def fake_run(cmd, check):
        captured["cmd"] = cmd
        return None

    lane_config = {
        "predict": {
            "members": [1],
            "steps": [24],
            "dates": ["20250926"],
            "num_gpus_per_model": 4,
        },
        "prepare": {"foo": "bar"},
    }
    host_config = {"environment_setup": {"exports": {}}}

    args = type("Args", (), {})()
    args.checkpoint = "/tmp/fake.ckpt"
    args.source_grib_root = None
    args.bundle_dir = str(bundles)
    args.mode = "manual"

    with patch.object(eval_cli.subprocess, "run", side_effect=fake_run):
        eval_cli.cmd_predict(args, lane_config, host_config, tmp_path)

    cmd = captured["cmd"]
    assert cmd[0] == "srun"
    assert "--ntasks" in cmd
    n_idx = cmd.index("--ntasks")
    assert cmd[n_idx + 1] == "4"
    # No --gpus-per-task: each rank must see all GPUs (model loader uses
    # cuda:<local_rank> and would crash with per-task GPU binding).
    assert "--gpus-per-task" not in cmd
    # The inner Python invocation still carries --num-gpus-per-model
    assert "--num-gpus-per-model" in cmd
    ng_idx = cmd.index("--num-gpus-per-model")
    assert cmd[ng_idx + 1] == "4"
