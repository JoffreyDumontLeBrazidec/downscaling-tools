"""Smoke tests for precip_dist runner argv construction."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from eval.evaluators.precip_dist import runner


def test_run_invokes_tp_histogram_comparison_module(tmp_path):
    predictions_dir = tmp_path / "preds"
    predictions_dir.mkdir()
    (predictions_dir / "predictions_20250926_step024.nc").touch()
    output_dir = tmp_path / "out"
    eval_config = {"run_label": "fixtureA"}

    with patch("eval.evaluators.precip_dist.runner.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        result = runner.run(predictions_dir, {}, eval_config, output_dir=output_dir)

    assert result == output_dir
    args, _ = mock_run.call_args
    cmd = args[0]
    assert "-m" in cmd
    assert "eval._backends.precip.tp_histogram_comparison" in cmd
    assert "--predictions-dir" in cmd
    assert str(predictions_dir) in cmd
    assert "--out-pdf" in cmd
    out_idx = cmd.index("--out-pdf")
    assert cmd[out_idx + 1].endswith("tp_histograms.pdf")
    assert "/plots/" in cmd[out_idx + 1]
    assert "--ensemble-member-index" in cmd and "0" in cmd
    assert "--style" in cmd and "compact" in cmd
    assert "--run-label" in cmd and "fixtureA" in cmd


def test_run_defaults_when_eval_config_empty(tmp_path):
    predictions_dir = tmp_path / "preds"
    predictions_dir.mkdir()
    (predictions_dir / "predictions_20250926_step024.nc").touch()
    output_dir = tmp_path / "out"
    with patch("eval.evaluators.precip_dist.runner.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        runner.run(predictions_dir, {}, {}, output_dir=output_dir)
    args, _ = mock_run.call_args
    cmd = args[0]
    assert "--run-label" not in cmd
    assert "--style" in cmd and "compact" in cmd
    out_idx = cmd.index("--out-pdf")
    assert "/plots/" in cmd[out_idx + 1]


def test_run_raises_when_no_predictions(tmp_path):
    predictions_dir = tmp_path / "preds_empty"
    predictions_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        runner.run(predictions_dir, {}, {}, output_dir=tmp_path / "out")
