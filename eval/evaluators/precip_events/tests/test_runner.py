"""Smoke tests for precip_events runner argv + merge orchestration."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from eval.evaluators.precip_events import runner
from eval._backends.region_plotting.precip_events import Event


def _fake_events(tmp_path):
    return [
        Event(nc_path=tmp_path / "predictions_20250927_step048.nc", date="20250927",
              step=48, peak_value=0.20, lat=-10.0, lon=30.0,
              bbox=[-22.0, 2.0, 15.0, 45.0], label="event01_20250927_step048"),
        Event(nc_path=tmp_path / "predictions_20250926_step024.nc", date="20250926",
              step=24, peak_value=0.05, lat=20.0, lon=-70.0,
              bbox=[8.0, 32.0, -85.0, -55.0], label="event02_20250926_step024"),
    ]


def test_run_invokes_precip_event_plotter_once(tmp_path):
    predictions_dir = tmp_path / "preds"
    predictions_dir.mkdir()
    (predictions_dir / "predictions_20250926_step024.nc").touch()
    output_dir = tmp_path / "out"
    eval_config = {"n_events": 2, "dlat": 12, "dlon": 15, "rank_by": "truth", "run_label": "fixtureA"}

    with patch("eval.evaluators.precip_events.runner.find_precip_events",
               return_value=_fake_events(predictions_dir)) as mock_find, \
         patch("eval.evaluators.precip_events.runner.subprocess.run") as mock_run, \
         patch("eval.evaluators.precip_events.runner._validate_pdf") as mock_validate:
        mock_run.return_value = MagicMock(returncode=0)
        result = runner.run(predictions_dir, {}, eval_config, output_dir=output_dir)

    assert result == output_dir
    mock_find.assert_called_once()
    assert mock_run.call_count == 1
    first_cmd = mock_run.call_args_list[0].args[0]
    assert "eval._backends.region_plotting.plot_precip_events" in first_cmd
    assert "--predictions-dir" in first_cmd
    assert str(predictions_dir) in first_cmd
    assert "--out" in first_cmd
    assert str(output_dir / "plots" / "precip_events_local.pdf") in first_cmd
    assert "--var" in first_cmd and "tp" in first_cmd
    assert "--n-top" in first_cmd and "2" in first_cmd
    assert "--dlat" in first_cmd and "12" in first_cmd
    assert "--dlon" in first_cmd and "15" in first_cmd
    assert "--rank-by" in first_cmd and "truth" in first_cmd
    assert "--run-label" in first_cmd and "fixtureA" in first_cmd
    assert (output_dir / "events.json").exists()
    mock_validate.assert_called_once_with(output_dir / "plots" / "precip_events_local.pdf")


def test_run_defaults_to_tight_three_event_plots(tmp_path):
    predictions_dir = tmp_path / "preds"
    predictions_dir.mkdir()
    (predictions_dir / "predictions_20250926_step024.nc").touch()
    output_dir = tmp_path / "out"

    with patch("eval.evaluators.precip_events.runner.find_precip_events",
               return_value=_fake_events(predictions_dir)) as mock_find, \
         patch("eval.evaluators.precip_events.runner.subprocess.run") as mock_run, \
         patch("eval.evaluators.precip_events.runner._validate_pdf"):
        mock_run.return_value = MagicMock(returncode=0)
        runner.run(predictions_dir, {}, {}, output_dir=output_dir)

    _, kwargs = mock_find.call_args
    assert kwargs["n_events"] == 3
    assert kwargs["dlat"] == pytest.approx(2.0)
    assert kwargs["dlon"] == pytest.approx(2.5)
    assert kwargs["rank_by"] == "truth"
    cmd = mock_run.call_args_list[0].args[0]
    assert "--n-top" in cmd and "3" in cmd
    assert "--dlat" in cmd and "2" in cmd
    assert "--dlon" in cmd and "2.5" in cmd
    assert "--rank-by" in cmd and "truth" in cmd


def test_run_raises_when_no_predictions(tmp_path):
    predictions_dir = tmp_path / "empty"
    predictions_dir.mkdir()
    with patch("eval.evaluators.precip_events.runner.find_precip_events",
               side_effect=FileNotFoundError("none")):
        with pytest.raises(FileNotFoundError):
            runner.run(predictions_dir, {}, {}, output_dir=tmp_path / "out")
