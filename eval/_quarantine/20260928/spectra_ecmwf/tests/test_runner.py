from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from eval.config.loader import load_lane
from eval.evaluators.spectra_ecmwf import runner


@pytest.mark.parametrize(
    ("input_grid", "expected"),
    [
        ("O48", 95),
        ("o96", 319),
        ("O320", 1279),
        ("o1280", 2559),
    ],
)
def test_resolve_truncation_from_lane_input_grid(input_grid: str, expected: int) -> None:
    lane_config = {"prepml": {"input": {"grid": input_grid}}}

    assert runner._resolve_truncation(lane_config, {}) == expected


@pytest.mark.parametrize(
    ("lane_name", "expected"),
    [
        ("o48_o96", 95),
        ("o96_o320", 319),
        ("o320_o1280", 1279),
        ("o1280_o2560", 2559),
    ],
)
def test_canonical_lane_declares_nominal_truncation(lane_name: str, expected: int) -> None:
    lane_config = load_lane(lane_name)

    assert lane_config["spectra_ecmwf"]["truncation"] == expected
    assert runner._resolve_truncation(lane_config, lane_config["spectra_ecmwf"]) == expected


def test_resolve_truncation_prefers_explicit_override() -> None:
    lane_config = {"prepml": {"input": {"grid": "O320"}}}

    assert runner._resolve_truncation(lane_config, {"truncation": 511}) == 511


@pytest.mark.parametrize("value", [0, -1, True, 319.5, "T319"])
def test_resolve_truncation_rejects_invalid_override(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        runner._resolve_truncation({}, {"truncation": value})


def test_resolve_truncation_requires_known_lane_or_override() -> None:
    with pytest.raises(ValueError, match="Set spectra_ecmwf.truncation explicitly"):
        runner._resolve_truncation({"prepml": {"input": {"grid": "O640"}}}, {})


def test_run_passes_lane_truncation_to_pipeline(tmp_path: Path) -> None:
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    output_dir = tmp_path / "spectra_ecmwf"
    lane_config = {
        "predict": {"dates": ["20230829"], "steps": [120]},
        "prepml": {"input": {"grid": "O320"}},
    }
    eval_config = {"weather_states": ["2t"], "members": [1]}

    with (
        patch.object(runner.socket, "gethostname", return_value="ac-test"),
        patch.object(runner, "_run_pipeline") as mock_pipeline,
    ):
        result = runner.run(
            predictions_dir,
            lane_config,
            eval_config,
            output_dir=output_dir,
        )

    assert result == output_dir
    assert mock_pipeline.call_args.kwargs["truncation"] == 1279


def test_amplitude_cache_requires_matching_truncation_metadata(tmp_path: Path) -> None:
    amp_dir = tmp_path / "truth" / "spectra"
    state_dir = amp_dir / "2t_sfc"
    state_dir.mkdir(parents=True)
    (state_dir / "ampl_sample.npy").touch()
    summary_path = amp_dir.parent / "spectra_summary.json"

    assert not runner._has_amplitudes(amp_dir, ["2t"], truncation=1279)

    summary_path.write_text(json.dumps({"truncation": 319}), encoding="utf-8")
    assert not runner._has_amplitudes(amp_dir, ["2t"], truncation=1279)

    summary_path.write_text(json.dumps({"truncation": 1279}), encoding="utf-8")
    assert runner._has_amplitudes(amp_dir, ["2t"], truncation=1279)


def test_compute_amplitudes_forwards_truncation(tmp_path: Path) -> None:
    with patch.object(runner.subprocess, "run") as mock_run:
        runner._compute_amplitudes(
            sh_dir=tmp_path / "spectral_harmonics",
            amp_dir=tmp_path / "spectra",
            weather_states="2t",
            summary_path=tmp_path / "spectra_summary.json",
            truncation=1279,
        )

    command = mock_run.call_args.args[0]
    assert command[:2] == ["bash", "-c"]
    assert '--truncation "1279"' in command[2]


def test_amplitude_computer_passes_truncation_to_metview(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_metview = MagicMock()
    fake_metview.Fieldset.return_value = []
    fake_metview.spec_graph.return_value = [
        None,
        {"INPUT_X_VALUES": [1.0, 2.0], "INPUT_Y_VALUES": [3.0, 4.0]},
    ]
    monkeypatch.setitem(sys.modules, "metview", fake_metview)

    module_name = "eval.evaluators.spectra_ecmwf._amplitude_computer"
    sys.modules.pop(module_name, None)
    amplitude_computer = importlib.import_module(module_name)
    try:
        wavenumbers, amplitudes = amplitude_computer.read_curve(
            Path("field_20230829_120_1_nopoles.grb_sh"),
            amplitude_computer.CONFIGS["2t"],
            truncation=1279,
        )
    finally:
        sys.modules.pop(module_name, None)

    assert np.array_equal(wavenumbers, np.array([1.0, 2.0]))
    assert np.array_equal(amplitudes, np.array([3.0, 4.0]))
    assert fake_metview.spec_graph.call_args.kwargs["truncation"] == 1279
