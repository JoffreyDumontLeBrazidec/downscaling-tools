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


def test_amplitude_computer_reads_curves_with_eccodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """read_curve now goes through the harmonics module, not Metview.

    The parameter and level are forwarded so a curve staged into the wrong
    parameter directory is caught rather than silently scored.
    """
    module_name = "eval.evaluators.spectra_ecmwf._amplitude_computer"
    sys.modules.pop(module_name, None)
    amplitude_computer = importlib.import_module(module_name)
    seen: dict = {}

    def fake_amplitude_curve(path, *, truncation, param=None, level=None):
        seen.update(path=path, truncation=truncation, param=param, level=level)
        return np.array([1.0, 2.0]), np.array([3.0, 4.0])

    monkeypatch.setattr(
        amplitude_computer.harmonics, "amplitude_curve", fake_amplitude_curve
    )
    try:
        wavenumbers, amplitudes = amplitude_computer.read_curve(
            Path("field_20230829_120_1_nopoles.grb_sh"),
            amplitude_computer.CONFIGS["t_850"],
            truncation=1279,
        )
    finally:
        sys.modules.pop(module_name, None)

    assert np.array_equal(wavenumbers, np.array([1.0, 2.0]))
    assert np.array_equal(amplitudes, np.array([3.0, 4.0]))
    assert seen["truncation"] == 1279
    assert seen["param"] == "t"
    assert seen["level"] == "850"


def test_spectra_path_no_longer_imports_metview() -> None:
    """The whole point of the eccodes switch: no Metview in this path."""
    source = (
        Path(runner.__file__).parent / "_amplitude_computer.py"
    ).read_text(encoding="utf-8")

    assert "import metview" not in source
    # The prose above the function still names spec_graph to explain the
    # history, so look for a call rather than a mention.
    assert "mv.spec_graph(" not in source
    assert "mv.read(" not in source


def test_stage_three_needs_no_module_block(tmp_path: Path) -> None:
    """Stage 3 runs on the venv alone now, so it loads no modules at all."""
    with patch.object(runner.subprocess, "run") as mock_run:
        runner._compute_amplitudes(
            sh_dir=tmp_path / "sh",
            amp_dir=tmp_path / "amp",
            weather_states="2t",
            summary_path=tmp_path / "s.json",
            truncation=1279,
        )

    script = mock_run.call_args.args[0][2]
    assert "module load" not in script
    assert "METVIEW" not in script


def test_coefficient_walk_matches_the_definition() -> None:
    """power[n] = sum over m<=n of Re^2 + Im^2, with m>0 counted once."""
    from eval._backends.spectra import harmonics

    truncation = 3
    n_coefficients = (truncation + 1) * (truncation + 2) // 2
    rng = np.random.default_rng(0)
    values = rng.normal(size=2 * n_coefficients)

    power = harmonics.power_from_coefficients(values, truncation)

    coefficients = values.reshape(n_coefficients, 2)
    expected = np.zeros(truncation + 1)
    index = 0
    for m in range(truncation + 1):
        for n in range(m, truncation + 1):
            expected[n] += coefficients[index, 0] ** 2 + coefficients[index, 1] ** 2
            index += 1
    assert np.allclose(power, expected)


def test_coefficient_walk_rejects_a_wrong_sized_array() -> None:
    from eval._backends.spectra import harmonics

    with pytest.raises(ValueError, match="expected 20 coefficient values for T3"):
        harmonics.power_from_coefficients(np.zeros(19), 3)


def test_run_gptosp_passes_explicit_truncation(tmp_path: Path) -> None:
    """Stage 2 must be told the truncation, not left to derive its own.

    Without -T, gptosp derives the truncation from the staged grid's latitude
    count, which on a pole-masked grid disagrees with what stage 3 is told.
    """
    with (
        patch.object(runner.subprocess, "run") as mock_run,
        patch.object(runner, "_verify_truncation") as mock_verify,
    ):
        runner._run_gptosp(
            grb_dir=tmp_path / "grb",
            sh_dir=tmp_path / "spectral_harmonics",
            weather_states=["2t"],
            truncation=1279,
        )

    command = mock_run.call_args.args[0]
    assert command[:3] == ["bash", "--login", "-c"]
    script = command[3] if len(command) > 3 else command[2]
    assert "gptosp.ser -T 1279 -g" in script
    assert "gptosp.ser -l" not in script
    assert mock_verify.call_args.kwargs["truncation"] == 1279


def test_verify_truncation_accepts_matching_files(tmp_path: Path) -> None:
    sh_dir = tmp_path / "spectral_harmonics"
    (sh_dir / "2t_sfc").mkdir(parents=True)
    (sh_dir / "2t_sfc" / "1_20230826_120_1_nopoles.grb_sh").touch()

    with patch.object(runner, "_read_achieved_truncation", return_value=1279):
        runner._verify_truncation(sh_dir, ["2t_sfc"], truncation=1279)


def test_verify_truncation_rejects_mismatch(tmp_path: Path) -> None:
    """A stage-2/stage-3 disagreement is a wrong number, so it must stop."""
    sh_dir = tmp_path / "spectral_harmonics"
    (sh_dir / "2t_sfc").mkdir(parents=True)
    (sh_dir / "2t_sfc" / "1_20230826_120_1_nopoles.grb_sh").touch()

    with patch.object(runner, "_read_achieved_truncation", return_value=2531):
        with pytest.raises(RuntimeError, match="requested T1279, file carries T2531"):
            runner._verify_truncation(sh_dir, ["2t_sfc"], truncation=1279)


def test_verify_truncation_rejects_empty_output(tmp_path: Path) -> None:
    """An empty stage 2 must be reported here, not three stages later."""
    sh_dir = tmp_path / "spectral_harmonics"
    (sh_dir / "2t_sfc").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="produced no harmonics"):
        runner._verify_truncation(sh_dir, ["2t_sfc"], truncation=1279)


def test_pipeline_uses_one_truncation_for_both_stages(tmp_path: Path) -> None:
    """The gptosp argument and the amplitude argument are the same integer."""
    with (
        patch.object(runner, "_stage_gribs"),
        patch.object(runner, "_run_gptosp") as mock_gptosp,
        patch.object(runner, "_compute_amplitudes") as mock_amplitudes,
    ):
        runner._run_pipeline(
            label="truth",
            predictions_dir=tmp_path / "predictions",
            output_dir=tmp_path / "out",
            prediction_var="y",
            weather_states=["2t"],
            weather_states_str="2t",
            template_root="",
            template_grib_root="",
            date_list="20230826",
            step_list="120",
            truncation=1279,
        )

    assert mock_gptosp.call_args.kwargs["truncation"] == 1279
    assert mock_amplitudes.call_args.kwargs["truncation"] == 1279
    assert (
        mock_gptosp.call_args.kwargs["truncation"]
        == mock_amplitudes.call_args.kwargs["truncation"]
    )


# --- reference cache: window addressing and validation -----------------------


def _make_cache(root: Path, *, truncation: int | None, files: list[dict] | None) -> Path:
    """Build a minimal reference cache directory and return its spectra dir."""
    amp_dir = root / "spectra"
    (amp_dir / "2t_sfc").mkdir(parents=True)
    (amp_dir / "2t_sfc" / "ampl_20230826_120_2t_n1.npy").touch()
    summary: dict = {}
    if truncation is not None:
        summary["truncation"] = truncation
    if files is not None:
        summary["files"] = files
    (root / "spectra_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return amp_dir


def _entries(dates: list[str], step: int = 120, member: int = 1) -> list[dict]:
    return [
        {"date": int(d), "step_hours": step, "member": member, "weather_state": "2t"}
        for d in dates
    ]


def test_cache_is_valid_for_the_window_it_was_computed_from(tmp_path: Path) -> None:
    amp = _make_cache(tmp_path, truncation=1279, files=_entries(["20230826", "20230827"]))

    assert (
        runner._validated_cache(
            amp, ["2t"], truncation=1279, dates=["20230826"], steps=[120], members=[1]
        )
        == runner.CACHE_VALID
    )


def test_cache_is_stale_for_a_different_month(tmp_path: Path) -> None:
    """The bug this replaces: an Idalia-2023 cache scored a 2025 evaluation."""
    amp = _make_cache(tmp_path, truncation=1279, files=_entries(["20230826", "20230827"]))

    assert (
        runner._validated_cache(
            amp, ["2t"], truncation=1279, dates=["20250901"], steps=[120], members=[1]
        )
        == runner.CACHE_STALE
    )


def test_cache_is_stale_for_an_uncovered_step_or_member(tmp_path: Path) -> None:
    amp = _make_cache(tmp_path, truncation=1279, files=_entries(["20230826"]))

    assert (
        runner._validated_cache(
            amp, ["2t"], truncation=1279, dates=["20230826"], steps=[240], members=[1]
        )
        == runner.CACHE_STALE
    )
    assert (
        runner._validated_cache(
            amp, ["2t"], truncation=1279, dates=["20230826"], steps=[120], members=[7]
        )
        == runner.CACHE_STALE
    )


def test_cache_without_date_metadata_is_never_valid(tmp_path: Path) -> None:
    """An unverifiable cache must not be presented as truth, even at the right T."""
    amp = _make_cache(tmp_path, truncation=1279, files=None)

    assert (
        runner._validated_cache(
            amp, ["2t"], truncation=1279, dates=["20230826"], steps=[120], members=[1]
        )
        == runner.CACHE_UNVERIFIABLE
    )


def test_cache_without_truncation_metadata_is_unverifiable(tmp_path: Path) -> None:
    amp = _make_cache(tmp_path, truncation=None, files=_entries(["20230826"]))

    assert (
        runner._validated_cache(
            amp, ["2t"], truncation=1279, dates=["20230826"], steps=[120], members=[1]
        )
        == runner.CACHE_UNVERIFIABLE
    )


def test_unknown_window_falls_back_to_the_truncation_only_check(tmp_path: Path) -> None:
    amp = _make_cache(tmp_path, truncation=1279, files=None)

    assert runner._validated_cache(amp, ["2t"], truncation=1279) == runner.CACHE_VALID
    assert runner._has_amplitudes(amp, ["2t"], truncation=1279)
    assert not runner._has_amplitudes(amp, ["2t"], truncation=319)


def test_window_key_is_stable_and_order_insensitive() -> None:
    a = runner._window_key(
        dates=["20230826", "20230830"], steps=[120], members=[1], truncation=1279
    )
    b = runner._window_key(
        dates=["20230830", "20230826"], steps=[120], members=[1], truncation=1279
    )

    assert a == b
    assert a.startswith("d20230826-20230830_s120_m1_T1279_")


@pytest.mark.parametrize(
    "changed",
    [
        {"dates": ["20250901", "20250902"]},
        {"steps": [240]},
        {"members": [2]},
        {"truncation": 2559},
    ],
)
def test_window_key_separates_different_windows(changed: dict) -> None:
    """Two windows must never share a directory, or they end up mixed."""
    base = {
        "dates": ["20230826", "20230830"],
        "steps": [120],
        "members": [1],
        "truncation": 1279,
    }

    assert runner._window_key(**base) != runner._window_key(**{**base, **changed})


def test_unspecified_members_are_marked_in_the_key() -> None:
    key = runner._window_key(dates=["20230826"], steps=[120], members=[], truncation=1279)

    assert "_mALL_" in key


# --- plotter: finding the reference now that references are window-addressed ---


def test_plotter_uses_the_recorded_reference_directory(tmp_path: Path) -> None:
    """References moved under a window key, which the plotter cannot reconstruct.

    It previously built reference_dir/truth/spectra, which stopped existing, so
    the PDF silently lost its truth and input curves.
    """
    from eval.evaluators.spectra_ecmwf import plotter

    results = tmp_path / "run"
    (results / "spectra" / "2t_sfc").mkdir(parents=True)
    recorded = tmp_path / "ref" / "truth" / "dWINDOW_T1279_abcd1234" / "spectra"
    (results / "spectra_summary.json").write_text(
        json.dumps({"reference_spectra_dir": str(recorded)}), encoding="utf-8"
    )

    with patch.object(plotter, "__name__", plotter.__name__):
        with patch(
            "eval.evaluators.spectra_ecmwf._plotter.build_pdf_ecmwf_with_references"
        ) as mock_build:
            mock_build.return_value = 1
            plotter.plot(results, {}, {"reference_dir": str(tmp_path / "ref")})

    kwargs = mock_build.call_args.kwargs
    assert kwargs["truth_amp_dir"] == recorded
    # input sits beside truth under the same window key
    assert kwargs["input_amp_dir"] == Path(str(recorded).replace("/truth/", "/input/", 1))


def test_plotter_falls_back_to_the_old_layout(tmp_path: Path) -> None:
    """A summary written before the window key still has to plot."""
    from eval.evaluators.spectra_ecmwf import plotter

    results = tmp_path / "run"
    (results / "spectra" / "2t_sfc").mkdir(parents=True)
    (results / "spectra_summary.json").write_text(json.dumps({}), encoding="utf-8")
    ref = tmp_path / "ref"

    with patch(
        "eval.evaluators.spectra_ecmwf._plotter.build_pdf_ecmwf_with_references"
    ) as mock_build:
        mock_build.return_value = 1
        plotter.plot(results, {}, {"reference_dir": str(ref)})

    kwargs = mock_build.call_args.kwargs
    assert kwargs["truth_amp_dir"] == ref / "truth" / "spectra"
    assert kwargs["input_amp_dir"] == ref / "input" / "spectra"
