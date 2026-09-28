"""The spectra_ecmwf_v2 scorer: relative L2 of mean curves above wavenumber 100,
under spectra_v2_* metric names."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval._backends.spectra import naming
from eval.evaluators.spectra_ecmwf_v2 import scorer

FIELD_DIRS = {"10u": "10u_sfc", "10v": "10v_sfc", "2t": "2t_sfc", "msl": "msl_sfc"}


def _power_law(n: int) -> tuple[np.ndarray, np.ndarray]:
    wvn = np.arange(n, dtype=np.float64)
    return wvn, (np.maximum(wvn, 1.0) ** -3.0) * 1e6


def _write_run(tmp_path: Path, factors: dict[str, float], *, n: int = 320,
               members=(1, 2), low_k_factor: float = 1.0) -> Path:
    """A results dir plus truth reference; prediction = truth * factor above k=100."""
    results = tmp_path / "evaluators" / "spectra_ecmwf_v2"
    truth_root = tmp_path / "ref" / "truth" / "dWIN_T319_abcd1234" / "spectra"
    files = []
    wvn, truth = _power_law(n)
    for state, factor in factors.items():
        fd = FIELD_DIRS[state]
        for member in members:
            key = dict(date=20230826, step=120, field_dir=fd, token="1", member=member)
            pred = truth.copy()
            pred[wvn > 100] *= factor
            pred[wvn <= 100] *= low_k_factor
            for root, arr in ((results / "spectra" / fd, pred), (truth_root / fd, truth)):
                root.mkdir(parents=True, exist_ok=True)
                np.save(root / naming.canonical_name("ampl", **key), arr)
                np.save(root / naming.canonical_name("wvn", **key), wvn)
            files.append({
                "weather_state": state,
                "amplitudes": str(results / "spectra" / fd / naming.canonical_name("ampl", **key)),
                "wavenumbers": str(results / "spectra" / fd / naming.canonical_name("wvn", **key)),
                "date": "20230826", "step_hours": 120, "member": member,
            })
    (results / "spectra_summary.json").write_text(json.dumps({
        "truncation": n - 1, "reference_spectra_dir": str(truth_root), "files": files,
    }))
    return results


def test_score_curve_pair_is_relative_l2_above_100():
    wvn, truth = _power_law(320)
    pred = truth.copy()
    pred[wvn > 100] *= 1.25
    pred[wvn <= 100] *= 5.0  # large-scale error is outside the scored band
    assert scorer.score_curve_pair(pred, truth, wvn) == pytest.approx(0.25)


def test_score_emits_v2_rows_per_variable_and_mean(tmp_path):
    results = _write_run(tmp_path, {"10u": 0.9, "10v": 1.2, "2t": 1.0}, low_k_factor=3.0)
    rows = {r["metric"]: r for r in scorer.score(results, {}, {})}
    assert rows["spectra_v2_10u_relative_l2"]["value"] == pytest.approx(0.1)
    assert rows["spectra_v2_10u_score"]["value"] == pytest.approx(0.9)
    assert rows["spectra_v2_10v_relative_l2"]["value"] == pytest.approx(0.2)
    assert rows["spectra_v2_2t_relative_l2"]["value"] == pytest.approx(0.0)
    assert rows["spectra_v2_2t_score"]["value"] == pytest.approx(1.0)
    assert rows["spectra_v2_mean_relative_l2"]["value"] == pytest.approx(0.1)
    assert rows["spectra_v2_mean_score"]["value"] == pytest.approx(0.9)
    assert rows["spectra_v2_mean_score"]["unit"] == "score_0_1"
    # never under the proxy's names
    assert not any(m.startswith("spectra_") and not m.startswith("spectra_v2_") for m in rows)
    details = json.loads((results / "spectra_v2_scores.json").read_text())
    assert details["variables"]["10u"]["n_pairs"] == 2
    assert details["variables"]["10u"]["wavenumber_min_exclusive"] == 100.0


def test_score_is_clamped_at_zero(tmp_path):
    results = _write_run(tmp_path, {"10u": 3.0, "10v": 1.0, "2t": 1.0})
    rows = {r["metric"]: r["value"] for r in scorer.score(results, {}, {})}
    assert rows["spectra_v2_10u_relative_l2"] == pytest.approx(2.0)
    assert rows["spectra_v2_10u_score"] == 0.0


def test_no_mean_row_below_three_variables(tmp_path):
    results = _write_run(tmp_path, {"10u": 0.9, "msl": 1.1})
    metrics = {r["metric"] for r in scorer.score(results, {}, {})}
    assert "spectra_v2_msl_score" in metrics
    assert "spectra_v2_mean_score" not in metrics


def test_low_truncation_scores_the_top_two_thirds(tmp_path):
    # O96 is analysed at T95: nothing lies above 100, so the band starts at 95/3.
    results = _write_run(tmp_path, {"10u": 1.0, "10v": 1.0, "2t": 1.0}, n=96)
    rows = {r["metric"]: r["value"] for r in scorer.score(results, {}, {})}
    assert rows["spectra_v2_mean_relative_l2"] == pytest.approx(0.0)
    details = json.loads((results / "spectra_v2_scores.json").read_text())
    assert details["variables"]["2t"]["wavenumber_min_exclusive"] == pytest.approx(95 / 3)


def test_no_truth_reference_gives_no_rows(tmp_path):
    results = _write_run(tmp_path, {"10u": 0.9, "10v": 1.0, "2t": 1.0})
    summary = json.loads((results / "spectra_summary.json").read_text())
    summary["reference_spectra_dir"] = ""
    (results / "spectra_summary.json").write_text(json.dumps(summary))
    assert scorer.score(results, {}, {}) == []


def test_aggregator_reads_v2_rows_under_their_own_evaluator(tmp_path):
    from eval.scoreboard.aggregator import aggregate_scores

    _write_run(tmp_path, {"10u": 0.9, "10v": 1.2, "2t": 1.0})
    records = aggregate_scores(tmp_path, {}, evaluators=["spectra_ecmwf_v2", "spectra"])
    assert {r.evaluator for r in records} == {"spectra_ecmwf_v2"}
    assert "spectra_v2_mean_score" in {r.metric for r in records}
