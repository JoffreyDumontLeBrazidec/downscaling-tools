"""Scoreboard scorer contract test for precip_scores."""
from __future__ import annotations

import json

from eval.evaluators.precip_scores.scorer import score


def test_score_reads_summary_and_emits_baseline_row(tmp_path):
    payload = {"summary": {
        "model_rmse_mm": 1.5, "model_ens_rmse_mm": 1.1, "model_bias_mm": -0.2,
        "model_corr": 0.61, "model_p999_mm": 40.0, "truth_p999_mm": 50.0,
        "model_max_mm": 200.0, "truth_max_mm": 400.0,
        "model_wet_frac": 0.42, "truth_wet_frac": 0.40, "model_neg_frac": 0.06,
        "baseline_rmse_mm": 1.2, "baseline_corr": 0.70,
        "model_over_baseline_rmse_ratio": 1.25,
    }}
    (tmp_path / "scores.json").write_text(json.dumps(payload))
    # eval.cli passes predictions_dir as an extra keyword; the scoreboard
    # aggregator does not — score() must accept both call shapes.
    records = score(tmp_path, {}, {}, predictions_dir="/tmp/x")
    by_metric = {r["metric"]: r for r in records}
    assert by_metric["tp_rmse_mm"]["value"] == 1.5
    assert by_metric["tp_p999_ratio"]["value"] == 0.8
    assert by_metric["tp_max_ratio"]["value"] == 0.5
    assert by_metric["tp_baseline_rmse_mm"]["value"] == 1.2
    assert by_metric["tp_rmse_vs_baseline_ratio"]["value"] == 1.25
    assert all({"metric", "value", "unit"} <= set(r) for r in records)


def test_score_empty_results_dir(tmp_path):
    assert score(tmp_path, {}, {}) == []
