"""Tests for the TC raw-extremes scoreboard loader.

Per the run-trust contract (epics/run-trust-and-validation/tc-extremes-contract-decision.md,
decided 2026-06-21), TC quality is reported as RAW extremes only — model / OPER / ENFO side
by side on the same grid, with no score, ratio, anchor or curated-AN. These tests pin the
loader's extraction behavior; the retired depth/score/anchor functions are gone.
"""

from __future__ import annotations

import json

import pytest

from eval._backends.scoreboard.tc import load_tc_extreme_scores_from_json


def _write_stats(tmp_path, events):
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps({"events": events}))
    return stats_path


def test_emits_raw_extremes_model_oper_enfo(tmp_path):
    """model / OPER / ENFO each emit the four raw extremes from the same stats JSON."""
    stats_path = _write_stats(tmp_path, {
        "idalia": {
            "extreme_tail": {
                "rows": [
                    {"exp": "OPER_O320_0001", "mslp_min": 985.4, "mslp_p001": 993.5,
                     "wind_max": 24.5, "wind_p9999": 19.7},
                    {"exp": "ENFO_O320_0001", "mslp_min": 969.1, "mslp_p001": 993.7,
                     "wind_max": 42.9, "wind_p9999": 18.1},
                    {"exp": "manual_deadbeef_new_o96_o320", "mslp_min": 976.3, "mslp_p001": 995.3,
                     "wind_max": 34.9, "wind_p9999": 17.6},
                ]
            }
        }
    })
    result = load_tc_extreme_scores_from_json(
        stats_path, run_id="manual_deadbeef_new_o96_o320", event_names=("idalia",),
    )
    # model (bare keys)
    assert result["idalia_mslp_min"] == pytest.approx(976.3)
    assert result["idalia_mslp_p001"] == pytest.approx(995.3)
    assert result["idalia_wind_max"] == pytest.approx(34.9)
    assert result["idalia_wind_p9999"] == pytest.approx(17.6)
    # OPER baseline
    assert result["idalia_oper_mslp_min"] == pytest.approx(985.4)
    assert result["idalia_oper_wind_max"] == pytest.approx(24.5)
    # ENFO reference (strongest tails)
    assert result["idalia_enfo_mslp_min"] == pytest.approx(969.1)
    assert result["idalia_enfo_wind_p9999"] == pytest.approx(18.1)
    # No score / ratio / anchor keys under the raw contract.
    assert "idalia" not in result
    assert not any(k.endswith("_ratio") or "extreme_score" in k or k.endswith("_dev")
                   for k in result)


def test_missing_source_is_omitted(tmp_path):
    """A source absent from the rows simply doesn't appear (no crash, no fabricated value)."""
    stats_path = _write_stats(tmp_path, {
        "humberto": {
            "extreme_tail": {
                "rows": [
                    {"exp": "manual_95a07500_new_o48_o96", "mslp_min": 945.6, "wind_max": 47.3},
                ]
            }
        }
    })
    result = load_tc_extreme_scores_from_json(
        stats_path, run_id="manual_95a07500_new_o48_o96", event_names=("humberto",),
    )
    assert result["humberto_mslp_min"] == pytest.approx(945.6)
    assert result["humberto_wind_max"] == pytest.approx(47.3)
    # No OPER/ENFO rows present -> no oper_/enfo_ keys.
    assert not any(k.startswith("humberto_oper_") or k.startswith("humberto_enfo_")
                   for k in result)
    # p0.1 / p99.9 absent in the row -> not emitted.
    assert "humberto_mslp_p001" not in result


def test_unknown_event_yields_empty(tmp_path):
    stats_path = _write_stats(tmp_path, {"idalia": {"extreme_tail": {"rows": []}}})
    result = load_tc_extreme_scores_from_json(
        stats_path, run_id="whatever", event_names=("nonexistent",),
    )
    assert result == {}


def test_legacy_kwargs_are_ignored(tmp_path):
    """Old anchor kwargs are accepted (back-compat) but do not affect the raw output."""
    stats_path = _write_stats(tmp_path, {
        "idalia": {"extreme_tail": {"rows": [
            {"exp": "manual_x_new_o96_o320", "mslp_min": 970.0, "wind_max": 40.0},
        ]}}
    })
    result = load_tc_extreme_scores_from_json(
        stats_path, run_id="manual_x_new_o96_o320", event_names=("idalia",),
        canonical_analysis_by_event={"idalia": {"mslp_min": 960.0}},
        extreme_reference_expid="ENFO_O320_0001",
    )
    assert result == {"idalia_mslp_min": pytest.approx(970.0),
                      "idalia_wind_max": pytest.approx(40.0)}
