"""Tests of the anchored TC scores, moved out of eval/jobs/tests/test_scoreboard_metrics.py.

The TC scorer changed on 2026-06-21 to the raw-extremes contract: it emits the four
raw extremes per source (model, OPER, ENFO, EEFO) and no anchored score, no reach,
no ENFO-match and no tail ratio. These tests exercise the behaviour that was
removed, so they fail against the current code and are kept here, not collected
(pytest.ini norecursedirs), instead of being deleted. The live replacements are
in eval/jobs/tests/test_scoreboard_metrics.py.
"""
from __future__ import annotations

import json

import pytest

from eval.jobs import scoreboard_metrics as metrics


def test_load_tc_extreme_scores_analysis_anchored(tmp_path):
    """When analysis (OPER) and tail percentiles are present, use analysis-anchored scoring."""
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "events": {
                    "idalia": {
                        "extreme_tail": {
                            "rows": [
                                {
                                    "exp": "OPER_O320_0001",
                                    "mslp_p1": 970.0, "mslp_p01": 965.0, "mslp_min": 960.0,
                                    "wind_p99": 30.0, "wind_p999": 35.0, "wind_max": 40.0,
                                },
                                {
                                    "exp": "ENFO_O320_0001",
                                    "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_min": 970.0,
                                    "wind_p99": 25.0, "wind_p999": 28.0, "wind_max": 32.0,
                                },
                                {
                                    "exp": "manual_0c446b41_new_o96_o320",
                                    "mslp_p1": 975.0, "mslp_p01": 970.0, "mslp_min": 965.0,
                                    "wind_p99": 28.0, "wind_p999": 32.0, "wind_max": 36.0,
                                },
                            ]
                        }
                    },
                }
            }
        )
    )

    result = metrics.load_tc_extreme_scores_from_json(stats_path, run_id="manual_0c446b41_new_o96_o320")

    # Model is between analysis and ENFO — should score > 0.5 but < 1.0
    assert 0.5 < result["idalia"] < 1.0
    # ENFO deviation should be present
    assert "idalia_enfo_dev" in result
    assert result["idalia_enfo_dev"] > 0.0


def test_load_tc_extreme_scores_legacy_fallback(tmp_path):
    """When no OPER row or tail percentiles, fall back to batch-relative normalization."""
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "events": {
                    "idalia": {
                        "extreme_tail": {
                            "rows": [
                                {"exp": "ENFO_O320", "mslp_980_990_fraction": 0.1, "wind_gt_25_fraction": 0.1},
                                {"exp": "manual_0c446b41_new_o96_o320", "mslp_980_990_fraction": 0.8, "wind_gt_25_fraction": 0.6},
                            ]
                        }
                    },
                    "franklin": {
                        "extreme_tail": {
                            "rows": [
                                {"exp": "ENFO_O320", "mslp_980_990_fraction": 0.2, "wind_gt_25_fraction": 0.2},
                                {"exp": "manual_0c446b41_new_o96_o320", "mslp_980_990_fraction": 0.5, "wind_gt_25_fraction": 0.8},
                            ]
                        }
                    },
                }
            }
        )
    )

    result = metrics.load_tc_extreme_scores_from_json(stats_path, run_id="manual_0c446b41_new_o96_o320")

    assert result["idalia"] == pytest.approx(1.0)
    assert result["franklin"] == pytest.approx(1.0)


def test_load_tc_extreme_scores_supports_custom_event_names(tmp_path):
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "events": {
                    "humberto": {
                        "extreme_tail": {
                            "rows": [
                                {"exp": "ENFO_O320", "mslp_980_990_fraction": 0.1, "wind_gt_25_fraction": 0.1},
                                {"exp": "manual_95a07500_new_o48_o96", "mslp_980_990_fraction": 0.9, "wind_gt_25_fraction": 0.8},
                            ]
                        }
                    }
                }
            }
        )
    )

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_95a07500_new_o48_o96",
        event_names=("humberto",),
    )

    assert result == {"humberto": pytest.approx(1.0)}


def _enfo_match_stats(model_keys: dict[str, float]) -> dict:
    """Build a minimal TC stats JSON payload for the ENFO match tests."""
    enfo_keys = {
        "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_min": 970.0,
        "wind_p99": 25.0, "wind_p999": 28.0, "wind_max": 32.0,
    }
    analysis_keys = {
        "mslp_p1": 985.0, "mslp_p01": 980.0, "mslp_min": 975.0,
        "wind_p99": 22.0, "wind_p999": 25.0, "wind_max": 28.0,
    }
    return {
        "events": {
            "idalia": {
                "extreme_tail": {
                    "rows": [
                        {"exp": "OPER_O320_0001", **analysis_keys},
                        {"exp": "ENFO_O320_0001", **enfo_keys},
                        {"exp": "manual_abc12345_new_o96_o320", **model_keys},
                    ]
                }
            },
        }
    }


def test_enfo_match_perfect(tmp_path):
    """Model row identical to ENFO row across all 6 keys → enfo_match == 1.0."""
    enfo_keys = {
        "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_min": 970.0,
        "wind_p99": 25.0, "wind_p999": 28.0, "wind_max": 32.0,
    }
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(_enfo_match_stats(enfo_keys)))

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_abc12345_new_o96_o320",
        canonical_analysis_by_event={},
        canonical_eefo_by_event={},
        extreme_reference_expid="ENFO_O320_0001",
    )

    assert result["idalia_enfo_match"] == pytest.approx(1.0)


def test_enfo_match_undershoot(tmp_path):
    """Model wind = 0.5x ENFO, MSLP equal → score = 0.75 (mean of group means)."""
    # Wind ratios all 0.5; MSLP keys all match (depth ratio = 1.0).
    model_keys = {
        "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_min": 970.0,
        "wind_p99": 12.5, "wind_p999": 14.0, "wind_max": 16.0,
    }
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(_enfo_match_stats(model_keys)))

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_abc12345_new_o96_o320",
        canonical_analysis_by_event={},
        canonical_eefo_by_event={},
        extreme_reference_expid="ENFO_O320_0001",
    )

    # Wind group: each ratio 0.5 → score 0.5; mean 0.5
    # MSLP group: each ratio 1.0 → score 1.0; mean 1.0
    # Combined: (0.5 + 1.0) / 2 = 0.75
    assert result["idalia_enfo_match"] == pytest.approx(0.75)


def test_enfo_match_overshoot_symmetric(tmp_path):
    """Model wind = 1.5x ENFO, MSLP equal → score = 0.75 (symmetric to undershoot)."""
    model_keys = {
        "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_min": 970.0,
        "wind_p99": 37.5, "wind_p999": 42.0, "wind_max": 48.0,
    }
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(_enfo_match_stats(model_keys)))

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_abc12345_new_o96_o320",
        canonical_analysis_by_event={},
        canonical_eefo_by_event={},
        extreme_reference_expid="ENFO_O320_0001",
    )

    # Wind group: each ratio 1.5 → score 0.5; mean 0.5
    # MSLP group: each ratio 1.0 → score 1.0; mean 1.0
    # Combined: (0.5 + 1.0) / 2 = 0.75 — same as undershoot case (symmetric)
    assert result["idalia_enfo_match"] == pytest.approx(0.75)


def test_enfo_match_absent_when_extreme_reference_expid_not_passed(tmp_path):
    """When extreme_reference_expid is None, no enfo_match keys appear in the result."""
    enfo_keys = {
        "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_min": 970.0,
        "wind_p99": 25.0, "wind_p999": 28.0, "wind_max": 32.0,
    }
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(_enfo_match_stats(enfo_keys)))

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_abc12345_new_o96_o320",
    )

    assert "idalia_enfo_match" not in result
    assert "idalia_mslp_reach" not in result
    assert "idalia_wind_reach" not in result
    # Existing scores still computed
    assert "idalia" in result
    assert "idalia_enfo_dev" in result


def test_reach_at_an(tmp_path):
    """Model values equal to AN exactly → reach == 0.0."""
    an_keys = {
        "mslp_p1": 985.0, "mslp_p01": 980.0, "mslp_min": 975.0,
        "wind_p99": 22.0, "wind_p999": 25.0, "wind_max": 28.0,
    }
    # Override the synthesized model row with the AN values.
    payload = _enfo_match_stats(an_keys)
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(payload))

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_abc12345_new_o96_o320",
        canonical_analysis_by_event={},
        canonical_eefo_by_event={},
        extreme_reference_expid="ENFO_O320_0001",
    )

    assert result["idalia_mslp_reach"] == pytest.approx(0.0)
    assert result["idalia_wind_reach"] == pytest.approx(0.0)


def test_reach_at_enfo(tmp_path):
    """Model values equal to ENFO exactly → reach == 1.0."""
    enfo_keys = {
        "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_min": 970.0,
        "wind_p99": 25.0, "wind_p999": 28.0, "wind_max": 32.0,
    }
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(_enfo_match_stats(enfo_keys)))

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_abc12345_new_o96_o320",
        canonical_analysis_by_event={},
        canonical_eefo_by_event={},
        extreme_reference_expid="ENFO_O320_0001",
    )

    assert result["idalia_mslp_reach"] == pytest.approx(1.0)
    assert result["idalia_wind_reach"] == pytest.approx(1.0)


def test_reach_below_an(tmp_path):
    """Model less extreme than AN (shallower MSLP, lower wind) → reach < 0."""
    # AN mslp_min depth = 1013.25 - 975 = 38.25; AN wind_max = 28.0
    # Model mslp_min = 990 (depth = 23.25, less extreme than AN);
    #       wind_max = 20 (less extreme than AN).
    model_keys = {
        "mslp_p1": 1000.0, "mslp_p01": 995.0, "mslp_min": 990.0,
        "wind_p99": 18.0, "wind_p999": 19.0, "wind_max": 20.0,
    }
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(_enfo_match_stats(model_keys)))

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_abc12345_new_o96_o320",
        canonical_analysis_by_event={},
        canonical_eefo_by_event={},
        extreme_reference_expid="ENFO_O320_0001",
    )

    assert result["idalia_mslp_reach"] < 0
    assert result["idalia_wind_reach"] < 0


def test_reach_skips_inverted_axis_when_an_more_extreme_than_enfo(tmp_path):
    """When AN's extreme exceeds ENFO's at a key (e.g. small-sample p999 artefact),
    the AN→ENFO interpolation is undefined and that key must be skipped — not flipped."""
    # AN row has wind_max=40 (very deep tail from small sample); ENFO has wind_max=30.
    # ENFO − AN = -10 (negative) → key must be skipped.
    # Both mslp keys keep their normal ordering (ENFO more extreme than AN).
    payload = {
        "events": {
            "idalia": {
                "extreme_tail": {
                    "rows": [
                        {
                            "exp": "OPER_O320_0001",
                            "mslp_p1": 985.0, "mslp_p01": 980.0, "mslp_min": 975.0,
                            "wind_p99": 22.0, "wind_p999": 38.0, "wind_max": 40.0,
                        },
                        {
                            "exp": "ENFO_O320_0001",
                            "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_min": 970.0,
                            "wind_p99": 25.0, "wind_p999": 28.0, "wind_max": 30.0,
                        },
                        {
                            "exp": "manual_abc12345_new_o96_o320",
                            "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_min": 970.0,
                            "wind_p99": 25.0, "wind_p999": 28.0, "wind_max": 30.0,
                        },
                    ]
                }
            },
        }
    }
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(payload))

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_abc12345_new_o96_o320",
        canonical_analysis_by_event={},
        canonical_eefo_by_event={},
        extreme_reference_expid="ENFO_O320_0001",
    )

    # MSLP keys still valid → mslp_reach computed.
    assert result["idalia_mslp_reach"] == pytest.approx(1.0)
    # Wind keys both have inverted axis → wind_reach is None (key absent from result).
    assert "idalia_wind_reach" not in result


def test_reach_degenerate_when_an_equals_enfo(tmp_path):
    """When AN and ENFO have identical extremes, both reaches return None (all keys skipped)."""
    same_keys = {
        "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_min": 970.0,
        "wind_p99": 25.0, "wind_p999": 28.0, "wind_max": 32.0,
    }
    payload = {
        "events": {
            "idalia": {
                "extreme_tail": {
                    "rows": [
                        # AN row uses same extremes as ENFO → denominator collapses.
                        {"exp": "OPER_O320_0001", **same_keys},
                        {"exp": "ENFO_O320_0001", **same_keys},
                        {"exp": "manual_abc12345_new_o96_o320", **same_keys},
                    ]
                }
            },
        }
    }
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(payload))

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_abc12345_new_o96_o320",
        canonical_analysis_by_event={},
        canonical_eefo_by_event={},
        extreme_reference_expid="ENFO_O320_0001",
    )

    assert "idalia_mslp_reach" not in result
    assert "idalia_wind_reach" not in result


def _ratio_stats(model_keys: dict[str, float]) -> dict:
    """Stats payload with mslp_p001 + wind_p9999 fields populated for ratio tests."""
    enfo_keys = {
        "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_p001": 970.0, "mslp_min": 965.0,
        "wind_p99": 25.0, "wind_p999": 28.0, "wind_p9999": 32.0, "wind_max": 38.0,
    }
    analysis_keys = {
        "mslp_p1": 985.0, "mslp_p01": 980.0, "mslp_p001": 975.0, "mslp_min": 972.0,
        "wind_p99": 22.0, "wind_p999": 25.0, "wind_p9999": 28.0, "wind_max": 32.0,
    }
    return {
        "events": {
            "idalia": {
                "extreme_tail": {
                    "rows": [
                        {"exp": "OPER_O320_0001", **analysis_keys},
                        {"exp": "ENFO_O320_0001", **enfo_keys},
                        {"exp": "manual_abc12345_new_o96_o320", **model_keys},
                    ]
                }
            },
        }
    }


def test_tail_ratios_an_row_equals_one(tmp_path):
    """AN row's tail-extreme ratios should always be 1.0 by construction."""
    enfo_keys = {
        "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_p001": 970.0, "mslp_min": 965.0,
        "wind_p99": 25.0, "wind_p999": 28.0, "wind_p9999": 32.0, "wind_max": 38.0,
    }
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(_ratio_stats(enfo_keys)))

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="OPER_O320_0001",
        canonical_analysis_by_event={},
        canonical_eefo_by_event={},
    )

    assert result["idalia_mslp_p001_ratio"] == pytest.approx(1.0)
    assert result["idalia_mslp_min_ratio"] == pytest.approx(1.0)
    assert result["idalia_wind_p9999_ratio"] == pytest.approx(1.0)
    assert result["idalia_wind_max_ratio"] == pytest.approx(1.0)


def test_tail_ratios_model_more_extreme_than_an(tmp_path):
    """Model row with deeper minima / stronger winds than AN yields ratios > 1."""
    # AN mslp_p001=975 → depth = 1013.25−975 = 38.25
    # Model mslp_p001=970 → depth = 43.25 → ratio = 43.25/38.25 ≈ 1.131
    # AN wind_max=32; model wind_max=38 → ratio = 38/32 = 1.1875
    model_keys = {
        "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_p001": 970.0, "mslp_min": 965.0,
        "wind_p99": 25.0, "wind_p999": 28.0, "wind_p9999": 32.0, "wind_max": 38.0,
    }
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(_ratio_stats(model_keys)))

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_abc12345_new_o96_o320",
        canonical_analysis_by_event={},
        canonical_eefo_by_event={},
    )

    # Model = ENFO row values (which exceed AN at all keys).
    assert result["idalia_mslp_p001_ratio"] > 1.0
    assert result["idalia_mslp_min_ratio"] > 1.0
    assert result["idalia_wind_p9999_ratio"] > 1.0
    assert result["idalia_wind_max_ratio"] > 1.0


def test_tail_ratios_model_less_extreme(tmp_path):
    """Model row with shallower minima / weaker winds than AN yields ratios < 1."""
    # AN: mslp_p001=975 (depth 38.25), wind_max=32
    # Model values are at AN's wind_p99 and mslp_p01 levels — less extreme than AN at p001/max.
    model_keys = {
        "mslp_p1": 1000.0, "mslp_p01": 995.0, "mslp_p001": 992.0, "mslp_min": 990.0,
        "wind_p99": 18.0, "wind_p999": 20.0, "wind_p9999": 22.0, "wind_max": 24.0,
    }
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(_ratio_stats(model_keys)))

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_abc12345_new_o96_o320",
        canonical_analysis_by_event={},
        canonical_eefo_by_event={},
    )

    assert 0.0 < result["idalia_mslp_p001_ratio"] < 1.0
    assert 0.0 < result["idalia_mslp_min_ratio"] < 1.0
    assert 0.0 < result["idalia_wind_p9999_ratio"] < 1.0
    assert 0.0 < result["idalia_wind_max_ratio"] < 1.0


def test_tail_ratios_missing_new_fields(tmp_path):
    """When mslp_p001 / wind_p9999 are absent (older stats), the ratios for those
    specific keys should be omitted from the result dict; min/max ratios still
    work if those fields are present."""
    # Drop p001 and p9999 from all rows.
    enfo_keys = {"mslp_p1": 980, "mslp_p01": 975, "mslp_min": 965, "wind_p99": 25, "wind_p999": 28, "wind_max": 38}
    an_keys = {"mslp_p1": 985, "mslp_p01": 980, "mslp_min": 972, "wind_p99": 22, "wind_p999": 25, "wind_max": 32}
    model_keys = {"mslp_p1": 985, "mslp_p01": 980, "mslp_min": 970, "wind_p99": 22, "wind_p999": 25, "wind_max": 35}
    payload = {
        "events": {
            "idalia": {
                "extreme_tail": {
                    "rows": [
                        {"exp": "OPER_O320_0001", **an_keys},
                        {"exp": "ENFO_O320_0001", **enfo_keys},
                        {"exp": "manual_abc12345_new_o96_o320", **model_keys},
                    ]
                }
            },
        }
    }
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(payload))

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_abc12345_new_o96_o320",
        canonical_analysis_by_event={},
        canonical_eefo_by_event={},
    )

    assert "idalia_mslp_p001_ratio" not in result
    assert "idalia_wind_p9999_ratio" not in result
    assert result["idalia_mslp_min_ratio"] > 1.0  # 41.25 / 41.25 ... actually 43.25/41.25
    assert result["idalia_wind_max_ratio"] > 1.0
