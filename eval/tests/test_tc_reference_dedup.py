"""Lock in the TC reference-dedup guard (duplicate input/target curves).

Regression history: the duplicate input O320==EEFO_O320 / target O1280==ENFO_O1280 curves
were removed in 0f45ed7 ("truth-aware bundle") then silently re-added by 8656a1a. This test
keeps the code-level guard honest so a future config revert cannot re-break the plots.
"""
from eval.evaluators.tc.runner import (
    _expid_from_grib_path,
    _strip_bundle_duplicate_references,
)


def test_expid_from_grib_path():
    assert (
        _expid_from_grib_path(
            "/r/eefo_o320_0001_date20230826_time0000_mem1to10_step24to120_sfc.grib"
        )
        == "EEFO_O320_0001"
    )
    assert (
        _expid_from_grib_path(
            "/r/enfo_o1280_0001_date20230826_time0000_mem1to10_step24to120_sfc_y.grib"
        )
        == "ENFO_O1280_0001"
    )
    assert _expid_from_grib_path(None) is None
    assert _expid_from_grib_path("/r/no_expid_token.grib") is None


def test_strips_input_and_target_duplicates():
    lane = {
        "prepare": {
            "args": {
                "lres_sfc_grib": "{root}/eefo_o320_0001_date{date}_time0000_sfc.grib",
                "target_sfc_grib": "{root}/enfo_o1280_0001_date{date}_time0000_sfc_y.grib",
            }
        }
    }
    ev = {"input_label": "input O320", "target_nc_label": "target O1280"}
    refs = ("EEFO_O320_0001", "ENFO_O1280_0001", "OPER_O1280_0001")
    # the two bundle-duplicate refs are dropped; the independent analysis survives
    assert _strip_bundle_duplicate_references(refs, lane, ev) == ("OPER_O1280_0001",)


def test_noop_without_prepare_block():
    refs = ("ENFO_O1280_0001",)
    assert _strip_bundle_duplicate_references(refs, {}, {}) == refs


def test_noop_when_empty():
    assert _strip_bundle_duplicate_references((), {"prepare": {"args": {}}}, {}) == ()


import json

from eval._backends.scoreboard.row_matching import bundle_enfo_labels
from eval._backends.scoreboard.tc import load_tc_extreme_scores_from_json


def test_bundle_enfo_labels_target_is_enfo():
    lane = {"prepare": {"args": {
        "lres_sfc_grib": "r/eefo_o320_0001_date{d}_sfc.grib",
        "target_sfc_grib": "r/enfo_o1280_0001_date{d}_sfc_y.grib",
    }}}
    ev = {"input_label": "input O320", "target_nc_label": "target O1280"}
    assert bundle_enfo_labels(lane, ev) == {"target O1280"}


def test_bundle_enfo_labels_input_is_enfo():
    lane = {"prepare": {"args": {
        "lres_sfc_grib": "r/enfo_o1280_0001_date{d}_input.grib",
        "target_sfc_grib": "r/destine_rd_fc_oper_o2560_date{d}_y.grib",
    }}}
    ev = {"input_label": "x_interp", "target_label": "IEKM"}
    assert bundle_enfo_labels(lane, ev) == {"x_interp"}


def test_enfo_column_sourced_from_bundle_target(tmp_path):
    stats = {"events": {"franklin": {"extreme_tail": {"rows": [
        {"exp": "manual", "mslp_min": 969.0, "mslp_p001": 979.1, "wind_max": 49.5, "wind_p9999": 35.4},
        {"exp": "target O1280", "mslp_min": 942.5, "mslp_p001": 965.8, "wind_max": 57.9, "wind_p9999": 36.0},
        {"exp": "input O320", "mslp_min": 973.3, "mslp_p001": 981.9, "wind_max": 29.1, "wind_p9999": 25.6},
        {"exp": "OPER-AN O1280", "mslp_min": 951.0, "mslp_p001": 966.5, "wind_max": 41.6, "wind_p9999": 35.0},
    ]}}}}
    p = tmp_path / "stats.json"
    p.write_text(json.dumps(stats))
    sc = load_tc_extreme_scores_from_json(
        p, run_id="manual", event_names=("franklin",), enfo_labels={"target O1280"}
    )
    assert sc["franklin_enfo_mslp_min"] == 942.5   # enfo from the bundle target(=ENFO) row
    assert sc["franklin_mslp_min"] == 969.0        # model still the manual row
    assert sc["franklin_oper_mslp_min"] == 951.0   # oper still OPER-AN
    assert "franklin_eefo_mslp_min" not in sc      # eefo stays blank (matches existing boards)
