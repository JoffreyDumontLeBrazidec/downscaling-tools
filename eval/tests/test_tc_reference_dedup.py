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
