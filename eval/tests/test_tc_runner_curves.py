"""Regression cover for two defects found in production on 2026-08-24."""
import logging

import pytest

from eval.evaluators.tc import runner


# --- 1. the input curve must not vanish ---------------------------------------
# The de-duplication dropped a reference saying the input was "already drawn",
# while the draw site skipped it because the config key was unset. The curve
# disappeared with only a warning asserting the opposite.


def test_input_label_has_a_default_so_the_input_is_always_drawn():
    input_label, target_nc_label = runner.resolve_bundle_curve_labels({})
    assert input_label, "an unset input_label must still draw the input curve"
    assert target_nc_label is None, (
        "the target must NOT default: most lanes draw it from target_grib and a "
        "default would add a duplicate target curve"
    )


def test_explicit_labels_win():
    got = runner.resolve_bundle_curve_labels(
        {"input_label": "ENFO O1280", "target_nc_label": "IEKM nc"})
    assert got == ("ENFO O1280", "IEKM nc")


def _lane(lres, target):
    return {"prepare": {"args": {"lres_sfc_grib": lres, "target_sfc_grib": target}}}


def test_reference_duplicating_the_input_is_dropped_because_the_input_is_drawn():
    lane = _lane("/x/enfo_o1280_0001_date20250926.grib", "/x/iekm_o2560_iekm_date20250926.grib")
    kept = runner._strip_bundle_duplicate_references(
        ["ENFO_O1280_0001"], lane, {"target_grib": "/x/iekm*.grib"})
    assert kept == ()


def test_reference_duplicating_the_target_is_KEPT_when_no_target_curve_is_drawn():
    """The bug shape, on the target side: dropping a reference that nothing
    replaces removes the curve rather than de-duplicating it."""
    lane = _lane("/x/enfo_o1280_0001_date20250926.grib", "/x/iekm_o2560_0002_date20250926.grib")
    kept = runner._strip_bundle_duplicate_references(["IEKM_O2560_0002"], lane, {})
    assert kept == ("IEKM_O2560_0002",), "must not drop a curve nothing will redraw"


def test_reference_duplicating_the_target_is_dropped_when_the_target_grib_draws_it():
    lane = _lane("/x/enfo_o1280_0001_date20250926.grib", "/x/iekm_o2560_0002_date20250926.grib")
    kept = runner._strip_bundle_duplicate_references(
        ["IEKM_O2560_0002"], lane, {"target_grib": "/x/iekm*.grib"})
    assert kept == ()


def test_a_target_filename_without_a_numeric_expid_yields_no_product():
    """The real IEKM target on this lane is 'iekm_o2560_iekm_date...', whose third
    token is not four digits, so no expid is derived and the target side of the
    dedup simply never engages. Pinned so a future regex change is a deliberate
    decision rather than a surprise."""
    assert runner._expid_from_grib_path(
        "/x/iekm_o2560_iekm_date20250926_time0000_step24to120_sfc_y.grib") is None
    assert runner._expid_from_grib_path(
        "/x/enfo_o1280_0001_date20250926.grib") == "ENFO_O1280_0001"


def test_unrelated_references_are_never_touched():
    lane = _lane("/x/enfo_o1280_0001_date20250926.grib", "/x/iekm_o2560_iekm_date20250926.grib")
    kept = runner._strip_bundle_duplicate_references(
        ["OPER_O1280_0001"], lane, {"target_grib": "/x/iekm*.grib"})
    assert kept == ("OPER_O1280_0001",)


def test_dedup_is_a_no_op_without_bundle_products():
    kept = runner._strip_bundle_duplicate_references(
        ["ENFO_O1280_0001"], {"prepare": {"args": {}}}, {})
    assert kept == ["ENFO_O1280_0001"]
