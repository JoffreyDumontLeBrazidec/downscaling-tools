from __future__ import annotations

import numpy as np

from eval.evaluators.tctracks import plotter, scorer

from .test_scorer import _source


def _sources():
    return {
        "model": _source("model", n_tracks=8, mslp_min_base=980.0),
        "ctrl": _source("ctrl", n_tracks=7, mslp_min_base=985.0),
        "target": _source("target", n_tracks=12, mslp_min_base=960.0),
        "input": _source("input", n_tracks=6, mslp_min_base=990.0),
    }


def test_select_cases_extended_fields():
    sources = _sources()
    cases = scorer.select_cases(sources, ["202509"], "atl", top_k=2)
    assert cases and cases[0]["basin"] == "atl"
    member = cases[0]["members"]["target"][0]
    # original keys stay
    assert {"init_date", "member", "track_id"} <= set(member)
    # new per-track fields for the case pages
    assert {"mslp_min_hpa", "mslp_min_lat", "mslp_min_lon_e",
            "mslp_min_valid_time", "wind_max_ms"} <= set(member)
    assert member["mslp_min_hpa"] == 960.0


def test_case_label_readable():
    sources = _sources()
    case = scorer.select_cases(sources, ["202509"], "atl", top_k=1)[0]
    label = plotter.case_label(case)
    assert label.startswith("ATL, deepest 2025-09-")
    assert "960 hPa" in label and "near" in label


def test_render_all_page_report(tmp_path):
    sources = _sources()
    metrics = scorer.score_sources(sources, months=["202509"], basins=["atl"])
    paths = plotter.render_all(sources, metrics, ["202509"], ["atl"], tmp_path,
                               top_k_cases=2)
    # report 1 = per-basin tc-style distribution pages; report 2 = diagnostics
    for pdf_name in ("tc_tracks_report.pdf", "tc_tracks_diagnostics.pdf"):
        pdf = tmp_path / pdf_name
        assert pdf.exists() and pdf.stat().st_size > 0
    names = {p.name for p in paths}
    assert "dist_atl.png" in names
    assert "page1_overview.png" in names
    assert "page2_atl_all_tcs.png" in names
    # single basin -> no other-basins page; 2 case pages
    assert not any(n.startswith("page3") for n in names)
    assert sum(n.startswith("case_atl_case") for n in names) == 2


def test_render_all_multi_basin_and_per_month(tmp_path):
    sources = _sources()
    # clone the atl tracks into a second basin so wnp is populated
    for src in sources.values():
        for key in ("records", "summary"):
            other = src[key].copy()
            other["basin"] = "wnp"
            other["init_date"] = other["init_date"].str.replace("202509", "202510")
            src[key] = __import__("pandas").concat([src[key], other], ignore_index=True)
        fc = src["forecasts"].copy()
        fc["init_date"] = fc["init_date"].str.replace("202509", "202510")
        src["forecasts"] = __import__("pandas").concat([src["forecasts"], fc], ignore_index=True)
    months = ["202509", "202510"]
    metrics = scorer.score_sources(sources, months=months, basins=["atl", "wnp"])
    paths = plotter.render_all(sources, metrics, months, ["atl", "wnp"], tmp_path,
                               per_month=True, top_k_cases=1)
    names = {p.name for p in paths}
    assert {"dist_atl.png", "dist_wnp.png"} <= names
    assert "page3_other_basins.png" in names
    assert {"month_atl_202509.png", "month_atl_202510.png"} <= names
    # case pages come only from the default case basin (atl)
    assert sum(n.startswith("case_atl_") for n in names) == 1
    assert not any(n.startswith("case_wnp_") for n in names)


def test_haversine_and_latlon_format():
    assert plotter._fmt_latlon(25.0, 285.0) == "25N 75W"
    assert plotter._fmt_latlon(-10.0, 100.0) == "10S 100E"
    d = plotter._haversine_km(0.0, 0.0, 0.0, 1.0)
    assert np.isclose(d, 111.19, atol=0.5)
