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


def test_role_styles_follow_the_house_roles():
    from eval.plotting import INPUT_COLOR, MODEL_COLOR, TRUTH_COLOR

    assert plotter.role_color("target") == TRUTH_COLOR
    assert plotter.role_color("model") == MODEL_COLOR
    assert plotter.role_color("input") == INPUT_COLOR
    assert plotter.role_color("ctrl") not in (TRUTH_COLOR, MODEL_COLOR, INPUT_COLOR)
    assert plotter.role_color("extra", 0) not in (TRUTH_COLOR, MODEL_COLOR, INPUT_COLOR)


def test_source_and_role_labels_are_readable():
    assert plotter.source_name({"provenance": {"source_id": "od_enfo_0001"}}, "target") == "ENFO"
    assert plotter.source_name({"provenance": {"source_id": "ai_enfo_0001"}}, "target") == "AI ENFO"
    assert plotter.role_label("target", {"provenance": {"source_id": "od_enfo_0001"}}) == "truth (ENFO)"
    assert plotter.role_label("model", {"provenance": {"source_id": "ja6g"}}) == "model (ja6g)"
    assert plotter.month_text("202509") == "September 2025"
    assert plotter.basin_title("wnp") == "Western North Pacific (WNP)"


def test_render_all_writes_png_and_pdf_siblings(tmp_path):
    sources = _sources()
    metrics = scorer.score_sources(sources, months=["202509"], basins=["atl"])
    plotter.render_all(sources, metrics, ["202509"], ["atl"], tmp_path, top_k_cases=1)
    assert (tmp_path / "figures" / "dist_atl.png").exists()
    assert (tmp_path / "figures" / "dist_atl.pdf").exists()


def _map_axes(fig):
    return [ax for ax in fig.axes if hasattr(ax, "coastlines")]


def test_basin_grid_has_one_map_per_source_and_no_inset(tmp_path):
    """Two sources besides the truth give two maps that share the row (no empty third slot),
    and the ratio to the truth is an axes of its own, not an inset over the curves."""
    import matplotlib.pyplot as plt

    sources = {k: v for k, v in _sources().items() if k in ("model", "target", "input")}
    metrics = scorer.score_sources(sources, months=["202509"], basins=["atl"])
    fig = plotter.page_basin_grid(sources, metrics, ["202509"], "atl", "all")
    try:
        assert len(_map_axes(fig)) == 2
        assert all(not ax.child_axes for ax in fig.axes)
        ratio_axes = [ax for ax in fig.axes if ax.get_ylabel().startswith("Ratio to")]
        assert len(ratio_axes) == 1
    finally:
        plt.close(fig)


def test_basin_grid_with_three_sources_has_three_maps():
    import matplotlib.pyplot as plt

    sources = _sources()
    del sources["ctrl"]
    sources["ctrl2"] = _source("ctrl", n_tracks=5, mslp_min_base=983.0)
    metrics = scorer.score_sources(sources, months=["202509"], basins=["atl"])
    fig = plotter.page_basin_grid(sources, metrics, ["202509"], "atl", "all")
    try:
        assert len(_map_axes(fig)) == 3
    finally:
        plt.close(fig)


def test_overview_ratio_to_truth_is_its_own_axes_under_the_pdf():
    import matplotlib.pyplot as plt

    sources = _sources()
    metrics = scorer.score_sources(sources, months=["202509"], basins=["atl"])
    fig = plotter.page_overview(sources, metrics, ["202509"], ["atl"], "atl")
    try:
        assert all(not ax.child_axes for ax in fig.axes)
        pdf_ax = next(ax for ax in fig.axes if ax.get_yscale() == "log")
        ratio_ax = next(ax for ax in fig.axes if ax.get_ylabel().startswith("Ratio to"))
        assert ratio_ax.get_position().y1 <= pdf_ax.get_position().y0 + 1e-9
    finally:
        plt.close(fig)
