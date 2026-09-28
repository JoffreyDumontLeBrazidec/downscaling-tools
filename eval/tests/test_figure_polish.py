"""Layout and wording fixes of 2026-09-29: the probabilistic source line and band legend, the TC
ratio legends, the lane diagnostics limits, the evolution grid, the ladder panel titles and
the storm map extent."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def captured_figures(monkeypatch):
    """Collect the figures handed to ``eval.plotting.save_figure`` instead of writing files."""
    import eval.plotting as P

    figs = []

    def fake_save(fig, out, **kwargs):
        figs.append(fig)
        return []

    monkeypatch.setattr(P, "save_figure", fake_save)
    yield figs
    for fig in figs:
        plt.close(fig)


# ---------------------------------------------------------------- probabilistic source line

@pytest.mark.parametrize("lane, expected", [
    ("o320_o1280", "ENFO O1280"),
    ("o1280_o2560", "IEKM O2560"),
    ("o96_o320", "ENFO O320"),
    ("o48_o96", "IEKM O96"),
])
def test_truth_of_the_local_probabilistic_evaluator_comes_from_the_lane(lane, expected):
    from eval.config.loader import load_lane
    from eval.evaluators.probabilistic.core.plotting import truth_from_lane

    assert truth_from_lane(load_lane(lane)) == expected


def test_truth_is_unknown_without_a_lane():
    from eval.evaluators.probabilistic.core.plotting import truth_from_lane
    from eval.plotting.probabilistic import SOURCE_LOCAL, source_local

    assert truth_from_lane({}) is None and truth_from_lane(None) is None
    assert source_local(None) == SOURCE_LOCAL
    assert "ENFO" not in SOURCE_LOCAL
    assert source_local("IEKM O2560").endswith("truth = IEKM O2560 member 0")


def _summary_csv(path):
    rows = ["step,weather_state,domain,metric,mean,std,stderr,n_dates,n_points_total"]
    for step in (24, 48, 72):
        for metric in ("crps", "fcrps", "spread", "rmse_ens_mean"):
            rows.append(f"{step},msl,europe,{metric},{1.0 + step / 100},0.1,0.05,5,1000")
    path.write_text("\n".join(rows) + "\n")


@pytest.mark.parametrize("lane, expected", [("o320_o1280", "ENFO O1280"), ("o1280_o2560", "IEKM O2560")])
def test_local_probabilistic_figure_names_the_truth_of_its_lane(tmp_path, monkeypatch, lane, expected):
    from eval.config.loader import load_lane
    from eval.evaluators.probabilistic.core import plotting
    from eval.evaluators.probabilistic.plotter import plot

    seen = {}

    def fake(curves, source, out, **kwargs):
        seen["source"], seen["kwargs"] = source, kwargs
        return []

    monkeypatch.setattr(plotting, "plot_probabilistic_scores", fake)
    _summary_csv(tmp_path / "summary_by_lead.csv")
    plot(tmp_path, load_lane(lane), {}, output_dir=tmp_path / "out")
    assert f"truth = {expected} member 0" in seen["source"]
    assert "95 % confidence interval" in seen["kwargs"]["band_label"]


def test_probabilistic_legend_explains_the_shaded_band(tmp_path, monkeypatch):
    import pandas as pd

    from eval.plotting import probabilistic as prob
    from eval.plotting.style import FigureBook

    rows = []
    for lead in (24, 48, 72):
        rows.append(dict(metric="fcrps", variable="msl", domain="europe", lead_h=lead,
                         series_role="model", series_label="model", value=float(lead),
                         ci_low=lead - 5.0, ci_high=lead + 5.0))
    figs = []
    orig = FigureBook.add

    def add(self, fig, **kw):
        figs.append(fig)
        return orig(self, fig, **kw)

    monkeypatch.setattr(FigureBook, "add", add)
    prob.plot_probabilistic_scores(pd.DataFrame(rows), prob.SOURCE_LOCAL, tmp_path / "p",
                                   band_label="the band means this")
    texts = [t.get_text() for t in figs[0].legends[0].get_texts()]
    assert "the band means this" in texts
    # no band in the table: no band entry
    figs.clear()
    prob.plot_probabilistic_scores(pd.DataFrame(rows).drop(columns=["ci_low", "ci_high"]),
                                   prob.SOURCE_LOCAL, tmp_path / "q", band_label="the band means this")
    assert "the band means this" not in [t.get_text() for t in figs[0].legends[0].get_texts()]


# ---------------------------------------------------------------- TC ratio legend

def _tc_stats():
    mids = np.arange(2.0, 60.0, 4.0)
    def var(scale):
        return {"bin_mids": mids.tolist(), "bin_edges": list(np.arange(0.0, 62.0, 4.0)),
                "oper_histogram": np.exp(-mids / scale).tolist(),
                "curves": {"truth": {"histogram": np.exp(-mids / (scale * 1.1)).tolist()},
                           "model": {"histogram": np.exp(-mids / (scale * 1.2)).tolist()}}}
    return {"analysis_key": "oper", "curve_order": ["truth", "model"], "support_mode": "native",
            "event": "x", "variables": {"mslp_hpa": var(20.0), "wind10m_ms": var(10.0)}}


def test_tc_ratio_legend_sits_below_the_axes():
    from eval.evaluators.tc.core import pdf_plot
    from eval.evaluators.tc.core.plot_config import resolve_plot_config

    stats = _tc_stats()
    stats["variables"]["mslp_hpa"]["bin_edges"] = list(np.arange(890.0, 1030.0, 10.0))
    cfg = resolve_plot_config("franklin", {})
    fig = pdf_plot.plot_pdf_ratios(cfg, event_stats=stats)
    try:
        fig.canvas.draw()
        for ax in fig.axes:
            leg = ax.get_legend()
            assert leg is not None
            assert leg.get_window_extent().y1 <= ax.get_window_extent().y0 + 1.0
    finally:
        plt.close(fig)


# ---------------------------------------------------------------- lane diagnostics figure 1

def test_capacity_curve_limits_follow_the_data(tmp_path):
    from eval.evaluators.lane_diagnostics import figures

    rng = np.random.default_rng(0)
    rows = []
    # a small gap that the model overshoots: the fraction closed reaches about 1.6 in "0-5"
    for gap, closed, n in ((-3.0, 1.0, 12), (2.5, 4.0, 12), (7.0, 3.0, 12), (15.0, 5.0, 12),
                           (25.0, 6.0, 12), (40.0, 7.0, 12)):
        for _ in range(n):
            g = gap + rng.normal(0, 0.3)
            c = closed + rng.normal(0, 0.3)
            rows.append(dict(step=24, member=1, date="20250926", truth=990.0, interp=990.0 + g,
                             model=990.0 + g - c, truth_lat=20.0, truth_lon=-60.0, interp_lat=20.0,
                             interp_lon=-60.0, model_lat=20.0, model_lon=-60.0))
    path = tmp_path / "cap.json"
    path.write_text(json.dumps(rows))
    cap = figures.load_capacity(path)
    fig, _caption = figures.fig01_capacity_curve(cap, None)
    try:
        ax = fig.axes[1]
        tallest = np.nanmax([p.get_height() for p in ax.patches])
        assert tallest > 1.45
        top = ax.get_ylim()[1]
        assert top > tallest
        # error bars are inside the axes too
        for line in ax.lines:
            y = np.asarray(line.get_ydata(), dtype=float)
            assert not np.isfinite(y).any() or np.nanmax(y) <= top + 1e-9
    finally:
        plt.close(fig)


# ---------------------------------------------------------------- evolution grid

def _card(step_values, key_values):
    return {"lane": "l", "profile_pins": {"budget": {"dates": "d", "steps": "s", "members": "m"}},
            "rows": [{"step": s, "metrics": {k: v * (1 + i * 0.01) for k, v in key_values.items()}}
                     for i, s in enumerate(step_values)]}


def _evolution_inputs():
    key = "probabilistic_10u_n.hem_rmse_ens_mean_mean"
    exp = _card([1000, 2000, 3000], {key: 2.0})
    flat = {key: 1.5}
    return exp, flat


def test_evolution_leaves_out_rows_and_columns_without_data(tmp_path, captured_figures):
    from eval.jobs import evolution

    exp, flat = _evolution_inputs()
    evolution.render([("run", exp)], tmp_path / "e.png", reference=("ref", exp),
                     input_ref=("in", flat), target_ref=("tgt", flat),
                     rows=["10u", "tp"], columns=["rmse", "spectra"])
    fig = captured_figures[-1]
    assert len(fig.axes) == 1                       # one row, one column
    footer = " ".join(t.get_text() for t in fig.texts)
    assert "Left out, no data on this lane" in footer
    assert "rows tp" in footer and "columns Spectral relative L2 distance" in footer


def test_evolution_keep_empty_restores_the_empty_panels(tmp_path, captured_figures):
    from eval.jobs import evolution

    exp, flat = _evolution_inputs()
    evolution.render([("run", exp)], tmp_path / "e.png", reference=("ref", exp),
                     input_ref=("in", flat), target_ref=("tgt", flat),
                     rows=["10u", "tp"], columns=["rmse", "spectra"], keep_empty=True)
    fig = captured_figures[-1]
    assert len(fig.axes) == 4
    assert "Left out" not in " ".join(t.get_text() for t in fig.texts)


def test_evolution_with_nothing_to_draw_says_so(tmp_path):
    from eval.jobs import evolution

    exp, flat = _evolution_inputs()
    with pytest.raises(SystemExit, match="none of the requested rows and columns"):
        evolution.render([("run", exp)], tmp_path / "e.png", reference=("ref", exp),
                         input_ref=("in", flat), target_ref=("tgt", flat),
                         rows=["tp"], columns=["spectra"])


# ---------------------------------------------------------------- ladder panel titles

def test_ladder_verdict_does_not_share_a_line_with_the_panel_title(monkeypatch, captured_figures):
    import argparse

    from eval.jobs import ladder

    key = "probabilistic_2t_n.hem_rmse_ens_mean_mean"
    card = {"card_id": "t", "rows": [{"step": 1000, "metrics": {key: 2.0}, "checkpoint": "/x/r/c.ckpt"},
                                     {"step": 2000, "metrics": {key: 1.9}, "checkpoint": "/x/r/c.ckpt"}],
            "baselines": {"ref": {"metrics": {key: 3.0}, "eval_core_sha": "abc"}}, "loss": {}}
    monkeypatch.setattr(ladder, "load_profile", lambda name: {"_name": "t", "card_id": "t"})
    monkeypatch.setattr(ladder, "load_ladder", lambda prof: card)
    ladder.cmd_plot(argparse.Namespace(profile="t", out="unused.png"))
    fig = captured_figures[-1]
    fig.canvas.draw()
    ax = fig.axes[0]
    verdicts = [t for t in ax.texts if t.get_text() == "better"]
    assert verdicts and ax.get_title(loc="right") == ""
    renderer = fig.canvas.get_renderer()
    assert not verdicts[0].get_window_extent(renderer).overlaps(
        ax._left_title.get_window_extent(renderer))


# ---------------------------------------------------------------- storm map extent

@pytest.mark.parametrize("extent", [(-80.0, -50.0, 15.0, 35.0), (100.0, 140.0, 5.0, 30.0),
                                    (-30.0, 10.0, -35.0, -10.0)])
def test_inscribed_extent_lies_inside_the_box(extent):
    import cartopy.crs as ccrs

    from eval.plotting.maps import inscribed_extent, select_projection

    west, east, south, north = extent
    proj = select_projection(*extent)
    if not isinstance(proj, ccrs.LambertConformal):
        pytest.skip("only conic projections have wedges")
    x0, x1, y0, y1 = inscribed_extent(proj, extent)
    assert x0 < x1 and y0 < y1
    xs, ys = np.meshgrid(np.linspace(x0, x1, 15), np.linspace(y0, y1, 15))
    pts = ccrs.PlateCarree().transform_points(proj, xs.ravel(), ys.ravel())
    lon = (pts[:, 0] - west + 180.0) % 360.0 - 180.0 + west
    tol = 1e-6
    assert lon.min() >= west - tol and lon.max() <= east + tol
    assert pts[:, 1].min() >= south - tol and pts[:, 1].max() <= north + tol


def test_filled_map_axes_use_the_inscribed_rectangle():
    import cartopy.crs as ccrs

    from eval.plotting.maps import inscribed_extent, map_grid, select_projection

    extent = (-80.0, -50.0, 15.0, 35.0)
    fig, axes = map_grid(1, 1, extent, fill=True, geography=False)
    try:
        proj = select_projection(*extent)
        x0, x1, y0, y1 = inscribed_extent(proj, extent)
        got = axes[0, 0].get_extent(crs=proj)
        assert np.allclose(got, (x0, x1, y0, y1), rtol=1e-6)
    finally:
        plt.close(fig)
