from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eval._backends.tc import pdf_plot
from eval._backends.tc.plot_config import REFERENCE_STYLES, TCPlotConfig
from eval.plotting import INPUT_COLOR, MODEL_COLOR, TRUTH_COLOR, role_style
from eval.evaluators.tc import plotter


def test_tc_plotter_writes_only_overview_pages_in_canonical_order(tmp_path: Path, monkeypatch):
    results_dir = tmp_path / "tc"
    results_dir.mkdir()
    def event_stats(event: str, mode: str) -> dict:
        contract = {
            "geographic_box": {"north": 40.0, "south": 10.0, "east": -80.0, "west": -100.0},
            "support_mode": mode,
            "regrid_resolution_degrees": 0.25,
            "ensemble_members": 10,
            "lead_times_hours": [24],
            "start_dates": ["2023-08-26"],
            "valid_dates": ["2023-08-27"],
            "analysis_reference": "OPER_O320_0001",
        }
        return {
            "event": event,
            "support_mode": mode,
            "comparison_contract": contract,
            "reference_comparison_contract": dict(contract),
        }

    stats = {
        "events": {
            "idalia": event_stats("idalia", "regridded"),
            "franklin__native": event_stats("franklin", "native"),
            "idalia__native": event_stats("idalia", "native"),
            "franklin": event_stats("franklin", "regridded"),
        }
    }
    (results_dir / "stats.json").write_text(json.dumps(stats), encoding="utf-8")

    saved_pages: list[str] = []

    class RecordingFigureBook:
        def __init__(self, path, *, png=False):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def add(self, fig, name=None, **kwargs):
            saved_pages.append(fig._tc_page_kind)  # type: ignore[attr-defined]
            plt.close(fig)

    def _figure(event_stats: dict):
        fig = plt.figure()
        fig._tc_page_kind = f"overview:{event_stats['event']}:{event_stats['support_mode']}"  # type: ignore[attr-defined]
        return fig

    monkeypatch.setattr(plotter, "FigureBook", RecordingFigureBook)
    monkeypatch.setattr(
        plotter,
        "plot_pdf_distribution_overview",
        lambda *args, **kwargs: _figure(kwargs["event_stats"]),
    )
    monkeypatch.setattr(
        plotter,
        "plot_pdf_ratios",
        lambda *args, **kwargs: pytest.fail("ratio-to-OPER pages must not be rendered"),
    )
    monkeypatch.setattr(
        plotter,
        "plot_pdf_log", lambda *args, **kwargs: pytest.fail("log-density pages must not be rendered"),
    )

    plotter.plot(results_dir, {}, {"events": ["idalia", "franklin"]})

    assert saved_pages == [
        "overview:franklin:native",
        "overview:franklin:regridded",
        "overview:idalia:native",
        "overview:idalia:regridded",
    ]


def test_tc_log_plot_labels_oper_o320_explicitly():
    event_stats = {
        "analysis_key": "OPER_O320_0001",
        "curve_order": ["model"],
        "variables": {
            "mslp_hpa": {
                "bin_edges": [990.0, 995.0, 1000.0],
                "bin_mids": [992.5, 997.5],
                "oper_histogram": [0.1, 0.2],
                "curves": {"model": {"histogram": [0.2, 0.1]}},
            },
            "wind10m_ms": {
                "bin_edges": [0.0, 4.0, 8.0],
                "bin_mids": [2.0, 6.0],
                "oper_histogram": [0.2, 0.1],
                "curves": {"model": {"histogram": [0.1, 0.2]}},
            },
        },
    }

    fig = pdf_plot.plot_pdf_log(TCPlotConfig(plot_title="Idalia"), event_stats=event_stats)
    labels = [text.get_text() for ax in fig.axes for text in ax.get_legend().get_texts()]
    plt.close(fig)

    # the analysis is the truth and is named by its grid, never by the raw key or "OPER AN"
    assert "truth (operational analysis O320)" in labels
    assert "OPER AN" not in labels
    assert "OPER_O320_0001" not in labels


def test_tc_log_plot_uses_the_fixed_reference_style_for_enfo_o320():
    style = pdf_plot.curve_style(
        "ENFO_O320_0001",
        ml_palette=None,  # This reference style does not consume the model palette.
        ml_index=0,
    )

    assert isinstance(style["color"], str)
    assert style["color"] == REFERENCE_STYLES["ENFO_O320_0001"]["color"]
    assert style["linestyle"] == REFERENCE_STYLES["ENFO_O320_0001"]["linestyle"]
    assert style["color"].lower() not in {TRUTH_COLOR, MODEL_COLOR.lower(), INPUT_COLOR.lower()}


def test_tc_overview_plot_matches_operational_distribution_style():
    event_stats = {
        "analysis_key": "OPER_O320_0001",
        "curve_order": ["ENFO_O320_0001", "model"],
        "variables": {
            "mslp_hpa": {
                "bin_edges": [985.0, 990.0, 995.0, 1000.0, 1005.0],
                "bin_mids": [987.5, 992.5, 997.5, 1002.5],
                "data_range_msl": [990.0, 1000.0],
                "oper_histogram": [0.0, 0.1, 0.2, 0.0],
                "curves": {
                    "ENFO_O320_0001": {"histogram": [0.0, 0.08, 0.22, 0.04]},
                    "model": {"histogram": [0.02, 0.0, 0.22, 0.0]},
                },
            },
            "wind10m_ms": {
                "bin_edges": [0.0, 4.0, 8.0, 12.0],
                "bin_mids": [2.0, 6.0, 10.0],
                "data_range_wind": [0.0, 10.0],
                "oper_histogram": [0.2, 0.0, 0.01],
                "curves": {
                    "ENFO_O320_0001": {"histogram": [0.18, 0.12, 0.0]},
                    "model": {"histogram": [0.0, 0.12, 0.02]},
                },
            },
        },
    }

    fig = pdf_plot.plot_pdf_distribution_overview(TCPlotConfig(plot_title="Idalia"), event_stats=event_stats)
    try:
        mslp_ax, wind_ax = fig.axes
        assert mslp_ax.get_xlim()[0] > mslp_ax.get_xlim()[1]
        assert wind_ax.get_xlim()[0] < wind_ax.get_xlim()[1]
        assert mslp_ax.get_title() == "Mean sea level pressure"
        assert wind_ax.get_title() == "10 m wind speed"
        assert mslp_ax.get_xlabel() == "Mean sea level pressure (hPa)"
        assert wind_ax.get_xlabel() == "10 m wind speed (m s⁻¹)"
        assert any(line.get_visible() for ax in fig.axes for line in [*ax.get_xgridlines(), *ax.get_ygridlines()])

        oper_line = mslp_ax.lines[0]
        assert oper_line.get_color() == TRUTH_COLOR
        assert oper_line.get_linestyle() == "-"
        assert oper_line.get_linewidth() == role_style("truth")["linewidth"]

        # legend order is truth, model, input, references: the model comes second
        model_line = mslp_ax.lines[1]
        assert model_line.get_color() == MODEL_COLOR

        enfo_line = mslp_ax.lines[2]
        assert enfo_line.get_color() == REFERENCE_STYLES["ENFO_O320_0001"]["color"]
        labels = [t.get_text() for t in mslp_ax.get_legend().get_texts()]
        assert "ENFO O320" in labels and "ENFO_O320_0001" not in labels

        assert any(np.isnan(line.get_ydata()).any() for ax in fig.axes for line in ax.lines)
        assert all(np.all(np.asarray(line.get_ydata())[np.isfinite(line.get_ydata())] > 0.0) for ax in fig.axes for line in ax.lines)
    finally:
        plt.close(fig)

def test_tc_plotter_rejects_stats_without_a_comparison_contract(tmp_path: Path, monkeypatch):
    results_dir = tmp_path / "tc"
    results_dir.mkdir()
    (results_dir / "stats.json").write_text(
        json.dumps({"events": {"idalia": {"event": "idalia", "support_mode": "regridded"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(plotter, "plot_pdf_log", lambda *args, **kwargs: plt.figure())

    with pytest.raises(ValueError, match="comparison contract"):
        plotter.plot(results_dir, {}, {"events": ["idalia"]})


def test_tc_reference_styles_are_unique_and_avoid_role_colours():
    pairs = [(str(v["color"]).lower(), str(v["linestyle"])) for v in REFERENCE_STYLES.values()]
    assert len(set(pairs)) == len(pairs)
    colours = {c for c, _ in pairs}
    assert not colours & {TRUTH_COLOR, MODEL_COLOR.lower(), INPUT_COLOR.lower(), "black", "red"}
    assert all("_" not in str(v["label"]) for v in REFERENCE_STYLES.values())


def test_tc_curve_roles_and_labels_hide_raw_keys():
    role = pdf_plot.curve_role
    assert role("target O1280", analysis_key="target O1280") == "truth"
    assert role("input O320", analysis_key="target O1280") == "input"
    assert role("eval_inputs", analysis_key="target O1280") == "model"
    assert role("ENFO_O1280_0001", analysis_key="OPER_O1280_0001") == "reference"
    assert role("OPER_O1280_0001", analysis_key="ENFO_O1280_0001") == "reference"
    assert role("x", analysis_key="y", curve_roles={"x": "model"}) == "model"
    assert pdf_plot.curve_label("eval_inputs", {}, oper_key="target O1280") == "model"
    assert pdf_plot.curve_label("input O320", {}, oper_key="target O1280") == "input (O320)"
    assert pdf_plot.curve_label("ENFO_O1280_0001", {}, oper_key="OPER_O1280_0001") == "ENFO O1280"


def test_tc_several_models_take_distinct_sequence_colours():
    styles, roles = pdf_plot._figure_styles(["an", "run_a", "run_b", "input"], analysis_key="an")
    assert roles == {"an": "truth", "run_a": "model", "run_b": "model", "input": "input"}
    assert styles["run_a"]["color"] != styles["run_b"]["color"]
    assert MODEL_COLOR not in (styles["run_a"]["color"], styles["run_b"]["color"])


def test_tc_plotter_labels_bundle_curves_from_the_lane():
    lane = {"prepare": {"args": {
        "lres_sfc_grib": "r/eefo_o320_0001_date{d}_sfc.grib",
        "target_sfc_grib": "r/enfo_o1280_0001_date{d}_sfc_y.grib",
    }}}
    ev_cfg = {"input_label": "input O320", "target_nc_label": "target O1280"}
    stats = {"analysis_key": "target O1280", "curve_order": ["eval_inputs", "input O320"]}
    labels, roles = plotter.curve_labels_and_roles(stats, lane, ev_cfg)
    assert labels == {"target O1280": "truth (ENFO O1280)", "input O320": "input (EEFO O320)"}
    assert roles == {"input O320": "input"}
