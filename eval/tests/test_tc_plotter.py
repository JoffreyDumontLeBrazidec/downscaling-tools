from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eval._backends.tc import pdf_plot
from eval._backends.tc.plot_config import TCPlotConfig
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

    class RecordingPdfPages:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def savefig(self, fig, *, dpi=None):
            saved_pages.append(fig._tc_page_kind)  # type: ignore[attr-defined]

    def _figure(event_stats: dict):
        fig = plt.figure()
        fig._tc_page_kind = f"overview:{event_stats['event']}:{event_stats['support_mode']}"  # type: ignore[attr-defined]
        return fig

    monkeypatch.setattr(plotter, "PdfPages", RecordingPdfPages)
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

    assert "OPER O320" in labels
    assert "OPER AN" not in labels


def test_tc_log_plot_uses_high_contrast_orange_for_enfo_o320():
    style = pdf_plot.curve_style(
        "ENFO_O320_0001",
        ml_palette=None,  # This reference style does not consume the model palette.
        ml_index=0,
    )

    assert isinstance(style["color"], str)
    assert style["color"] == "#E69F00"
    assert style["linestyle"] == "-."


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
        assert mslp_ax.get_title() == "Mean sea level pressure (PDF)"
        assert wind_ax.get_title() == "Wind speed (PDF)"
        assert any(line.get_visible() for ax in fig.axes for line in [*ax.get_xgridlines(), *ax.get_ygridlines()])

        oper_line = mslp_ax.lines[0]
        assert oper_line.get_color() == "#000000"
        assert oper_line.get_linestyle() == "-"
        assert oper_line.get_linewidth() >= 3.0

        enfo_line = mslp_ax.lines[1]
        assert enfo_line.get_color() == "#E69F00"
        assert enfo_line.get_linestyle() == "-."

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
