from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from eval.evaluators.tc import plotter


def test_tc_plotter_puts_all_overview_pages_before_ratio_and_log(tmp_path: Path, monkeypatch):
    results_dir = tmp_path / "tc"
    results_dir.mkdir()
    stats = {
        "events": {
            "humberto": {"event": "humberto", "support_mode": "regridded"},
            "humberto__native": {"event": "humberto", "support_mode": "native"},
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

    def _figure(kind: str, event_stats: dict):
        fig = plt.figure()
        fig._tc_page_kind = f"{kind}:{event_stats['support_mode']}"  # type: ignore[attr-defined]
        return fig

    monkeypatch.setattr(plotter, "PdfPages", RecordingPdfPages)
    monkeypatch.setattr(
        plotter,
        "plot_pdf_distribution_overview",
        lambda plot_config, *, event_stats: _figure("overview", event_stats),
    )
    monkeypatch.setattr(
        plotter,
        "plot_pdf_ratios",
        lambda plot_config, *, event_stats: _figure("ratio", event_stats),
    )
    monkeypatch.setattr(
        plotter,
        "plot_pdf_log",
        lambda plot_config, *, event_stats: _figure("log", event_stats),
    )

    plotter.plot(results_dir, {}, {})

    assert saved_pages == [
        "overview:regridded",
        "overview:native",
        "ratio:regridded",
        "log:regridded",
        "ratio:native",
        "log:native",
    ]
