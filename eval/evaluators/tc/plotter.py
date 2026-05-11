"""TC evaluator — visualization (member maps + PDF ratio plots).

Delegates to eval.tc.pdf_plot and eval.tc.member_plot for rendering.
Uses eval.shared.plotting for coastline rendering.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from eval._backends.tc.events import EVENTS
from eval._backends.tc.pdf_plot import plot_pdf_log, plot_pdf_ratios
from dataclasses import replace

from eval._backends.tc.plot_config import TCPlotConfig, resolve_plot_config

LOG = logging.getLogger(__name__)


def plot(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    stats_filename: str = "stats.json",
) -> Path:
    """Generate TC PDF ratio plots from previously computed stats.

    Reads stats JSON from results_dir, renders one figure per event,
    and writes a combined PDF to output_dir/plots/.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    stats_path = results_dir / stats_filename
    if not stats_path.exists():
        raise FileNotFoundError(f"TC stats file not found: {stats_path}")

    with open(stats_path) as f:
        payload = json.load(f)

    events_data = payload.get("events", {})
    if not events_data:
        LOG.warning("No event data in %s", stats_path)
        return plots_dir

    pdf_path = plots_dir / "all_tc_distributions.pdf"
    with PdfPages(pdf_path) as pdf:
        for event_key, event_stats in events_data.items():
            if event_stats.get("prediction_only"):
                LOG.info("Skipping plot for prediction-only event=%s", event_key)
                continue

            # Use base event name for config lookup (strip __native etc.)
            base_event = event_stats.get("event", event_key)
            mode = event_stats.get("support_mode", "")
            mode_suffix = f" [{mode}]" if mode else ""

            plot_cfg = resolve_plot_config(base_event, eval_config)
            # Override title with mode annotation
            plot_cfg = replace(plot_cfg, plot_title=plot_cfg.plot_title + mode_suffix)

            fig = plot_pdf_ratios(
                plot_cfg,
                event_stats=event_stats,
            )
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            fig_log = plot_pdf_log(
                plot_cfg,
                event_stats=event_stats,
            )
            pdf.savefig(fig_log, dpi=300)
            plt.close(fig_log)
            LOG.info("Plotted TC ratio + log for event=%s mode=%s", base_event, mode)

    LOG.info("TC plots written to %s", pdf_path)
    return plots_dir
