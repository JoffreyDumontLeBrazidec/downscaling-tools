"""TC evaluator visualization: one overview PDF page per event/support."""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from eval._backends.tc.pdf_plot import plot_pdf_distribution_overview, plot_pdf_log, plot_pdf_ratios
from eval._backends.tc.plot_config import resolve_plot_config
from eval.evaluators.tc.comparison_contract import validate_comparison_contracts

LOG = logging.getLogger(__name__)


def _ordered_event_stats(events_data: dict, eval_config: dict):
    """Return events in the stable native/regridded Franklin/Idalia report order."""
    by_event_and_mode: dict[tuple[str, str], dict] = {}
    for event_stats in events_data.values():
        if event_stats.get("prediction_only"):
            continue
        event = str(event_stats.get("event", ""))
        mode = str(event_stats.get("support_mode", ""))
        by_event_and_mode[(event, mode)] = event_stats

    configured = [str(event) for event in eval_config.get("events", [])]
    event_order = ["franklin", "idalia"]
    event_order.extend(event for event in configured if event not in event_order)
    event_order.extend(
        event for event, _mode in by_event_and_mode
        if event not in event_order
    )

    ordered: list[dict] = []
    for event in event_order:
        modes = ["native", "regridded"]
        modes.extend(mode for current_event, mode in by_event_and_mode if current_event == event and mode not in modes)
        for mode in modes:
            event_stats = by_event_and_mode.get((event, mode))
            if event_stats is not None:
                ordered.append(event_stats)
    return ordered


def _validate_event_contract(event_stats: dict) -> None:
    candidate = event_stats.get("comparison_contract")
    reference = event_stats.get("reference_comparison_contract")
    if not isinstance(candidate, dict) or not isinstance(reference, dict):
        event = event_stats.get("event", "unknown")
        mode = event_stats.get("support_mode", "unknown")
        raise ValueError(
            f"TC plot for event={event!r} mode={mode!r} has no comparison contract. "
            "Re-run eval.cli evaluate without --plot-only to rebuild validated statistics."
        )
    validate_comparison_contracts({"prediction": candidate, "reference": reference})


def plot(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    stats_filename: str = "stats.json",
) -> Path:
    """Write a compact overview-style TC-distribution PDF from saved event statistics."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    stats_path = results_dir / stats_filename
    if not stats_path.exists():
        raise FileNotFoundError(f"TC stats file not found: {stats_path}")

    with open(stats_path) as f:
        events_data = json.load(f).get("events", {})
    if not events_data:
        LOG.warning("No event data in %s", stats_path)
        return plots_dir

    ordered_events = _ordered_event_stats(events_data, eval_config)
    for event_stats in ordered_events:
        _validate_event_contract(event_stats)

    pdf_path = plots_dir / "all_tc_distributions.pdf"
    with PdfPages(pdf_path) as pdf:
        for event_stats in ordered_events:
            event = str(event_stats["event"])
            mode = str(event_stats["support_mode"])
            plot_cfg = resolve_plot_config(event, eval_config)
            plot_cfg = replace(plot_cfg, plot_title=f"{plot_cfg.plot_title.replace('normed pdfs', 'TC distributions')} [{mode}]")
            fig = plot_pdf_distribution_overview(plot_cfg, event_stats=event_stats)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
            LOG.info("Plotted overview TC distribution for event=%s mode=%s", event, mode)

    LOG.info("TC plots written to %s", pdf_path)
    return plots_dir
