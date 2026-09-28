"""TC evaluator visualization: one overview PDF page per event/support."""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

from eval._backends.tc.pdf_plot import plot_pdf_distribution_overview, plot_pdf_log, plot_pdf_ratios
from eval._backends.tc.plot_config import resolve_plot_config
from eval.evaluators.tc.comparison_contract import validate_comparison_contracts
from eval.plotting import FigureBook, readable_label

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


def _bundle_source_names(lane_config: dict) -> tuple[str | None, str | None]:
    """Readable names ("EEFO O320", "ENFO O1280") of the lane's input and target GRIB products."""
    from eval.evaluators.tc.runner import _expid_from_grib_path

    args = ((lane_config or {}).get("prepare") or {}).get("args") or {}
    names = []
    for key in ("lres_sfc_grib", "target_sfc_grib"):
        expid = _expid_from_grib_path(args.get(key))
        names.append(readable_label(expid) if expid else None)
    return names[0], names[1]


def _analysis_label(analysis_key: str) -> str:
    """Readable name of an analysis curve that is not the truth, e.g. the operational analysis."""
    text = readable_label(analysis_key)
    if text.startswith("operational analysis"):
        return text
    low = str(analysis_key).lower()
    if low.startswith(("oper-an", "oper an", "oper_", "oper ")):
        return f"operational analysis ({str(analysis_key).replace('_', ' ')})"
    return f"analysis ({text})"


def curve_labels_and_roles(event_stats: dict, lane_config: dict, eval_config: dict):
    """Legend labels and roles for the curves of one event, from the lane configuration.

    The truth is always the bundle target (``target_nc_label``, the ENFO field the model is
    trained towards). On native-support lanes the analysis the statistics are normalised by
    IS that target, so it is the truth; on regridded lanes the analysis is the operational
    analysis (OPER-AN), a reference, and the target is a separate curve that is the truth.
    The bundle's input (``input_label``) is the input; the remaining non-reference curve is the
    model run, whatever key it was stored under (for example ``eval_inputs``).
    """
    input_name, target_name = _bundle_source_names(lane_config)
    analysis_key = event_stats.get("analysis_key")
    order = list(event_stats.get("curve_order", []))
    input_label = eval_config.get("input_label", "input")
    target_keys = {eval_config.get("target_nc_label"), eval_config.get("target_label")} - {None}
    other_target_like = {eval_config.get("analysis_display_label")} - {None}
    labels: dict[str, str] = {}
    roles: dict[str, str] = {}

    truth_key = None
    if analysis_key in target_keys:
        truth_key = analysis_key
    else:
        truth_key = next((k for k in order if k in target_keys), None)
    if truth_key is not None:
        roles[truth_key] = "truth"
        labels[truth_key] = f"truth ({target_name})" if target_name else "truth"
    if analysis_key and analysis_key != truth_key:
        roles[analysis_key] = "reference"
        labels[analysis_key] = _analysis_label(analysis_key)
    for key in order:
        if key == truth_key or key == analysis_key:
            continue
        if key == input_label:
            roles[key] = "input"
            if input_name:
                labels[key] = f"input ({input_name})"
        elif key in target_keys or key in other_target_like:
            roles[key] = "reference"
            if target_name:
                labels[key] = f"{target_name} target"
    run_label = eval_config.get("run_label")
    if run_label and run_label in order:
        roles.setdefault(run_label, "model")
    return labels, roles


def _page_title(plot_cfg, event: str, mode: str) -> str:
    base = plot_cfg.plot_title.replace("normed pdfs", "").strip() or event.capitalize()
    return f"{base}: TC distributions of grid-point values in the event box ({mode} support)"


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
    with FigureBook(pdf_path, png=True) as book:
        for event_stats in ordered_events:
            event = str(event_stats["event"])
            mode = str(event_stats["support_mode"])
            plot_cfg = resolve_plot_config(event, eval_config)
            plot_cfg = replace(plot_cfg, plot_title=_page_title(plot_cfg, event, mode))
            labels, roles = curve_labels_and_roles(event_stats, lane_config, eval_config)
            fig = plot_pdf_distribution_overview(plot_cfg, event_stats=event_stats,
                                                 exp_labels=labels, curve_roles=roles)
            book.add(fig, name=f"{event}_{mode}")
            LOG.info("Plotted overview TC distribution for event=%s mode=%s", event, mode)

    LOG.info("TC plots written to %s", pdf_path)
    return plots_dir
