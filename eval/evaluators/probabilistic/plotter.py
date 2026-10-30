"""Plotter for local probabilistic diagnostics."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from eval.evaluators.probabilistic.core import (
    plot_band_scores,
    plot_probabilistic_summary,
    plot_rank_histograms,
)

LOG = logging.getLogger(__name__)


def plot(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    **kwargs,
) -> Path:
    """Render the lead-time curves, the rank histograms and, after a scale split, the band curves."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    plots_dir = output_dir / "plots"
    summary_csv = results_dir / "summary_by_lead.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing probabilistic summary CSV: {summary_csv}")
    out_pdf = plots_dir / "probabilistic_scores.pdf"
    reference_curves = eval_config.get("reference_curves")
    title = eval_config.get("title", "Probabilistic scores")
    plot_probabilistic_summary(
        summary_csv,
        out_pdf,
        title_prefix=title,
        reference_curves=reference_curves,
        lane_config=lane_config,
    )
    LOG.info("Probabilistic plot written to %s", out_pdf)

    # the two new figures read outputs an older run may not have; they skip quietly then
    hist_pdf = plot_rank_histograms(results_dir / "rank_histograms.csv", plots_dir / "rank_histograms.pdf",
                                    title=None if title == "Probabilistic scores" else title)
    if hist_pdf is not None:
        LOG.info("Rank histograms written to %s", hist_pdf)
    cut_km = None
    summary_json = results_dir / "probabilistic_summary.json"
    if summary_json.exists():
        try:
            cut_km = (json.loads(summary_json.read_text()).get("scale_split") or {}).get("cut_km")
        except ValueError:
            cut_km = None
    band_pdf = plot_band_scores(results_dir / "calibration_by_lead.csv", plots_dir / "band_scores.pdf",
                                lane_config=lane_config, title=None if title == "Probabilistic scores" else title,
                                cut_km=cut_km)
    if band_pdf is not None:
        LOG.info("Band scores written to %s", band_pdf)
    return out_pdf
