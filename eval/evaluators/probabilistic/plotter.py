"""Plotter for local probabilistic diagnostics."""
from __future__ import annotations

import logging
from pathlib import Path

from eval._backends.probabilistic import plot_probabilistic_summary

LOG = logging.getLogger(__name__)


def plot(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    **kwargs,
) -> Path:
    """Render quaver-style lead-time curves from summary_by_lead.csv."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    plots_dir = output_dir / "plots"
    summary_csv = results_dir / "summary_by_lead.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing probabilistic summary CSV: {summary_csv}")
    out_pdf = plots_dir / "probabilistic_scores.pdf"
    reference_curves = eval_config.get("reference_curves")
    plot_probabilistic_summary(
        summary_csv,
        out_pdf,
        title_prefix=eval_config.get("title", "Probabilistic scores"),
        reference_curves=reference_curves,
    )
    LOG.info("Probabilistic plot written to %s", out_pdf)
    return out_pdf
