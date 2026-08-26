"""Plotter for the ML-vs-ENFO spread proxy."""
from __future__ import annotations

import logging
from pathlib import Path

from eval._backends.spread_proxy import plot_all

LOG = logging.getLogger(__name__)


def plot(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    **kwargs,
) -> Path:
    """Render spread curves, ratio maps, and spread spectra PDFs."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    plots_dir = output_dir / "plots"
    written = plot_all(
        results_dir, plots_dir,
        title_prefix=eval_config.get("title", "Spread proxy"),
    )
    LOG.info("spread_proxy plots written: %s", [str(p) for p in written])
    return plots_dir
