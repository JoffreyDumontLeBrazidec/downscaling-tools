"""Region plot evaluator — discover predictions, extract regions.

Delegates to eval.discovery.predictions for file finding and
eval.region_plotting for actual rendering.
"""
from __future__ import annotations

import logging
from pathlib import Path

from eval.discovery.predictions import find_predictions

LOG = logging.getLogger(__name__)


def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    **kwargs,
) -> Path:
    """Run region plot evaluation — discover predictions and prepare for rendering."""
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "region_plot"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Region plot output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = find_predictions(predictions_dir)
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")

    regions = lane_config.get("regions", {}).get("interesting", {})
    LOG.info("Region plot: %d predictions, %d regions configured", len(pred_files), len(regions))

    return output_dir
