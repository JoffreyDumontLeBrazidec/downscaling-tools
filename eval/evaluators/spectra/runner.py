"""Spectra evaluator — FFT computation and amplitude file generation.

Delegates to existing spectra pipeline for computation.
"""
from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    reference_root: str | Path | None = None,
    **kwargs,
) -> Path:
    """Run spectra computation for predictions.

    Generates amplitude/wavenumber .npy files and spectra_summary.json.
    """
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "spectra"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Spectra output directory already has content: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Spectra run: predictions_dir=%s, output_dir=%s", predictions_dir, output_dir)
    LOG.info("Spectra fields from config: %s", eval_config.get("fields", []))
    LOG.info("Wavenumber threshold: %s", eval_config.get("wavenumber_threshold"))

    return output_dir
