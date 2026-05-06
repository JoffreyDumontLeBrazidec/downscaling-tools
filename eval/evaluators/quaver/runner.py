"""Quaver evaluator — CRPS / rank histogram computation.

Delegates to eval.quaver for the actual computation.
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
    **kwargs,
) -> Path:
    """Run probabilistic metric computation."""
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "quaver"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Quaver output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Quaver run: predictions_dir=%s, output_dir=%s", predictions_dir, output_dir)
    return output_dir
