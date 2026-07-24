"""Quaver evaluator — score phase.

Quaver writes its verification scores into its own offline database, which the
plot phase reads to render the scorecards. There is no lightweight per-field
metric to surface to the downscaling-tools scoreboard here, so score() returns
no metrics; the deliverable is the CRPS/spread scorecard PDFs from plot().
"""
from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


def score(results_dir, lane_config, eval_config, **kwargs):
    results_dir = Path(results_dir)
    if (results_dir / "skipped.json").exists():
        return []
    return []
