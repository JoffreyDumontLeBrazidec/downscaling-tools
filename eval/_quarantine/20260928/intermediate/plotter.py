"""Intermediate evaluator — plot-only re-entry from cached netCDF.

Used by `eval.cli evaluate --plot-only`. Reads the existing
`inter_states_<ckpt>.nc` from results_dir and re-renders the consolidated
multi-page PDF without invoking the GPU compute step.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .runner import render_only

LOG = logging.getLogger(__name__)


def plot(
    results_dir,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir=None,
) -> Path:
    out = Path(output_dir) if output_dir else Path(results_dir)
    LOG.info("Intermediate plot (re-render): results_dir=%s", out)
    return render_only(out, lane_config, eval_config)
