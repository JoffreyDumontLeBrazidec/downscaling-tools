"""Evaluator runner for local spread/CRPS diagnostics."""
from __future__ import annotations

import logging
from pathlib import Path

from eval.evaluators.probabilistic.core import compute_probabilistic_scores
from eval.evaluators.probabilistic.core.calibration import RANK_TIE_SEED
from eval.evaluators.probabilistic.core.scale_split import DEFAULT_CUT_KM, DEFAULT_GRID_KM
from eval.evaluators.probabilistic.core.scoring import (
    DEFAULT_BOX_EDGE_DEG,
    DEFAULT_DOMAINS,
    DEFAULT_WEATHER_STATES,
)

LOG = logging.getLogger(__name__)


def _or(value, fallback):
    """A lane key set to null means the default."""
    return fallback if value is None else value


def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    **kwargs,
) -> Path:
    """Compute probabilistic scores from local prediction NetCDFs.

    Lane keys: ``weather_states``, ``domains``, ``steps``, ``dates``, ``spread_ddof``;
    ``scale_split`` (bool, default off: costs about three times the full-field scoring),
    ``scale_cut_km`` (400), ``scale_grid_km`` (25), ``scale_edge_margin_km`` (default the
    cut), ``box_edge_deg`` (1: strip left out of every band on a regional box);
    ``rank_histogram`` (bool, default on), ``rank_tie_seed``; ``include_input`` (bool,
    default on: the input members' spread when the files carry ``x_interp`` members).
    """
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "probabilistic"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary_by_lead.csv"
    if summary_path.exists() and not overwrite:
        LOG.info("probabilistic summary already exists, skipping: %s", summary_path)
        return output_dir

    weather_states = _or(eval_config.get("weather_states"), list(DEFAULT_WEATHER_STATES))
    domains = _or(eval_config.get("domains"), list(DEFAULT_DOMAINS))
    steps = eval_config.get("steps")
    dates = eval_config.get("dates")
    spread_ddof = int(_or(eval_config.get("spread_ddof"), 1))
    scale_split = bool(_or(eval_config.get("scale_split"), False))
    scale_cut_km = float(_or(eval_config.get("scale_cut_km"), DEFAULT_CUT_KM))
    scale_grid_km = float(_or(eval_config.get("scale_grid_km"), DEFAULT_GRID_KM))
    scale_edge_margin_km = eval_config.get("scale_edge_margin_km")
    box_edge_deg = float(_or(eval_config.get("box_edge_deg"), DEFAULT_BOX_EDGE_DEG))
    rank_histogram = bool(_or(eval_config.get("rank_histogram"), True))
    rank_tie_seed = int(_or(eval_config.get("rank_tie_seed"), RANK_TIE_SEED))
    include_input = bool(_or(eval_config.get("include_input"), True))

    payload = compute_probabilistic_scores(
        predictions_dir,
        output_dir,
        weather_states=weather_states,
        domains=domains,
        steps=steps,
        dates=dates,
        spread_ddof=spread_ddof,
        scale_split=scale_split,
        scale_cut_km=scale_cut_km,
        scale_grid_km=scale_grid_km,
        scale_edge_margin_km=None if scale_edge_margin_km is None else float(scale_edge_margin_km),
        box_edge_deg=box_edge_deg,
        rank_histogram=rank_histogram,
        rank_tie_seed=rank_tie_seed,
        include_input=include_input,
    )
    LOG.info(
        "Probabilistic scores written to %s (%s rows, %s band rows, scale split %s)",
        output_dir,
        payload.get("n_rows"),
        payload.get("n_band_rows"),
        "on" if scale_split else "off",
    )
    return output_dir
