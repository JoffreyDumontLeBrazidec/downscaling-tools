"""Evaluator runner for local spread/CRPS diagnostics."""
from __future__ import annotations

import logging
from pathlib import Path

from eval._backends.probabilistic import compute_probabilistic_scores

LOG = logging.getLogger(__name__)


def _config_value(eval_config: dict, key: str, fallback=None):
    value = eval_config.get(key, fallback)
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
    """Compute probabilistic scores from local prediction NetCDFs."""
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "probabilistic"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary_by_lead.csv"
    if summary_path.exists() and not overwrite:
        LOG.info("probabilistic summary already exists, skipping: %s", summary_path)
        return output_dir

    weather_states = _config_value(
        eval_config,
        "weather_states",
        ["2t", "10ff", "2d", "msl", "t_850", "z_500"],
    )
    domains = _config_value(eval_config, "domains", ["n.hem", "tropics", "s.hem", "europe"])
    steps = eval_config.get("steps")
    dates = eval_config.get("dates")
    spread_ddof = int(eval_config.get("spread_ddof", 1))

    payload = compute_probabilistic_scores(
        predictions_dir,
        output_dir,
        weather_states=weather_states,
        domains=domains,
        steps=steps,
        dates=dates,
        spread_ddof=spread_ddof,
    )
    LOG.info(
        "Probabilistic scores written to %s (%s rows)",
        output_dir,
        payload.get("n_rows"),
    )
    return output_dir
