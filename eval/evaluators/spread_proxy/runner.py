"""Evaluator runner for the ML-vs-ENFO spread proxy."""
from __future__ import annotations

import logging
from pathlib import Path

from eval._backends.spread_proxy import compute_spread_proxy

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
    """Compare ML vs ENFO ensemble spread from local prediction NetCDFs."""
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "spread_proxy"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary_by_lead.csv"
    if summary_path.exists() and not overwrite:
        LOG.info("spread_proxy summary already exists, skipping: %s", summary_path)
        return output_dir

    payload = compute_spread_proxy(
        predictions_dir,
        output_dir,
        weather_states=eval_config.get("weather_states"),
        domains=eval_config.get("domains"),
        steps=eval_config.get("steps"),
        dates=eval_config.get("dates"),
        spread_ddof=int(eval_config.get("spread_ddof", 1)),
        map_bin_deg=float(eval_config.get("map_bin_deg", 0.5)),
        spectra=bool(eval_config.get("spectra", True)),
        spectra_fields=eval_config.get("spectra_fields"),
        spectra_steps=eval_config.get("spectra_steps"),
        spectra_nside=int(eval_config.get("spectra_nside", 256)),
        enfo_exclude_members=eval_config.get("enfo_exclude_members"),
        enfo_n_members=eval_config.get("enfo_n_members"),
        enfo_subsample_seed=int(eval_config.get("enfo_subsample_seed", 0)),
        include_input=bool(eval_config.get("include_input", True)),
    )
    LOG.info("spread_proxy scores written to %s (%s rows)", output_dir, payload.get("n_rows"))
    return output_dir
