"""Runner for local/global prediction parity checks."""
from __future__ import annotations

import logging
from pathlib import Path

from eval._backends.local_global_parity import compute_parity

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
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "local_global"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "local_global_parity.json"
    if summary_path.exists() and not overwrite:
        LOG.info("local/global parity already exists, skipping: %s", summary_path)
        return output_dir

    global_predictions_dir = eval_config.get("global_predictions_dir")
    if not global_predictions_dir:
        raise ValueError("local_global evaluator requires local_global.global_predictions_dir")

    variables = eval_config.get("variables")
    if isinstance(variables, str):
        variables = [item.strip() for item in variables.split(",") if item.strip()]

    compute_parity(
        local_predictions_dir=predictions_dir,
        global_predictions_dir=global_predictions_dir,
        output_dir=output_dir,
        coordinate_tolerance=float(eval_config.get("coordinate_tolerance", 1e-6)),
        variables=variables,
    )
    LOG.info("Local/global parity written to %s", summary_path)
    return output_dir
