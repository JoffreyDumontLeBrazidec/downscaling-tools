"""Leadtime evaluator — compute and persist per-step scores."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from eval._backends.leadtime.compute import compute_leadtime_scores

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
    """Compute per-step surface scores and spectra; write leadtime_scores.json."""
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "leadtime"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "leadtime_scores.json"
    if json_path.exists() and not overwrite:
        LOG.info("leadtime_scores.json already exists, skipping: %s", json_path)
        return output_dir

    nside = int(eval_config.get("nside", 128))
    lmax = int(eval_config.get("lmax", 319))
    skip_spectra = bool(eval_config.get("skip_spectra", False))
    surface_vars = eval_config.get("surface_vars") or None
    spectra_vars = eval_config.get("spectra_vars") or None

    scores = compute_leadtime_scores(
        predictions_dir,
        nside=nside,
        lmax=lmax,
        skip_spectra=skip_spectra,
        surface_vars=surface_vars,
        spectra_vars=spectra_vars,
    )

    json_path.write_text(json.dumps(scores, indent=2, default=str) + "\n")
    LOG.info("Leadtime scores written to %s", json_path)
    return output_dir
