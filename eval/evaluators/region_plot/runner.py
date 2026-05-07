"""Region-plot evaluator subprocess wrapper around eval.region_plotting.plot_regions.

The legacy module remains the canonical implementation. This runner translates
EvaluatorContext values into the legacy CLI argv shape.
"""
from __future__ import annotations

import logging
import subprocess
import sys
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
    """Run region plotting by subprocessing into plot_regions."""
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "region_plot"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Region plot output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = find_predictions(predictions_dir)
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")

    cmd = [
        sys.executable, "-m", "eval._legacy_kernels.region_plotting.plot_regions",
        "--predictions-nc", str(pred_files[0].path),
        "--out-dir", str(output_dir),
    ]

    region_names = eval_config.get("region_names")
    if region_names:
        cmd += ["--regions", ",".join(region_names)]

    LOG.info("region_plot subprocess: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_dir
