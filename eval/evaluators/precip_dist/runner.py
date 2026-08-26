"""precip_dist evaluator subprocess wrapper around tp_histogram_comparison.

Reads lane_config[precip_dist] for tunables and lane_config[precip] for the
truth/baseline GRIB fallbacks (used when the predictions embed no tp truth /
no usable x_interp tp — the o1280->o2560 main-lane case).
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
    checkpoint: str | None = None,
    **kwargs,
) -> Path:
    """Run tp distribution histograms by subprocessing into tp_histogram_comparison."""
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "precip_dist"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"precip_dist output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = find_predictions(predictions_dir)
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = plots_dir / "tp_histograms.pdf"
    run_label = eval_config.get("run_label", "")
    ensemble_member_index = int(eval_config.get("ensemble_member_index", 0))
    style = str(eval_config.get("style", "compact"))
    precip_cfg = dict(lane_config.get("precip", {}))

    cmd = [
        sys.executable, "-m", "eval._backends.precip.tp_histogram_comparison",
        "--predictions-dir", str(predictions_dir),
        "--out-pdf", str(out_pdf),
        "--ensemble-member-index", str(ensemble_member_index),
        "--style", style,
    ]
    if precip_cfg.get("truth_grib_tpl"):
        cmd += ["--truth-grib-tpl", str(precip_cfg["truth_grib_tpl"])]
    if precip_cfg.get("baseline_lres_grib_tpl"):
        cmd += ["--baseline-grib-tpl", str(precip_cfg["baseline_lres_grib_tpl"])]
    if precip_cfg.get("interp_index_cache"):
        cmd += ["--interp-index-cache", str(precip_cfg["interp_index_cache"])]
    if run_label:
        cmd += ["--run-label", str(run_label)]

    LOG.info("precip_dist subprocess: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_dir
