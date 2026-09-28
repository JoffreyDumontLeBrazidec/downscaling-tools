"""Observation CRPS evaluator — plot phase."""
from __future__ import annotations

import logging
from pathlib import Path

from eval._backends.obs_crps.plotting import plot_obs_crps_summary

LOG = logging.getLogger(__name__)


def plot(results_dir, lane_config, eval_config, *, output_dir=None, **kwargs) -> Path:
    results_dir = Path(results_dir)
    if (results_dir / "skipped.json").exists():
        LOG.info("obs_crps: run was skipped, nothing to plot")
        return results_dir
    output_dir = Path(output_dir) if output_dir else results_dir
    summary_csv = results_dir / "summary_by_lead.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing obs_crps summary CSV: {summary_csv}")
    expver = (eval_config or {}).get("expver", "")
    plot_obs_crps_summary(
        summary_csv,
        results_dir / "scores_by_date_and_lead.csv",
        output_dir / "plots" / "obs_crps_lead_curves.pdf",
        title_suffix=f" — {expver}" if expver else "",
    )
    return output_dir
