"""Leadtime evaluator — scalar metrics for the run summary.

The leadtime evaluator is diagnostic (not folded into the scoreboard composite),
but we emit a handful of scalar metrics so they appear in the run's metrics.json
and can be compared across experiments in logs/reports.

Emitted metrics (one per step, global region only):
  leadtime_{step}h_surface_nmse_full      — global area-weighted mean nMSE
  leadtime_{step}h_surface_nmse_residual  — same on the residual (correction) space
  leadtime_{step}h_skill_vs_interp        — skill vs bilinear-interp baseline
  leadtime_{step}h_spectra_rel_l2         — mean spectral relative-L2 (if available)
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

# Canonical surface variable weights (same as _surface_compute.SURFACE_VARIABLES).
_SURFACE_WEIGHTS: dict[str, float] = {
    "10u": 2.5, "10v": 2.5, "2d": 2.0, "2t": 2.0,
    "msl": 2.0, "skt": 0.5, "sp": 1.5, "tcw": 1.0,
}


def _weighted_mean(
    by_var: dict[str, dict[str, float]],
    metric: str,
) -> float | None:
    total_w, total_v = 0.0, 0.0
    for var, w in _SURFACE_WEIGHTS.items():
        entry = by_var.get(var, {})
        v = entry.get(metric)
        if v is None or not math.isfinite(v):
            continue
        total_v += v * w
        total_w += w
    return total_v / total_w if total_w > 0 else None


def _spectra_mean_rel_l2(spectra_step: dict[str, Any], lmax: int = 319) -> float | None:
    """Mean relative-L2 over spectra_vars, wavenumbers > 100."""
    import numpy as np

    ell = np.arange(lmax + 1, dtype=np.float64)
    band = ell > 100.0
    values = []
    for var, curves in spectra_step.items():
        pred = curves.get("pred_cl")
        truth = curves.get("truth_cl")
        if pred is None or truth is None:
            continue
        p = np.asarray(pred)
        t = np.asarray(truth)
        n = min(len(p), len(t), len(ell))
        m = band[:n] & np.isfinite(p[:n]) & np.isfinite(t[:n]) & (t[:n] > 0)
        if not m.any():
            continue
        rl2 = float(np.linalg.norm(p[:n][m] - t[:n][m]) / max(np.linalg.norm(t[:n][m]), 1e-12))
        values.append(rl2)
    return float(np.mean(values)) if values else None


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    **kwargs,
) -> list[dict[str, Any]]:
    """Return per-step scalar metrics from leadtime_scores.json."""
    results_dir = Path(results_dir)
    json_path = results_dir / "leadtime_scores.json"
    if not json_path.exists():
        LOG.warning("leadtime_scores.json not found in %s — no metrics emitted", results_dir)
        return []

    data = json.loads(json_path.read_text())
    steps: list[int] = data.get("steps", [])
    lmax: int = int(data.get("lmax", 319))
    spectra: dict = data.get("spectra", {})

    records: list[dict[str, Any]] = []
    for step in steps:
        key = str(step)
        global_by_var: dict[str, dict[str, float]] = data.get("by_step", {}).get(key, {}).get("global", {})

        for metric_key, metric_label in [
            ("nmse_full", "surface_nmse_full"),
            ("nmse_residual", "surface_nmse_residual"),
            ("skill_vs_interp", "skill_vs_interp"),
        ]:
            val = _weighted_mean(global_by_var, metric_key)
            if val is not None and math.isfinite(val):
                records.append({
                    "metric": f"leadtime_{step}h_{metric_label}",
                    "value": val,
                    "unit": "nmse" if "nmse" in metric_key else "score_0_1",
                })

        spec_rl2 = _spectra_mean_rel_l2(spectra.get(key, {}), lmax=lmax)
        if spec_rl2 is not None:
            records.append({
                "metric": f"leadtime_{step}h_spectra_rel_l2",
                "value": spec_rl2,
                "unit": "relative_l2",
            })

    return records
