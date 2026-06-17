"""sigma_loss evaluator — scorer.

Reads data/sigma_loss/per_sigma.csv + meta.json and emits scoreboard metrics:
  * loss_at_sigma_data   — total F-space loss at the sigma grid point nearest sigma_data
  * mean_loss_extreme    — mean total loss over the lane extreme band
  * mean_loss_fine       — mean total loss over the fine band (~[0.05, 0.3])
  * argmin_sigma         — sigma minimising the total F-space loss

Also writes data/sigma_loss/metrics.json (same records). Returns the records so
the eval.cli scoreboard aggregator can pick them up.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

DATA_SUBDIR = ("data", "sigma_loss")


def _data_dir(results_dir: Path) -> Path:
    return results_dir.joinpath(*DATA_SUBDIR)


def _load_total_curve(csv_path: Path) -> list[tuple[float, float]]:
    """Return sorted [(sigma, total_fspace_loss)] from the __total__ rows."""
    pts: list[tuple[float, float]] = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row["variable"] != "__total__":
                continue
            try:
                pts.append((float(row["sigma"]), float(row["fspace_loss"])))
            except (TypeError, ValueError):
                continue
    pts.sort(key=lambda p: p[0])
    return pts


def _band_mean(curve: list[tuple[float, float]], lo: float, hi: float) -> float | None:
    vals = [v for s, v in curve if lo <= s <= hi]
    return sum(vals) / len(vals) if vals else None


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    results_dir = Path(results_dir)
    data_dir = _data_dir(results_dir)
    csv_path = data_dir / "per_sigma.csv"
    meta_path = data_dir / "meta.json"
    if not csv_path.exists():
        LOG.warning("sigma_loss scorer: no per_sigma.csv at %s", csv_path)
        return []

    curve = _load_total_curve(csv_path)
    if not curve:
        LOG.warning("sigma_loss scorer: no __total__ rows in %s", csv_path)
        return []

    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}
    sigma_data = float(meta.get("sigma_data", 1.0))

    # Bands: configurable, defaults match the lane's lognormal heavy-tail regime.
    extreme_band = eval_config.get("extreme_band", [80.0, 500.0])
    fine_band = eval_config.get("fine_band", [0.05, 0.3])

    records: list[dict[str, Any]] = []

    # loss at the grid sigma nearest sigma_data
    nearest = min(curve, key=lambda p: abs(p[0] - sigma_data))
    records.append({
        "metric": "sigma_loss_at_sigma_data",
        "value": float(nearest[1]),
        "unit": "fspace_mse",
    })

    mean_ext = _band_mean(curve, float(extreme_band[0]), float(extreme_band[1]))
    if mean_ext is not None:
        records.append({
            "metric": "sigma_loss_mean_extreme",
            "value": float(mean_ext),
            "unit": "fspace_mse",
        })

    mean_fine = _band_mean(curve, float(fine_band[0]), float(fine_band[1]))
    if mean_fine is not None:
        records.append({
            "metric": "sigma_loss_mean_fine",
            "value": float(mean_fine),
            "unit": "fspace_mse",
        })

    argmin = min(curve, key=lambda p: p[1])
    records.append({
        "metric": "sigma_loss_argmin_sigma",
        "value": float(argmin[0]),
        "unit": "sigma",
    })

    metrics_path = data_dir / "metrics.json"
    metrics_path.write_text(json.dumps(records, indent=2, default=str) + "\n")
    LOG.info("sigma_loss scorer: wrote %d metrics -> %s", len(records), metrics_path)

    return records
