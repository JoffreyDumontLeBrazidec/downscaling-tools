"""ECMWF spectra evaluator, version two: scoring.

Each prediction curve the runner wrote (``spectra/<field_dir>/ampl_*.npy``) is
paired with the truth curve of the same date, step and member in the truth
reference the runner recorded in ``spectra_summary.json``
(``reference_spectra_dir``, which is ``<reference_dir>/truth/<window>/spectra``).
Per variable, the paired curves are averaged over dates, steps and members, and
the two mean curves are compared.

The score definition is the one the retired HEALPix proxy (``spectra``) used:

* relative L2 error = ||P - T|| / ||T|| over the wavenumbers above 100 where both
  mean curves are finite and positive (P = prediction mean curve, T = truth mean
  curve, no weighting). On a grid whose truncation does not reach past 100 (the
  O96 lane, T95) the band starts at one third of the largest wavenumber instead,
  which is the rule of ``eval.evaluators.spectra_ecmwf_v2.core.scoreboard.relative_l2``;
* per-variable score = max(0, 1 - relative L2 error);
* mean relative L2 error = the average over the variables that were scored,
  reported only when at least three were, and mean score = max(0, 1 - that mean).

The rows carry NEW metric names, ``spectra_v2_<variable>_relative_l2``,
``spectra_v2_<variable>_score``, ``spectra_v2_mean_relative_l2`` and
``spectra_v2_mean_score``, so they can never be read as the proxy's
``spectra_<variable>_*`` rows. The two instruments give different numbers for
the same run; they are not interchangeable. ``<variable>`` is the weather state
as the runner names it (``msl`` or ``sp`` for pressure, as the lane chooses).

Returns no rows, with a warning, when the run has no truth reference (the lane
section has no ``reference_dir``, or the reference was not computed).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from eval.evaluators.spectra_ecmwf_v2.core.scoreboard import (
    SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE,
    relative_l2,
    spectra_score,
)
from eval.evaluators.spectra_ecmwf_v2.core import naming

LOG = logging.getLogger(__name__)

METRIC_PREFIX = "spectra_v2"
MIN_VARIABLES_FOR_MEAN = 3


def _band_start(wavenumbers: np.ndarray) -> float:
    """The exclusive lower wavenumber of the scored band (mirrors relative_l2)."""
    max_wvn = float(np.nanmax(wavenumbers)) if wavenumbers.size else 0.0
    if max_wvn <= SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE:
        return max_wvn / 3.0
    return SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE


def score_curve_pair(
    prediction: np.ndarray, truth: np.ndarray, wavenumbers: np.ndarray
) -> float:
    """Relative L2 error of one prediction mean curve against one truth mean curve."""
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    wavenumbers = np.asarray(wavenumbers, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.shape != wavenumbers.shape:
        raise ValueError(
            f"curve length mismatch: prediction={prediction.shape} "
            f"truth={truth.shape} wavenumbers={wavenumbers.shape}"
        )
    return relative_l2(prediction, truth, wavenumbers=wavenumbers)


def _load_summary(results_dir: Path) -> dict:
    path = results_dir / "spectra_summary.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _pair_curves(
    files: list[dict], truth_root: Path, results_dir: Path
) -> dict[str, dict[str, Any]]:
    """Group prediction/truth curve pairs by weather state."""
    by_state: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict):
            continue
        ampl_path = Path(str(entry.get("amplitudes", "")))
        wvn_path = Path(str(entry.get("wavenumbers", "")))
        parsed = naming.parse(ampl_path.name)
        if parsed is None:
            continue
        field_dir = str(parsed["field_dir"])
        if not ampl_path.exists() or not wvn_path.exists():
            # The summary records absolute paths; follow a moved run directory.
            ampl_path = results_dir / "spectra" / field_dir / ampl_path.name
            wvn_path = results_dir / "spectra" / field_dir / wvn_path.name
            if not ampl_path.exists() or not wvn_path.exists():
                continue
        state = str(entry.get("weather_state") or parsed["weather_state"])
        token = str(parsed["token"] or naming.DEFAULT_TOKEN)
        slot = by_state.setdefault(
            state, {"pred": [], "truth": [], "wvn": None, "missing_truth": 0}
        )
        truth_path = naming.find(
            truth_root / field_dir, "ampl",
            date=parsed["date"], step=parsed["step"], field_dir=field_dir,
            member=parsed["member"], token=token,
        )
        if truth_path is None:
            slot["missing_truth"] += 1
            continue
        pred = np.asarray(np.load(ampl_path), dtype=np.float64)
        truth = np.asarray(np.load(truth_path), dtype=np.float64)
        wvn = np.asarray(np.load(wvn_path), dtype=np.float64)
        if pred.shape != truth.shape or pred.shape != wvn.shape:
            LOG.warning(
                "spectra_ecmwf_v2 scorer: %s and its truth %s differ in length; skipped",
                ampl_path.name, truth_path,
            )
            slot["missing_truth"] += 1
            continue
        if slot["wvn"] is None:
            slot["wvn"] = wvn
        elif slot["wvn"].shape != wvn.shape or not np.allclose(slot["wvn"], wvn):
            raise ValueError(f"spectra_ecmwf_v2 scorer: wavenumbers differ across {state} curves")
        slot["pred"].append(pred)
        slot["truth"].append(truth)
    return by_state


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    **kwargs,
) -> list[dict[str, Any]]:
    """Score the run's spectra against its truth reference.

    Returns ``{"metric", "value", "unit"}`` records (see the module docstring for
    the metric names), and writes ``spectra_v2_scores.json`` beside them with the
    per-variable curve counts and the scored wavenumber band.
    """
    results_dir = Path(results_dir)
    summary = _load_summary(results_dir)
    truth_dir = str(summary.get("reference_spectra_dir") or "").strip()
    if not truth_dir:
        LOG.warning(
            "spectra_ecmwf_v2 scorer: %s records no truth reference "
            "(set spectra_ecmwf_v2.reference_dir in the lane); no scoreboard rows.",
            results_dir,
        )
        return []
    truth_root = Path(truth_dir)
    if not truth_root.is_dir():
        LOG.warning(
            "spectra_ecmwf_v2 scorer: truth reference %s does not exist; no scoreboard rows.",
            truth_root,
        )
        return []

    by_state = _pair_curves(summary.get("files") or [], truth_root, results_dir)
    records: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    values: list[float] = []
    for state in sorted(by_state):
        slot = by_state[state]
        n = len(slot["pred"])
        details[state] = {"n_pairs": n, "missing_truth": slot["missing_truth"]}
        if not n:
            continue
        pred_mean = np.nanmean(np.stack(slot["pred"]), axis=0)
        truth_mean = np.nanmean(np.stack(slot["truth"]), axis=0)
        rel = score_curve_pair(pred_mean, truth_mean, slot["wvn"])
        details[state]["wavenumber_min_exclusive"] = _band_start(slot["wvn"])
        if not math.isfinite(rel):
            continue
        details[state]["relative_l2"] = rel
        values.append(rel)
        records.append({
            "metric": f"{METRIC_PREFIX}_{state}_relative_l2", "value": rel,
            "unit": "relative_l2",
        })
        records.append({
            "metric": f"{METRIC_PREFIX}_{state}_score", "value": spectra_score(rel),
            "unit": "score_0_1",
        })

    if len(values) >= MIN_VARIABLES_FOR_MEAN:
        mean_rel = float(sum(values) / len(values))
        records.append({
            "metric": f"{METRIC_PREFIX}_mean_relative_l2", "value": mean_rel,
            "unit": "relative_l2",
        })
        records.append({
            "metric": f"{METRIC_PREFIX}_mean_score", "value": spectra_score(mean_rel),
            "unit": "score_0_1",
        })
    elif values:
        LOG.warning(
            "spectra_ecmwf_v2 scorer: only %d variable(s) scored; no mean row "
            "(needs at least %d).", len(values), MIN_VARIABLES_FOR_MEAN,
        )

    try:
        (results_dir / "spectra_v2_scores.json").write_text(json.dumps({
            "method": "mean-curve relative L2 against truth, unweighted, above the band start",
            "truth_spectra_dir": str(truth_root),
            "variables": details,
        }, indent=2) + "\n", encoding="utf-8")
    except OSError:
        LOG.warning("spectra_ecmwf_v2 scorer: could not write spectra_v2_scores.json")
    return records
