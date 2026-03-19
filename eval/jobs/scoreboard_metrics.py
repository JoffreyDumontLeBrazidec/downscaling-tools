from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


SIGMA_LEVELS = (1.0, 5.0, 10.0, 100.0)
SPECTRA_FIELDS = ("10u", "10v", "2t")
RAW_FIELD_DIRS = {"10u": "10u_sfc", "10v": "10v_sfc", "2t": "2t_sfc"}
AMP_FILE_RE = re.compile(r"^ampl_(\d{8})_(\d+)_([a-z0-9]+)_([a-z0-9]+)_([^_]+)_n(\d+)\.npy$")
CHECKPOINT_TOKEN_RE = re.compile(r"(?:^|manual_)([0-9a-f]{7,64})(?:_|$)")


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sigma_fragment(sigma: float) -> str:
    return f"{sigma:g}"


def empty_spectra_metrics(source_path: str = "na") -> dict[str, Any]:
    return {
        "10u": None,
        "10v": None,
        "2t": None,
        "mean": None,
        "source_path": source_path,
        "coverage": "missing",
        "n_curves": None,
        "counts": {},
        "missing_reference_pairs": {},
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def extract_checkpoint_token(text: str) -> str | None:
    match = CHECKPOINT_TOKEN_RE.search(str(text).strip())
    if match:
        return match.group(1)
    return None


def load_sigma_losses_from_csv(csv_path: Path) -> dict[str, float]:
    sigma_losses: dict[str, float] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sigma = finite_float(row.get("sigma"))
            loss = finite_float(row.get("loss"))
            if sigma is None or loss is None:
                continue
            sigma_losses[f"sigma_{sigma_fragment(sigma)}"] = loss
    return sigma_losses


def sigma_losses_for_scoreboard(csv_path: Path) -> dict[str, float | None]:
    sigma_losses = load_sigma_losses_from_csv(csv_path)
    return {
        f"sigma_{sigma_fragment(level)}": sigma_losses.get(f"sigma_{sigma_fragment(level)}")
        for level in SIGMA_LEVELS
    }


def _normalize_tc_rows(rows: list[dict[str, Any]]) -> None:
    m_values = [v for v in (finite_float(row.get("mslp_980_990_fraction")) for row in rows) if v is not None]
    w_values = [v for v in (finite_float(row.get("wind_gt_25_fraction")) for row in rows) if v is not None]
    m_min = min(m_values) if m_values else None
    m_max = max(m_values) if m_values else None
    w_min = min(w_values) if w_values else None
    w_max = max(w_values) if w_values else None

    for row in rows:
        score = finite_float(row.get("extreme_score"))
        if score is not None:
            row["_extreme_score_value"] = score
            continue

        m_val = finite_float(row.get("mslp_980_990_fraction"))
        w_val = finite_float(row.get("wind_gt_25_fraction"))
        if m_val is None or w_val is None or m_min is None or m_max is None or w_min is None or w_max is None:
            row["_extreme_score_value"] = None
            continue

        m_norm = 0.0 if m_max <= m_min else (m_val - m_min) / (m_max - m_min)
        w_norm = 0.0 if w_max <= w_min else (w_val - w_min) / (w_max - w_min)
        row["_extreme_score_value"] = 0.5 * (m_norm + w_norm)


def _tc_candidates(run_id: str) -> list[str]:
    candidates = [run_id.strip()]
    token = extract_checkpoint_token(run_id)
    if token:
        candidates.extend({token, token[:8], token[:7]})
    return [candidate for candidate in candidates if candidate]


def _is_reference_row(exp: str) -> bool:
    exp_upper = exp.upper()
    return exp_upper.startswith("ENFO_O320") or exp_upper.startswith("IP6Y")


def _choose_tc_row(rows: list[dict[str, Any]], run_id: str) -> dict[str, Any] | None:
    candidates = _tc_candidates(run_id)
    for candidate in candidates:
        for row in rows:
            exp = str(row.get("exp", "")).strip()
            if exp == candidate or candidate in exp or exp in candidate:
                return row

    non_reference = [row for row in rows if not _is_reference_row(str(row.get("exp", "")).strip())]
    if len(non_reference) == 1:
        return non_reference[0]
    if non_reference:
        return non_reference[0]
    if len(rows) == 1:
        return rows[0]
    return None


def load_tc_extreme_scores_from_json(stats_path: Path, *, run_id: str) -> dict[str, float]:
    data = load_json(stats_path)
    events = data.get("events")
    if not isinstance(events, dict):
        return {}

    scores: dict[str, float] = {}
    for event_name in ("idalia", "franklin"):
        event_data = events.get(event_name)
        if not isinstance(event_data, dict):
            continue
        rows = event_data.get("extreme_tail", {}).get("rows", [])
        if not isinstance(rows, list):
            rows = event_data.get("rows", [])
        if not isinstance(rows, list):
            continue
        norm_rows = [row for row in rows if isinstance(row, dict)]
        _normalize_tc_rows(norm_rows)
        chosen = _choose_tc_row(norm_rows, run_id)
        if chosen is None:
            continue
        score = finite_float(chosen.get("_extreme_score_value"))
        if score is not None:
            scores[event_name] = score
    return scores


def finite_positive_mask(arr: np.ndarray) -> np.ndarray:
    idx = np.arange(arr.shape[0], dtype=np.int32)
    return (idx >= 3) & np.isfinite(arr) & (arr > 0.0)


def relative_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    keep = finite_positive_mask(pred) & finite_positive_mask(truth)
    if not np.any(keep):
        return float("nan")
    return float(np.linalg.norm((pred - truth)[keep]) / max(np.linalg.norm(truth[keep]), 1e-12))


def spectra_field_root(root: Path, field_dir: str) -> Path:
    direct = root / field_dir
    if direct.exists():
        return direct
    nested = root / "spectra" / field_dir
    if nested.exists():
        return nested
    return direct


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def _load_spectra_metadata(spectra_dir: Path) -> dict[str, Any]:
    staging_path = spectra_dir / "staging_summary.json"
    if staging_path.exists():
        return load_json(staging_path)

    summary_path = spectra_dir / "spectra_summary.json"
    if summary_path.exists():
        data = load_json(summary_path)
        if any(key in data for key in ("dates", "steps_hours", "ensemble_members", "template_root")):
            return data
    return {}


def _infer_spectra_token(spectra_dir: Path) -> str:
    for field_dir in RAW_FIELD_DIRS.values():
        for ampl_path in spectra_field_root(spectra_dir, field_dir).glob("ampl_*.npy"):
            match = AMP_FILE_RE.match(ampl_path.name)
            if match:
                return match.group(5)
    return "1"


def _extract_spectra_metrics_from_summary(data: dict[str, Any], source_path: str) -> dict[str, Any]:
    source: dict[str, Any] | None = None
    if isinstance(data.get("weather_states"), dict) and data["weather_states"]:
        source = data["weather_states"]
    elif any(isinstance(data.get(field), dict) for field in SPECTRA_FIELDS):
        source = data
    if source is None:
        return empty_spectra_metrics(source_path)

    scores: dict[str, float | None] = {}
    values: list[float] = []
    curve_counts: list[int] = []
    missing_reference_pairs: dict[str, int] = {}
    for field in SPECTRA_FIELDS:
        entry = source.get(field, {})
        if not isinstance(entry, dict):
            scores[field] = None
            continue
        value = finite_float(entry.get("relative_l2_mean_curve"))
        if value is not None:
            scores[field] = value
            values.append(value)
        else:
            scores[field] = None

        count = entry.get("n_curves")
        if count is None:
            count = entry.get("n_pairs")
        if isinstance(count, int):
            curve_counts.append(int(count))

        missing = entry.get("missing_reference_pairs")
        if isinstance(missing, int):
            missing_reference_pairs[field] = int(missing)

    mean = sum(values) / len(values) if len(values) == len(SPECTRA_FIELDS) else None
    return {
        "10u": scores.get("10u"),
        "10v": scores.get("10v"),
        "2t": scores.get("2t"),
        "mean": mean,
        "source_path": source_path,
        "coverage": f"{curve_counts[0]} pairs" if curve_counts and len(set(curve_counts)) == 1 else "mixed pairs",
        "n_curves": curve_counts[0] if curve_counts and len(set(curve_counts)) == 1 else None,
        "counts": {field: entry.get("n_pairs", entry.get("n_curves", 0)) for field, entry in source.items() if field in SPECTRA_FIELDS and isinstance(entry, dict)},
        "missing_reference_pairs": missing_reference_pairs,
    }


def _load_raw_spectra_metrics(spectra_dir: Path, reference_root: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    dates = _int_list(metadata.get("dates"))
    steps = _int_list(metadata.get("steps_hours", metadata.get("steps")))
    members = _int_list(metadata.get("ensemble_members", metadata.get("members")))
    if not dates or not steps or not members:
        return empty_spectra_metrics(str(spectra_dir))

    token = _infer_spectra_token(spectra_dir)
    ref_token = str(metadata.get("reference_token", "1")).strip() or "1"

    scores: dict[str, float | None] = {}
    values: list[float] = []
    coverage_parts: list[str] = []
    curve_count: int | None = None
    counts: dict[str, int] = {}
    missing_reference_pairs: dict[str, int] = {}
    for field, field_dir in RAW_FIELD_DIRS.items():
        exp_curves: list[np.ndarray] = []
        ref_curves: list[np.ndarray] = []
        used_dates: set[int] = set()
        used_steps: set[int] = set()
        used_members: set[int] = set()
        misses = 0
        exp_dir = spectra_field_root(spectra_dir, field_dir)
        ref_dir = spectra_field_root(reference_root, field_dir)
        for date in dates:
            for step in steps:
                for member in members:
                    exp_file = exp_dir / f"ampl_{date}_{step}_{field_dir}_{token}_n{member}.npy"
                    ref_file = ref_dir / f"ampl_{date}_{step}_{field_dir}_{ref_token}_n{member}.npy"
                    if not exp_file.exists():
                        continue
                    if not ref_file.exists():
                        misses += 1
                        continue
                    exp_curves.append(np.load(exp_file))
                    ref_curves.append(np.load(ref_file))
                    used_dates.add(date)
                    used_steps.add(step)
                    used_members.add(member)

        missing_reference_pairs[field] = misses
        counts[field] = len(exp_curves)
        if not exp_curves or not ref_curves:
            scores[field] = None
            coverage_parts.append(f"{field}: missing")
            continue

        exp_mean = np.nanmean(np.stack(exp_curves, axis=0), axis=0)
        ref_mean = np.nanmean(np.stack(ref_curves, axis=0), axis=0)
        if exp_mean.shape != ref_mean.shape:
            raise ValueError(
                f"spectra length mismatch for field={field}: "
                f"candidate={exp_mean.shape[0]} reference={ref_mean.shape[0]} "
                f"(root={spectra_dir}, reference_root={reference_root})"
            )
        score = relative_l2(exp_mean, ref_mean)
        scores[field] = score if math.isfinite(score) else None
        if scores[field] is not None:
            values.append(scores[field])
        coverage_parts.append(
            f"{field}: {len(used_dates)}d/{len(used_members)}m/{len(used_steps)}s/{len(exp_curves)}c"
        )
        if curve_count is None:
            curve_count = len(exp_curves)

    mean = sum(values) / len(values) if len(values) == len(SPECTRA_FIELDS) else None
    return {
        "10u": scores.get("10u"),
        "10v": scores.get("10v"),
        "2t": scores.get("2t"),
        "mean": mean,
        "source_path": str(spectra_dir),
        "coverage": "; ".join(coverage_parts) if coverage_parts else "missing",
        "n_curves": curve_count,
        "counts": counts,
        "missing_reference_pairs": missing_reference_pairs,
    }


def load_spectra_metrics(spectra_dir: Path, *, reference_root: Path | None = None) -> dict[str, Any]:
    summary_path = spectra_dir / "spectra_summary.json"
    if summary_path.exists():
        summary_data = load_json(summary_path)
        summary_metrics = _extract_spectra_metrics_from_summary(summary_data, str(summary_path))
        if summary_metrics["mean"] is not None or any(summary_metrics[field] is not None for field in SPECTRA_FIELDS):
            return summary_metrics
    else:
        summary_data = {}

    metadata = _load_spectra_metadata(spectra_dir)
    ref_path: Path | None = reference_root
    if ref_path is None:
        relative_to = str(summary_data.get("relative_to", "")).strip()
        if relative_to:
            ref_path = Path(relative_to)
    if ref_path is None:
        template_root = str(metadata.get("template_root", "")).strip()
        if template_root:
            ref_path = Path(template_root)
    if ref_path is None or not ref_path.exists():
        return empty_spectra_metrics(str(summary_path if summary_path.exists() else spectra_dir))

    return _load_raw_spectra_metrics(spectra_dir, ref_path, metadata)


def build_spectra_summary(spectra_dir: Path, *, reference_root: Path | None = None) -> dict[str, Any]:
    metrics = load_spectra_metrics(spectra_dir, reference_root=reference_root)
    relative_to = str(reference_root or "")
    if not relative_to:
        metadata = _load_spectra_metadata(spectra_dir)
        relative_to = str(metadata.get("template_root", "")).strip()

    summary: dict[str, Any] = {
        "method": "ecmwf_mean_curve_reference_l2",
        "relative_to": relative_to,
        "spectra_output_dir": str(spectra_dir),
        "scoreboard_fields": list(SPECTRA_FIELDS),
    }
    if metrics["mean"] is not None:
        summary["mean_relative_l2"] = float(metrics["mean"])
    if metrics.get("coverage"):
        summary["coverage"] = metrics["coverage"]

    counts = metrics.get("counts", {})
    missing_reference_pairs = metrics.get("missing_reference_pairs", {})
    for field in SPECTRA_FIELDS:
        score = metrics.get(field)
        field_summary: dict[str, Any]
        if score is None:
            field_summary = {"status": "missing"}
        else:
            field_summary = {"relative_l2_mean_curve": float(score)}
        if isinstance(counts.get(field), int):
            field_summary["n_pairs"] = int(counts[field])
        if isinstance(missing_reference_pairs.get(field), int):
            field_summary["missing_reference_pairs"] = int(missing_reference_pairs[field])
        summary[field] = field_summary
    return summary


def load_surface_weighted_mse(surface_json_path: Path) -> float | None:
    data = load_json(surface_json_path)
    return finite_float(data.get("weighted_surface_mse"))


def build_run_scoreboard_metrics(
    *,
    run_id: str,
    output_root: Path,
    sigma_run_id: str,
    tc_stats_path: Path,
    spectra_dir: Path,
    surface_json_path: Path,
) -> dict[str, Any]:
    out: dict[str, Any] = {"run_id": run_id}

    sigma_losses = {
        f"sigma_{sigma_fragment(level)}": None
        for level in SIGMA_LEVELS
    }
    if sigma_run_id:
        sigma_csv = output_root / "scoreboards" / "sigma" / f"{sigma_run_id}_sigma_eval.csv"
        if sigma_csv.exists():
            sigma_losses.update(sigma_losses_for_scoreboard(sigma_csv))
    out["sigma_losses"] = sigma_losses

    tc_scores = {}
    if tc_stats_path.exists():
        tc_scores = load_tc_extreme_scores_from_json(tc_stats_path, run_id=run_id)
    out["tc_extreme_scores"] = tc_scores

    spectra_metrics = load_spectra_metrics(spectra_dir)
    if spectra_metrics["mean"] is not None:
        out["spectra_mean_relative_l2"] = float(spectra_metrics["mean"])
        out["spectra_relative_l2"] = {
            field: float(spectra_metrics[field])
            for field in SPECTRA_FIELDS
            if spectra_metrics[field] is not None
        }

    surface_loss = load_surface_weighted_mse(surface_json_path)
    if surface_loss is not None:
        out["surface_weighted_mse"] = surface_loss

    return out
