from __future__ import annotations

import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


SIGMA_LEVELS = (1.0, 5.0, 10.0, 100.0)
SPECTRA_FIELDS = ("10u", "10v", "2t", "msl", "t_850", "z_500")
RAW_FIELD_DIRS = {
    "10u": "10u_sfc",
    "10v": "10v_sfc",
    "2t": "2t_sfc",
    "msl": "msl_sfc",
    "t_850": "t_850",
    "z_500": "z_500",
}
SPECTRA_FIELD_DIR_ALIASES = {
    "msl": ("sp_sfc",),
}
SPECTRA_SUMMARY_ALIASES = {
    "msl": ("sp",),
}
SURFACE_VAR_LABELS = {
    "10u": "10u",
    "10v": "10v",
    "2d": "2d",
    "2t": "2t",
    "msl": "MSLP",
    "skt": "SKT",
    "sp": "SP",
    "tcw": "TCW",
}
SURFACE_NORMALIZATION_SCHEME = "truth-std"
MSLP_REFERENCE_HPA = 1013.25
AMP_FILE_RE = re.compile(r"^ampl_(\d{8})_(\d+)_([a-z0-9]+)_([a-z0-9]+)_([^_]+)_n(\d+)\.npy$")
CHECKPOINT_TOKEN_RE = re.compile(r"(?:^|manual_)([0-9a-f]{7,64})(?:_|$)")
SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE = 100.0


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sigma_fragment(sigma: float) -> str:
    return f"{sigma:g}"


def empty_spectra_metrics(source_path: str = "na") -> dict[str, Any]:
    result: dict[str, Any] = {field: None for field in SPECTRA_FIELDS}
    result.update({
        "mean": None,
        "source_path": source_path,
        "coverage": "missing",
        "n_curves": None,
        "count_label": "pairs",
        "counts": {},
        "missing_reference_pairs": {},
        "score_wavenumber_min_exclusive": SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE,
    })
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def load_mapping_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return {}
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception:
            return {}
        try:
            data = yaml.safe_load(text)
        except Exception:
            return {}
    return data if isinstance(data, dict) else {}


def parse_json_object(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return dict(raw)
    if raw in (None, ""):
        return None
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extra_args_from_development_hacks(config: dict[str, Any]) -> dict[str, Any] | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    hacks = model.get("development_hacks")
    if not isinstance(hacks, dict):
        return None
    extra_args = hacks.get("extra_args")
    return dict(extra_args) if isinstance(extra_args, dict) else None


def extract_checkpoint_token(text: str) -> str | None:
    match = CHECKPOINT_TOKEN_RE.search(str(text).strip())
    if match:
        return match.group(1)
    return None


def parse_sampling_text_map(sampling_text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in str(sampling_text or "").split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            parsed[key] = value
    return parsed


def normalize_schedule_label(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text or text == "na":
        return ""
    if "experimental_piecewise" in text:
        return "piecewise"
    piecewise_match = re.search(r"(piecewise\d+)", text)
    if piecewise_match:
        return piecewise_match.group(1)
    if "piecewise" in text:
        return "piecewise"
    if "karras" in text:
        return "karras"
    if "exponential" in text:
        return "exponential"
    if "lognormal" in text or "lognorm" in text:
        return "lognorm"
    head = re.split(r"[_\s]", text, maxsplit=1)[0]
    return head if head and head != "experimental" else text


def format_step_count(raw: Any) -> str:
    if raw in (None, ""):
        return ""
    text = str(raw).strip()
    if not text or text.lower() == "na":
        return ""
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if not math.isfinite(number):
        return ""
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def infer_schedule_label(values: dict[str, Any]) -> str:
    for key in ("schedule", "schedule_name", "schedule_type", "scheduler", "scheduler_type", "noise_schedule"):
        label = normalize_schedule_label(values.get(key))
        if label:
            return label
    high = normalize_schedule_label(values.get("high_schedule_type"))
    low = normalize_schedule_label(values.get("low_schedule_type"))
    if high or low:
        return "piecewise"
    has_step_count = any(format_step_count(values.get(key)) for key in ("num_steps", "steps", "n_steps"))
    if has_step_count and any(values.get(key) not in (None, "", "na") for key in ("rho", "sigma_max", "sigma_min")):
        return "karras"
    return ""


def parse_python_dict(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_extra_args_from_log(log_path: Path) -> dict[str, Any] | None:
    scheduler: dict[str, Any] | None = None
    sampler: dict[str, Any] | None = None
    try:
        with log_path.open(errors="ignore") as handle:
            for line in handle:
                if scheduler is None:
                    if "noise_scheduler_config:" in line:
                        scheduler = parse_python_dict(line.split("noise_scheduler_config:", 1)[1])
                    elif "noise_scheduler_params:" in line:
                        scheduler = parse_python_dict(line.split("noise_scheduler_params:", 1)[1])
                if sampler is None:
                    if "diffusion_sampler_config:" in line:
                        sampler = parse_python_dict(line.split("diffusion_sampler_config:", 1)[1])
                    elif "sampler_params:" in line:
                        sampler = parse_python_dict(line.split("sampler_params:", 1)[1])
                if scheduler is not None and sampler is not None:
                    break
    except OSError:
        return None

    if scheduler is None and sampler is None:
        return None
    merged: dict[str, Any] = {}
    if scheduler is not None:
        merged.update(scheduler)
    if sampler is not None:
        merged.update(sampler)
    return merged


def infer_eval_sampler_min(extra_args: dict[str, Any] | None, sampling_text: str = "") -> str:
    parsed_text = parse_sampling_text_map(sampling_text)
    merged: dict[str, Any] = dict(parsed_text)
    if isinstance(extra_args, dict):
        merged.update(extra_args)

    schedule_label = infer_schedule_label(merged)
    step_count = ""
    for key in ("num_steps", "steps", "n_steps"):
        step_count = format_step_count(merged.get(key))
        if step_count:
            break

    if schedule_label.startswith("piecewise") and re.search(r"\d+$", schedule_label):
        return schedule_label
    if schedule_label and step_count:
        return f"{schedule_label}{step_count}"
    if schedule_label:
        return schedule_label
    if step_count:
        return f"steps{step_count}"
    return "na"


def infer_eval_sampler_min_from_run_root(run_root: Path) -> str:
    config_path = run_root / "EXPERIMENT_CONFIG.yaml"
    sampling_text = ""
    extra_args: dict[str, Any] | None = None

    if config_path.exists():
        config = load_mapping_file(config_path)
        parsed_sampling = parse_json_object(config.get("sampling_config_json"))
        if parsed_sampling is not None:
            extra_args = parsed_sampling
        else:
            raw_sampling = config.get("sampling_config_json")
            if raw_sampling not in (None, ""):
                sampling_text = str(raw_sampling).strip()
        if extra_args is None:
            config_extra_args = extra_args_from_development_hacks(config)
            if config_extra_args is not None:
                extra_args = config_extra_args

    label = infer_eval_sampler_min(extra_args, sampling_text)
    if label != "na":
        return label

    logs_dir = run_root / "logs"
    if not logs_dir.exists():
        return "na"

    seen: set[Path] = set()
    log_candidates = (
        sorted(logs_dir.glob("predict25_*.out"))
        + sorted(logs_dir.glob("predict_proxy_*.out"))
        + sorted(logs_dir.glob("*.out"))
    )
    for log_path in log_candidates:
        if log_path in seen:
            continue
        seen.add(log_path)
        parsed_from_log = extract_extra_args_from_log(log_path)
        if parsed_from_log is None:
            continue
        label = infer_eval_sampler_min(parsed_from_log)
        if label != "na":
            return label

    return "na"


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


def _mslp_depth(value: float) -> float:
    """Convert MSLP (hPa) to depth below standard reference: deeper = more extreme."""
    return max(MSLP_REFERENCE_HPA - value, 0.0)


def _is_analysis_row(exp: str) -> bool:
    return exp.upper().startswith("OPER")


def _find_row_by_predicate(rows: list[dict[str, Any]], predicate) -> dict[str, Any] | None:
    for row in rows:
        if predicate(str(row.get("exp", "")).strip()):
            return row
    return None


def _multi_depth_tc_score(
    model: dict[str, Any],
    analysis: dict[str, Any],
) -> float | None:
    """Compute analysis-anchored TC score using multi-depth tail percentiles.

    Returns 0-1 score: 1.0 = model matches analysis extremes, 0.0 = no extreme signal.
    """
    mslp_keys = ("mslp_p1", "mslp_p01", "mslp_min")
    wind_keys = ("wind_p99", "wind_p999", "wind_max")

    # MSLP: low tail, use depth metric
    mslp_ratios: list[float] = []
    for key in mslp_keys:
        m_val = finite_float(model.get(key))
        a_val = finite_float(analysis.get(key))
        if m_val is None or a_val is None:
            continue
        a_depth = _mslp_depth(a_val)
        if a_depth <= 0.0:
            continue
        m_depth = _mslp_depth(m_val)
        mslp_ratios.append(min(m_depth / a_depth, 1.0))

    # Wind: high tail, direct ratio
    wind_ratios: list[float] = []
    for key in wind_keys:
        m_val = finite_float(model.get(key))
        a_val = finite_float(analysis.get(key))
        if m_val is None or a_val is None:
            continue
        if a_val <= 0.0:
            continue
        wind_ratios.append(min(m_val / a_val, 1.0))

    if not mslp_ratios and not wind_ratios:
        return None

    scores: list[float] = []
    if mslp_ratios:
        scores.append(sum(mslp_ratios) / len(mslp_ratios))
    if wind_ratios:
        scores.append(sum(wind_ratios) / len(wind_ratios))
    return sum(scores) / len(scores)


def _multi_depth_enfo_deviation(
    model: dict[str, Any],
    enfo: dict[str, Any],
    analysis: dict[str, Any],
) -> float | None:
    """Compute ENFO deviation: how far the model diverges from ENFO, normalized by analysis."""
    mslp_keys = ("mslp_p1", "mslp_p01", "mslp_min")
    wind_keys = ("wind_p99", "wind_p999", "wind_max")

    mslp_devs: list[float] = []
    for key in mslp_keys:
        m_val = finite_float(model.get(key))
        e_val = finite_float(enfo.get(key))
        a_val = finite_float(analysis.get(key))
        if m_val is None or e_val is None or a_val is None:
            continue
        a_depth = _mslp_depth(a_val)
        if a_depth <= 0.0:
            continue
        mslp_devs.append(abs(_mslp_depth(m_val) - _mslp_depth(e_val)) / a_depth)

    wind_devs: list[float] = []
    for key in wind_keys:
        m_val = finite_float(model.get(key))
        e_val = finite_float(enfo.get(key))
        a_val = finite_float(analysis.get(key))
        if m_val is None or e_val is None or a_val is None:
            continue
        if a_val <= 0.0:
            continue
        wind_devs.append(abs(m_val - e_val) / a_val)

    if not mslp_devs and not wind_devs:
        return None

    devs: list[float] = []
    if mslp_devs:
        devs.append(sum(mslp_devs) / len(mslp_devs))
    if wind_devs:
        devs.append(sum(wind_devs) / len(wind_devs))
    return sum(devs) / len(devs)


def _normalize_tc_rows(rows: list[dict[str, Any]]) -> None:
    """Analysis-anchored TC scoring with multi-depth tail percentiles.

    Falls back to legacy batch-relative normalization when analysis row or
    tail percentiles are not available.
    """
    analysis_row = _find_row_by_predicate(rows, _is_analysis_row)
    enfo_row = _find_row_by_predicate(rows, _is_reference_row)

    # If analysis row exists and has tail percentiles, use analysis-anchored scoring
    if analysis_row is not None and finite_float(analysis_row.get("mslp_p1")) is not None:
        for row in rows:
            if _is_analysis_row(str(row.get("exp", "")).strip()):
                row["_extreme_score_value"] = 1.0  # analysis is perfect by definition
                row["_enfo_deviation_value"] = None
                continue
            score = _multi_depth_tc_score(row, analysis_row)
            row["_extreme_score_value"] = score
            if enfo_row is not None:
                row["_enfo_deviation_value"] = _multi_depth_enfo_deviation(row, enfo_row, analysis_row)
            else:
                row["_enfo_deviation_value"] = None
        return

    # Legacy fallback: batch-relative normalization (for old stats JSONs without tail percentiles)
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
            row["_enfo_deviation_value"] = None
            continue

        m_val = finite_float(row.get("mslp_980_990_fraction"))
        w_val = finite_float(row.get("wind_gt_25_fraction"))
        if m_val is None or w_val is None or m_min is None or m_max is None or w_min is None or w_max is None:
            row["_extreme_score_value"] = None
            row["_enfo_deviation_value"] = None
            continue

        m_norm = 0.0 if m_max <= m_min else (m_val - m_min) / (m_max - m_min)
        w_norm = 0.0 if w_max <= w_min else (w_val - w_min) / (w_max - w_min)
        row["_extreme_score_value"] = 0.5 * (m_norm + w_norm)
        row["_enfo_deviation_value"] = None


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


def load_tc_extreme_scores_from_json(
    stats_path: Path,
    *,
    run_id: str,
    event_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    """Load TC extreme scores and ENFO deviation from a stats JSON.

    Returns dict with keys like "idalia", "franklin" (scores) and
    "idalia_enfo_dev", "franklin_enfo_dev" (deviations, when available).
    """
    data = load_json(stats_path)
    events = data.get("events")
    if not isinstance(events, dict):
        return {}

    requested_events = tuple(event_names or ("idalia", "franklin"))
    scores: dict[str, float] = {}
    for event_name in requested_events:
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
        enfo_dev = finite_float(chosen.get("_enfo_deviation_value"))
        if enfo_dev is not None:
            scores[f"{event_name}_enfo_dev"] = enfo_dev
    return scores


def _default_wavenumbers(length: int) -> np.ndarray:
    return np.arange(1, length + 1, dtype=np.float64)


def finite_positive_mask(arr: np.ndarray, *, wavenumbers: np.ndarray | None = None) -> np.ndarray:
    wvn = _default_wavenumbers(arr.shape[0]) if wavenumbers is None else np.asarray(wavenumbers, dtype=np.float64)
    if wvn.shape != arr.shape:
        raise ValueError(f"wavenumber length mismatch: curve={arr.shape[0]} wavenumbers={wvn.shape[0]}")
    threshold = SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE
    max_wvn = float(wvn.max()) if wvn.size > 0 else 0.0
    if max_wvn <= threshold:
        threshold = max_wvn / 3.0
    return np.isfinite(arr) & (arr > 0.0) & np.isfinite(wvn) & (wvn > threshold)


def relative_l2(pred: np.ndarray, truth: np.ndarray, *, wavenumbers: np.ndarray | None = None) -> float:
    keep = finite_positive_mask(pred, wavenumbers=wavenumbers) & finite_positive_mask(truth, wavenumbers=wavenumbers)
    if not np.any(keep):
        return float("nan")
    return float(np.linalg.norm((pred - truth)[keep]) / max(np.linalg.norm(truth[keep]), 1e-12))


def relative_l2_weighted(pred: np.ndarray, truth: np.ndarray, *, wavenumbers: np.ndarray | None = None) -> float:
    """Fine-scale-weighted relative L2: wavenumber/max_wavenumber weighting.

    High wavenumber (fine scale) errors count more than low wavenumber (synoptic)
    errors. At wn=300 vs wn=100, fine-scale errors count ~3× more.
    """
    wvn = _default_wavenumbers(pred.shape[0]) if wavenumbers is None else np.asarray(wavenumbers, dtype=np.float64)
    keep = finite_positive_mask(pred, wavenumbers=wvn) & finite_positive_mask(truth, wavenumbers=wvn)
    if not np.any(keep):
        return float("nan")
    wn = wvn[keep]
    weights = wn / wn.max()
    diff = (pred - truth)[keep] * weights
    ref = truth[keep] * weights
    return float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-12))


def spectra_field_root(root: Path, field_dir: str) -> Path:
    direct = root / field_dir
    if direct.exists():
        return direct
    nested = root / "spectra" / field_dir
    if nested.exists():
        return nested
    return direct


def spectra_field_dir_candidates(field: str) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for field_dir in (RAW_FIELD_DIRS[field], *SPECTRA_FIELD_DIR_ALIASES.get(field, ())):
        if field_dir in seen:
            continue
        seen.add(field_dir)
        ordered.append(field_dir)
    return tuple(ordered)


def spectra_summary_keys(field: str) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for key in (field, *SPECTRA_SUMMARY_ALIASES.get(field, ())):
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return tuple(ordered)


def _resolve_spectra_field_root(root: Path, field: str) -> tuple[Path, str]:
    for field_dir in spectra_field_dir_candidates(field):
        candidate = spectra_field_root(root, field_dir)
        if candidate.exists():
            return candidate, field_dir
    fallback_dir = spectra_field_dir_candidates(field)[0]
    return spectra_field_root(root, fallback_dir), fallback_dir


def _extract_spectra_summary_entry(source: dict[str, Any], field: str) -> tuple[dict[str, Any], str | None]:
    for key in spectra_summary_keys(field):
        entry = source.get(key)
        if isinstance(entry, dict):
            return entry, key
    return {}, None


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
    seen: set[str] = set()
    for field in SPECTRA_FIELDS:
        for field_dir in spectra_field_dir_candidates(field):
            if field_dir in seen:
                continue
            seen.add(field_dir)
            for ampl_path in spectra_field_root(spectra_dir, field_dir).glob("ampl_*.npy"):
                match = AMP_FILE_RE.match(ampl_path.name)
                if match:
                    return match.group(5)
    return "1"


def _has_raw_spectra_arrays(spectra_dir: Path) -> bool:
    seen: set[str] = set()
    for field in SPECTRA_FIELDS:
        for field_dir in spectra_field_dir_candidates(field):
            if field_dir in seen:
                continue
            seen.add(field_dir)
            if any(spectra_field_root(spectra_dir, field_dir).glob("ampl_*.npy")):
                return True
    return False


def _load_curve_wavenumbers(root: Path, *, date: int, step: int, field_dir: str, token: str, member: int) -> np.ndarray | None:
    path = root / f"wvn_{date}_{step}_{field_dir}_{token}_n{member}.npy"
    if not path.exists():
        return None
    return np.asarray(np.load(path), dtype=np.float64)


def _extract_spectra_metrics_from_summary(data: dict[str, Any], source_path: str) -> dict[str, Any]:
    # Prefer residual-space scope when available
    source: dict[str, Any] | None = None
    count_label = "pairs"
    scopes = data.get("scopes")
    if isinstance(scopes, dict) and isinstance(scopes.get("residual"), dict):
        source = scopes["residual"]
        count_label = "curves"
    elif isinstance(data.get("weather_states"), dict) and data["weather_states"]:
        source = data["weather_states"]
        count_label = "curves"
    elif any(isinstance(data.get(field), dict) for field in SPECTRA_FIELDS):
        source = data
    if source is None:
        return empty_spectra_metrics(source_path)

    scores: dict[str, float | None] = {}
    values: list[float] = []
    curve_counts: list[int] = []
    counts: dict[str, int] = {}
    missing_reference_pairs: dict[str, int] = {}
    for field in SPECTRA_FIELDS:
        entry, _ = _extract_spectra_summary_entry(source, field)
        if not entry:
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
            counts[field] = int(count)

        missing = entry.get("missing_reference_pairs")
        if isinstance(missing, int):
            missing_reference_pairs[field] = int(missing)

    # Mean requires at least the 3 original fields; include any additional fields that have data
    available_count = len(values)
    mean = sum(values) / available_count if available_count >= 3 else None
    result: dict[str, Any] = {field: scores.get(field) for field in SPECTRA_FIELDS}
    result.update({
        "mean": mean,
        "source_path": source_path,
        "coverage": f"{curve_counts[0]} {count_label}" if curve_counts and len(set(curve_counts)) == 1 else f"mixed {count_label}",
        "n_curves": curve_counts[0] if curve_counts and len(set(curve_counts)) == 1 else None,
        "count_label": count_label,
        "counts": counts,
        "missing_reference_pairs": missing_reference_pairs,
        "score_wavenumber_min_exclusive": finite_float(data.get("score_wavenumber_min_exclusive"))
        or SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE,
    })
    return result


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
    for field in SPECTRA_FIELDS:
        exp_curves: list[np.ndarray] = []
        ref_curves: list[np.ndarray] = []
        wavenumbers: np.ndarray | None = None
        used_dates: set[int] = set()
        used_steps: set[int] = set()
        used_members: set[int] = set()
        misses = 0
        exp_dir, exp_field_dir = _resolve_spectra_field_root(spectra_dir, field)
        ref_dir, ref_field_dir = _resolve_spectra_field_root(reference_root, field)
        for date in dates:
            for step in steps:
                for member in members:
                    exp_file = exp_dir / f"ampl_{date}_{step}_{exp_field_dir}_{token}_n{member}.npy"
                    ref_file = ref_dir / f"ampl_{date}_{step}_{ref_field_dir}_{ref_token}_n{member}.npy"
                    if not exp_file.exists():
                        continue
                    if not ref_file.exists():
                        misses += 1
                        continue
                    exp_curves.append(np.load(exp_file))
                    ref_curves.append(np.load(ref_file))
                    curve_wavenumbers = _load_curve_wavenumbers(
                        exp_dir,
                        date=date,
                        step=step,
                        field_dir=exp_field_dir,
                        token=token,
                        member=member,
                    )
                    if curve_wavenumbers is None:
                        curve_wavenumbers = _load_curve_wavenumbers(
                            ref_dir,
                            date=date,
                            step=step,
                            field_dir=ref_field_dir,
                            token=ref_token,
                            member=member,
                        )
                    if curve_wavenumbers is not None:
                        if wavenumbers is None:
                            wavenumbers = curve_wavenumbers
                        elif curve_wavenumbers.shape != wavenumbers.shape or not np.allclose(curve_wavenumbers, wavenumbers):
                            raise ValueError(
                                f"wavenumber mismatch for field={field}: "
                                f"candidate={curve_wavenumbers.shape} reference={wavenumbers.shape} "
                                f"(root={spectra_dir}, reference_root={reference_root})"
                            )
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
        score = relative_l2_weighted(exp_mean, ref_mean, wavenumbers=wavenumbers)
        scores[field] = score if math.isfinite(score) else None
        if scores[field] is not None:
            values.append(scores[field])
        coverage_parts.append(
            f"{field}: {len(used_dates)}d/{len(used_members)}m/{len(used_steps)}s/{len(exp_curves)}c"
        )
        if curve_count is None:
            curve_count = len(exp_curves)

    available_count = len(values)
    mean = sum(values) / available_count if available_count >= 3 else None
    result: dict[str, Any] = {field: scores.get(field) for field in SPECTRA_FIELDS}
    result.update({
        "mean": mean,
        "source_path": str(spectra_dir),
        "coverage": "; ".join(coverage_parts) if coverage_parts else "missing",
        "n_curves": curve_count,
        "count_label": "pairs",
        "counts": counts,
        "missing_reference_pairs": missing_reference_pairs,
        "score_wavenumber_min_exclusive": SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE,
    })
    return result


def _match_comparison_model(
    models: list[dict[str, Any]],
    *,
    preferred_name: str = "",
    preferred_path: Path | None = None,
) -> dict[str, Any] | None:
    target_name = preferred_name.strip()
    target_path = str(preferred_path) if preferred_path is not None else ""
    target_path_name = preferred_path.name if preferred_path is not None else ""
    for model in models:
        name = str(model.get("name", "")).strip()
        path_text = str(model.get("path", "")).strip()
        path_name = Path(path_text).name if path_text else ""
        if target_path and path_text == target_path:
            return model
        if target_name and name == target_name:
            return model
        if target_path_name and path_name == target_path_name:
            return model
    return None


def _load_step_mean_curve(root: Path, *, field_dir: str, step: int) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    field_root = spectra_field_root(root, field_dir)
    w_arrays: list[np.ndarray] = []
    a_arrays: list[np.ndarray] = []
    pattern = f"wvn_*_{step}_{field_dir}_*_n*.npy"
    for w_path in sorted(field_root.glob(pattern)):
        a_path = w_path.with_name(w_path.name.replace("wvn_", "ampl_", 1))
        if not a_path.exists():
            continue
        try:
            wavenumbers = np.asarray(np.load(w_path), dtype=np.float64)
            amplitudes = np.asarray(np.load(a_path), dtype=np.float64)
        except (OSError, ValueError):
            continue
        if wavenumbers.shape != amplitudes.shape or wavenumbers.size < 8:
            continue
        if w_arrays and (wavenumbers.shape != w_arrays[0].shape or amplitudes.shape != a_arrays[0].shape):
            continue
        w_arrays.append(wavenumbers)
        a_arrays.append(amplitudes)
    if not a_arrays:
        return None, None, 0
    mean_wavenumbers = np.nanmean(np.stack(w_arrays, axis=0), axis=0)
    mean_amplitudes = np.nanmean(np.stack(a_arrays, axis=0), axis=0)
    return mean_wavenumbers, mean_amplitudes, len(a_arrays)


def _load_reference_model_comparison_metrics(
    spectra_dir: Path,
    comparison_path: Path,
    *,
    reference_root: Path | None = None,
) -> dict[str, Any]:
    data = load_json(comparison_path)
    models = data.get("models")
    per_param = data.get("per_param")
    if not isinstance(models, list) or not isinstance(per_param, dict):
        return empty_spectra_metrics(str(comparison_path))

    model_entries = [entry for entry in models if isinstance(entry, dict)]
    candidate_model = _match_comparison_model(
        model_entries,
        preferred_name=spectra_dir.name,
        preferred_path=spectra_dir,
    )
    if candidate_model is None:
        return empty_spectra_metrics(str(comparison_path))

    reference_model = None
    if reference_root is not None:
        reference_model = _match_comparison_model(
            model_entries,
            preferred_name=reference_root.name,
            preferred_path=reference_root,
        )
    if reference_model is None:
        reference_model = _match_comparison_model(model_entries, preferred_name="enfo_o320")
    if reference_model is None:
        return empty_spectra_metrics(str(comparison_path))

    candidate_name = str(candidate_model.get("name", "")).strip()
    reference_name = str(reference_model.get("name", "")).strip()
    candidate_root = Path(str(candidate_model.get("path", "")).strip())
    reference_model_root = Path(str(reference_model.get("path", "")).strip())
    if not candidate_name or not reference_name or not candidate_root.exists() or not reference_model_root.exists():
        return empty_spectra_metrics(str(comparison_path))

    scores: dict[str, float | None] = {}
    values: list[float] = []
    counts: dict[str, int] = {}
    coverage_parts: list[str] = []
    curve_count: int | None = None
    for field in SPECTRA_FIELDS:
        field_dir = None
        candidate_entry: dict[str, Any] | None = None
        reference_entry: dict[str, Any] | None = None
        for candidate_field_dir in spectra_field_dir_candidates(field):
            param_entry = per_param.get(candidate_field_dir)
            if not isinstance(param_entry, dict):
                continue
            trial_candidate = param_entry.get(candidate_name)
            trial_reference = param_entry.get(reference_name)
            if isinstance(trial_candidate, dict) and isinstance(trial_reference, dict):
                field_dir = candidate_field_dir
                candidate_entry = trial_candidate
                reference_entry = trial_reference
                break
        if field_dir is None or candidate_entry is None or reference_entry is None:
            scores[field] = None
            coverage_parts.append(f"{field}: missing")
            continue

        candidate_step = candidate_entry.get("step")
        reference_step = reference_entry.get("step")
        candidate_status = str(candidate_entry.get("status", "")).strip().lower()
        reference_status = str(reference_entry.get("status", "")).strip().lower()
        if (
            not isinstance(candidate_step, int)
            or not isinstance(reference_step, int)
            or candidate_status != "plotted"
            or reference_status != "plotted"
        ):
            scores[field] = None
            coverage_parts.append(f"{field}: missing")
            continue

        candidate_wvn, candidate_mean, candidate_count = _load_step_mean_curve(
            candidate_root,
            field_dir=field_dir,
            step=int(candidate_step),
        )
        _, reference_mean, reference_count = _load_step_mean_curve(
            reference_model_root,
            field_dir=field_dir,
            step=int(reference_step),
        )
        counts[field] = min(candidate_count, reference_count)
        if candidate_mean is None or reference_mean is None:
            scores[field] = None
            coverage_parts.append(f"{field}: missing")
            continue
        if candidate_mean.shape != reference_mean.shape:
            raise ValueError(
                f"spectra length mismatch for field={field}: "
                f"candidate={candidate_mean.shape[0]} reference={reference_mean.shape[0]} "
                f"(root={candidate_root}, reference_root={reference_model_root})"
            )
        score = relative_l2_weighted(candidate_mean, reference_mean, wavenumbers=candidate_wvn)
        scores[field] = score if math.isfinite(score) else None
        if scores[field] is not None:
            values.append(scores[field])
        coverage_parts.append(
            f"{field}: step{candidate_step}/{reference_step} {candidate_count}c/{reference_count}r"
        )
        if curve_count is None and candidate_count == reference_count:
            curve_count = candidate_count

    available_count = len(values)
    mean = sum(values) / available_count if available_count >= 3 else None
    result: dict[str, Any] = {field: scores.get(field) for field in SPECTRA_FIELDS}
    result.update({
        "mean": mean,
        "source_path": str(comparison_path),
        "coverage": "; ".join(coverage_parts) if coverage_parts else "missing",
        "n_curves": curve_count,
        "count_label": "curves",
        "counts": counts,
        "missing_reference_pairs": {},
        "score_wavenumber_min_exclusive": SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE,
    })
    return result


def _rescore_from_curve_summary(spectra_dir: Path) -> dict[str, Any]:
    """Rescore from spectra_curve_summary.json using adaptive threshold."""
    curve_path = spectra_dir / "spectra_curve_summary.json"
    if not curve_path.exists():
        return empty_spectra_metrics(str(spectra_dir))
    data = load_json(curve_path)
    ws = data.get("weather_states", {})
    if not ws:
        return empty_spectra_metrics(str(curve_path))

    scores: dict[str, float | None] = {}
    values: list[float] = []
    counts: dict[str, int] = {}
    for field in SPECTRA_FIELDS:
        entry, _ = _extract_spectra_summary_entry(ws, field)
        if not entry:
            scores[field] = None
            continue
        scopes = entry.get("scopes", {})
        scope = scopes.get("residual") or scopes.get("full_field") or entry
        wvn_raw = scope.get("wavenumbers")
        pred_raw = scope.get("prediction_mean")
        truth_raw = scope.get("truth_mean")
        if not wvn_raw or not pred_raw or not truth_raw:
            scores[field] = None
            continue
        wvn = np.asarray(wvn_raw, dtype=np.float64)
        pred = np.asarray(pred_raw, dtype=np.float64)
        truth = np.asarray(truth_raw, dtype=np.float64)
        if wvn.shape != pred.shape or wvn.shape != truth.shape:
            scores[field] = None
            continue
        score = relative_l2_weighted(pred, truth, wavenumbers=wvn)
        score_val = finite_float(score)
        scores[field] = score_val
        if score_val is not None:
            values.append(score_val)
        n = scope.get("n_curves")
        if isinstance(n, int):
            counts[field] = n

    available_count = len(values)
    mean = sum(values) / available_count if available_count >= 3 else None
    n_curves_list = list(counts.values())
    result: dict[str, Any] = {field: scores.get(field) for field in SPECTRA_FIELDS}
    result.update({
        "mean": mean,
        "source_path": str(curve_path),
        "coverage": f"{n_curves_list[0]} curves" if n_curves_list and len(set(n_curves_list)) == 1 else "mixed curves",
        "n_curves": n_curves_list[0] if n_curves_list and len(set(n_curves_list)) == 1 else None,
        "count_label": "curves",
        "counts": counts,
        "missing_reference_pairs": {},
        "score_wavenumber_min_exclusive": finite_float(data.get("score_wavenumber_min_exclusive"))
        or SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE,
    })
    return result


def load_spectra_metrics(spectra_dir: Path, *, reference_root: Path | None = None) -> dict[str, Any]:
    summary_path = spectra_dir / "spectra_summary.json"
    comparison_path = spectra_dir / "comparison_summary.json"
    summary_data: dict[str, Any] = {}
    if summary_path.exists():
        summary_data = load_json(summary_path)
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
    if ref_path is not None and ref_path.exists() and _has_raw_spectra_arrays(spectra_dir):
        raw_metrics = _load_raw_spectra_metrics(spectra_dir, ref_path, metadata)
        if raw_metrics["mean"] is not None or any(raw_metrics[field] is not None for field in SPECTRA_FIELDS):
            return raw_metrics
    summary_metrics = empty_spectra_metrics(str(summary_path if summary_path.exists() else spectra_dir))
    if summary_path.exists():
        summary_metrics = _extract_spectra_metrics_from_summary(summary_data, str(summary_path))
        if summary_metrics["mean"] is not None or any(summary_metrics[field] is not None for field in SPECTRA_FIELDS):
            return summary_metrics
    if comparison_path.exists():
        comparison_metrics = _load_reference_model_comparison_metrics(
            spectra_dir,
            comparison_path,
            reference_root=ref_path,
        )
        if comparison_metrics["mean"] is not None or any(comparison_metrics[field] is not None for field in SPECTRA_FIELDS):
            return comparison_metrics
    if summary_metrics["mean"] is not None or any(summary_metrics[field] is not None for field in SPECTRA_FIELDS):
        return summary_metrics
    curve_rescore = _rescore_from_curve_summary(spectra_dir)
    if curve_rescore["mean"] is not None or any(curve_rescore[field] is not None for field in SPECTRA_FIELDS):
        return curve_rescore
    return empty_spectra_metrics(str(summary_path if summary_path.exists() else spectra_dir))


def build_spectra_summary(spectra_dir: Path, *, reference_root: Path | None = None) -> dict[str, Any]:
    metrics = load_spectra_metrics(spectra_dir, reference_root=reference_root)
    relative_to = str(reference_root or "")
    if not relative_to:
        metadata = _load_spectra_metadata(spectra_dir)
        relative_to = str(metadata.get("template_root", "")).strip()

    summary: dict[str, Any] = {
        "method": "ecmwf_mean_curve_reference_weighted_l2",
        "relative_to": relative_to,
        "spectra_output_dir": str(spectra_dir),
        "scoreboard_fields": list(SPECTRA_FIELDS),
        "score_wavenumber_min_exclusive": SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE,
    }
    if metrics["mean"] is not None:
        summary["mean_relative_l2"] = float(metrics["mean"])
    if metrics.get("coverage"):
        summary["coverage"] = metrics["coverage"]

    counts = metrics.get("counts", {})
    missing_reference_pairs = metrics.get("missing_reference_pairs", {})
    count_label = str(metrics.get("count_label", "pairs")).strip() or "pairs"
    count_key = "n_curves" if count_label == "curves" else "n_pairs"
    for field in SPECTRA_FIELDS:
        score = metrics.get(field)
        field_summary: dict[str, Any]
        if score is None:
            field_summary = {"status": "missing"}
        else:
            field_summary = {"relative_l2_mean_curve": float(score)}
        if isinstance(counts.get(field), int):
            field_summary[count_key] = int(counts[field])
        if isinstance(missing_reference_pairs.get(field), int):
            field_summary["missing_reference_pairs"] = int(missing_reference_pairs[field])
        summary[field] = field_summary
    return summary


def _surface_truth_std_map_from_variables(variables: dict[str, Any]) -> dict[str, float]:
    truth_std_by_variable: dict[str, float] = {}
    for variable, entry in variables.items():
        if not isinstance(entry, dict):
            continue
        truth_std = finite_float(entry.get("truth_std"))
        if truth_std is None or truth_std <= 0.0:
            continue
        truth_std_by_variable[variable] = truth_std
    return truth_std_by_variable


def surface_weighted_nmse(
    variables: dict[str, Any],
    *,
    truth_std_by_variable: dict[str, float] | None = None,
) -> tuple[float | None, dict[str, float]]:
    per_variable_nmse: dict[str, float] = {}
    total = 0.0
    any_value = False
    for variable, entry in variables.items():
        if not isinstance(entry, dict):
            continue
        normalized_weight = finite_float(entry.get("normalized_weight"))
        if normalized_weight is None:
            continue
        mean_nmse = finite_float(entry.get("mean_nmse"))
        if mean_nmse is None:
            mean_mse = finite_float(entry.get("mean_mse"))
            truth_std = None
            if truth_std_by_variable is not None:
                truth_std = finite_float(truth_std_by_variable.get(variable))
            if mean_mse is not None and truth_std is not None and truth_std > 0.0:
                mean_nmse = mean_mse / (truth_std * truth_std)
        if mean_nmse is None or not math.isfinite(mean_nmse):
            continue
        per_variable_nmse[variable] = float(mean_nmse)
        total += float(mean_nmse) * normalized_weight
        any_value = True
    return (float(total) if any_value else None), per_variable_nmse


def load_x_interp_surface_metrics(
    predictions_dir: Path,
    *,
    truth_std_by_variable: dict[str, float] | None = None,
) -> dict[str, Any]:
    from eval.jobs import scoreboard_surface_loss as surface_loss

    metrics = surface_loss.process_predictions_dir(
        predictions_dir,
        prediction_var="x_interp",
        truth_var="y",
    )
    variables = metrics.get("variables", {})
    if not isinstance(variables, dict):
        variables = {}
    local_truth_std = _surface_truth_std_map_from_variables(variables)
    weighted_nmse = finite_float(metrics.get("weighted_surface_nmse"))
    if weighted_nmse is None and truth_std_by_variable is not None:
        weighted_nmse, per_variable_nmse = surface_weighted_nmse(
            variables,
            truth_std_by_variable=truth_std_by_variable,
        )
        for variable, value in per_variable_nmse.items():
            entry = variables.get(variable)
            if isinstance(entry, dict) and "mean_nmse" not in entry:
                entry["mean_nmse"] = value
    return {
        "weighted_mse": finite_float(metrics.get("weighted_surface_mse")),
        "weighted_nmse": weighted_nmse,
        "normalization_scheme": str(metrics.get("normalization_scheme", SURFACE_NORMALIZATION_SCHEME)),
        "truth_std_by_variable": local_truth_std,
        "variables": variables,
        "source_path": str(predictions_dir),
    }


def load_surface_loss_metrics(
    surface_json_path: Path,
    *,
    truth_std_by_variable: dict[str, float] | None = None,
) -> dict[str, Any]:
    data = load_json(surface_json_path)
    weighted_mse = finite_float(data.get("weighted_surface_mse"))
    variables = data.get("variables")
    variable_map = variables if isinstance(variables, dict) else {}
    local_truth_std = _surface_truth_std_map_from_variables(variable_map)
    if truth_std_by_variable is None and local_truth_std:
        truth_std_by_variable = local_truth_std
    weighted_nmse = finite_float(data.get("weighted_surface_nmse"))
    per_variable_nmse: dict[str, float] = {}
    if weighted_nmse is None and variable_map:
        weighted_nmse, per_variable_nmse = surface_weighted_nmse(
            variable_map,
            truth_std_by_variable=truth_std_by_variable,
        )
        for variable, value in per_variable_nmse.items():
            entry = variable_map.get(variable)
            if isinstance(entry, dict) and "mean_nmse" not in entry:
                entry["mean_nmse"] = value
    normalization_scheme = str(data.get("normalization_scheme", "")).strip()
    if not normalization_scheme and weighted_nmse is not None:
        normalization_scheme = SURFACE_NORMALIZATION_SCHEME
    top_contributors: list[dict[str, Any]] = []
    if variable_map:
        contributions: list[dict[str, Any]] = []
        use_normalized_components = weighted_nmse is not None
        for variable, entry in variable_map.items():
            if not isinstance(entry, dict):
                continue
            normalized_weight = finite_float(entry.get("normalized_weight"))
            if normalized_weight is None:
                continue
            weighted_component: float | None = None
            if use_normalized_components:
                mean_nmse = finite_float(entry.get("mean_nmse"))
                if mean_nmse is None:
                    mean_nmse = per_variable_nmse.get(variable)
                if mean_nmse is not None and math.isfinite(mean_nmse):
                    weighted_component = mean_nmse * normalized_weight
            if weighted_component is None:
                mean_mse = finite_float(entry.get("mean_mse"))
                if mean_mse is None:
                    continue
                weighted_component = mean_mse * normalized_weight
            contributions.append({
                "variable": variable,
                "label": SURFACE_VAR_LABELS.get(variable, variable),
                "weighted_component": weighted_component,
            })
        contributions.sort(key=lambda item: item["weighted_component"], reverse=True)
        if weighted_mse is None:
            weighted_mse = sum(item["weighted_component"] for item in contributions)
        total_source = weighted_nmse if use_normalized_components else weighted_mse
        total = total_source if total_source is not None and total_source > 0.0 else None
        for item in contributions[:2]:
            contributor = dict(item)
            share = contributor["weighted_component"] / total if total is not None else None
            if share is not None and math.isfinite(share):
                contributor["share"] = share
            top_contributors.append(contributor)
    return {
        "weighted_mse": weighted_mse,
        "weighted_nmse": weighted_nmse,
        "normalization_scheme": normalization_scheme,
        "truth_std_by_variable": truth_std_by_variable or {},
        "variables": variable_map,
        "top_contributors": top_contributors,
        "source_path": str(surface_json_path),
    }


def format_surface_loss_for_scoreboard(surface_metrics: dict[str, Any]) -> str:
    weighted_nmse = finite_float(
        surface_metrics.get("weighted_nmse", surface_metrics.get("weighted_surface_nmse"))
    )
    if weighted_nmse is not None:
        return f"{weighted_nmse:.4f}"
    weighted_mse = finite_float(
        surface_metrics.get("weighted_mse", surface_metrics.get("surface_weighted_mse"))
    )
    if weighted_mse is None:
        return "na"
    return f"{weighted_mse:.6e}"


def load_surface_weighted_mse(surface_json_path: Path) -> float | None:
    return finite_float(load_surface_loss_metrics(surface_json_path).get("weighted_mse"))


def build_run_scoreboard_metrics(
    *,
    run_id: str,
    output_root: Path,
    sigma_run_id: str,
    tc_stats_path: Path,
    spectra_dir: Path,
    surface_json_path: Path,
    event_names: tuple[str, ...] | list[str] | None = None,
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
    if all(v is None for v in sigma_losses.values()):
        run_root_csv = output_root / run_id / "sigma_eval_table.csv"
        if run_root_csv.exists():
            sigma_losses.update(sigma_losses_for_scoreboard(run_root_csv))
    out["sigma_losses"] = sigma_losses

    tc_scores = {}
    if tc_stats_path.exists():
        tc_scores = load_tc_extreme_scores_from_json(tc_stats_path, run_id=run_id, event_names=event_names)
    out["tc_extreme_scores"] = tc_scores

    spectra_metrics = load_spectra_metrics(spectra_dir)
    if spectra_metrics["mean"] is not None:
        out["spectra_mean_relative_l2"] = float(spectra_metrics["mean"])
        out["spectra_relative_l2"] = {
            field: float(spectra_metrics[field])
            for field in SPECTRA_FIELDS
            if spectra_metrics[field] is not None
        }

    surface_metrics = load_surface_loss_metrics(surface_json_path)
    surface_loss = finite_float(surface_metrics.get("weighted_mse"))
    if surface_loss is not None:
        out["surface_weighted_mse"] = surface_loss
    surface_nmse = finite_float(surface_metrics.get("weighted_nmse"))
    if surface_nmse is not None:
        out["surface_weighted_nmse"] = surface_nmse

    return out
