"""Plot local probabilistic scores through the shared probabilistic figure.

The figure itself (layout, role styles, units, wording) lives in
``eval.plotting.probabilistic`` and is shared with the ``quaver`` evaluator; this module only
turns the local evaluator's CSV files into the tidy table that function expects.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from eval.plotting.probabilistic import plot_probabilistic_scores, source_local

_METRICS = ("fcrps", "crps", "spread", "rmse_ens_mean")

# What the shaded band of the model curve is (see ``summary_to_curves``).
BAND_LABEL = "shaded band: 95 % confidence interval of the mean over dates (mean ± 1.96 standard errors)"

# Forecast systems that can be the truth of a lane, named as they appear in file names.
_STREAM_NAMES = (("iekm", "IEKM"), ("destine", "IEKM"), ("enfo", "ENFO"), ("eefo", "EEFO"))


def _find_key(node, key: str) -> str | None:
    """First non-empty string under ``key`` anywhere in a nested lane configuration."""
    if isinstance(node, dict):
        if isinstance(node.get(key), str) and node[key].strip():
            return node[key].strip()
        for value in node.values():
            found = _find_key(value, key)
            if found:
                return found
    return None


def _find_label(node) -> str | None:
    """Forecast system named by the lane: ``truth_label`` first, then the tc ``target_label``
    (which may carry a grid, as in ``IEKM_O96``, so only its first word is kept)."""
    label = _find_key(node, "truth_label") or _find_key(node, "target_label")
    return re.split(r"[_\s]", label.upper())[0] if label else None


def truth_from_lane(lane_config: dict | None) -> str | None:
    """Name of the truth of the local probabilistic evaluator on a lane, for example "ENFO O1280".

    The truth is member 0 of the target ensemble of the bundle (``y``), and the lane says where
    that comes from: the forecast system in the evaluator sections' ``truth_label`` (or the tc
    section's ``target_label``, or the name of the target GRIB file in ``prepare``), and the
    grid in the target file name or, failing that, in the name of the output template of
    ``prepml``. ``None`` when the lane does not say.
    """
    if not lane_config:
        return None
    args = ((lane_config.get("prepare") or {}).get("args") or {})
    target_file = Path(str(args.get("target_sfc_grib") or "")).name.lower()

    stream = _find_label(lane_config)
    if stream is None:
        stream = next((name for token, name in _STREAM_NAMES if token in target_file), None)
    if stream is None:
        return None

    grid = re.search(r"(?:^|_)(o\d+)(?:_|$)", target_file)
    if grid is None:
        template = Path(str((lane_config.get("prepml") or {}).get("output_template") or "")).name.lower()
        grid = re.match(r"(o\d+)-", template)
    return f"{stream} {grid.group(1).upper()}" if grid else stream


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _members(n) -> str:
    return f", {int(n)} members" if n else ""


def ensemble_labels(model_members: int | None, ref_members: dict[str, int],
                    truth: str | None) -> dict[str, str]:
    """Legend labels of the model and of the two reference ensembles, with member counts."""
    target = truth or "target"
    return {
        "model": f"Model{_members(model_members)}",
        "input": f"Input interpolated to the target grid{_members(ref_members.get('input'))}",
        "target": (f"{target} ensemble without the verifying member 0"
                   f"{_members(ref_members.get('target'))}"),
    }


# Reference ensembles scored by the evaluator: role (hence line style) of each source.
# The target ensemble is drawn black, the colour of the target system in every figure.
_REFERENCE_ROLE = {"input": "input", "target": "truth"}


def summary_to_curves(rows: list[dict[str, str]], ref_rows: list[dict[str, str]] | None = None,
                      *, model_label: str = "model ensemble",
                      ensemble_rows: list[dict[str, str]] | None = None,
                      labels: dict[str, str] | None = None) -> list[dict]:
    """Tidy rows (see ``eval.plotting.probabilistic``) from ``summary_by_lead.csv`` rows.

    The model curve carries a 95 percent band (mean plus or minus 1.96 standard errors over
    dates) when the standard error is positive. Reference rows (from an exported quaver
    curve file) become "reference" series, one per label. ``ensemble_rows`` (from
    ``reference_summary_by_lead.csv``) are the input and target reference ensembles scored
    by the evaluator itself: the input is drawn blue dashed, the target black; no band.
    """
    curves: list[dict] = []
    labels = labels or {}
    for r in ensemble_rows or []:
        source = r.get("source", "")
        if r.get("metric") not in _METRICS or source not in _REFERENCE_ROLE:
            continue
        curves.append({
            "metric": r["metric"], "variable": r["weather_state"], "domain": r["domain"],
            "lead_h": _float(r, "step"), "series_role": _REFERENCE_ROLE[source],
            "series_label": labels.get(source, source),
            "value": _float(r, "mean"), "ci_low": None, "ci_high": None, "n": float("nan"),
        })
    for r in rows:
        if r["metric"] not in _METRICS:
            continue
        mean = _float(r, "mean")
        se = _float(r, "stderr")
        curves.append({
            "metric": r["metric"], "variable": r["weather_state"], "domain": r["domain"],
            "lead_h": _float(r, "step"), "series_role": "model", "series_label": model_label,
            "value": mean,
            "ci_low": mean - 1.96 * se if se > 0.0 else None,
            "ci_high": mean + 1.96 * se if se > 0.0 else None,
            "n": _float(r, "n_dates", float("nan")),
        })
    for r in ref_rows or []:
        if r.get("metric", "") not in _METRICS:
            continue
        curves.append({
            "metric": r["metric"], "variable": r.get("weather_state", ""), "domain": r.get("domain", ""),
            "lead_h": _float(r, "step"), "series_role": "reference",
            "series_label": r.get("label") or "reference",
            "value": _float(r, "value", _float(r, "mean")),
            "ci_low": None, "ci_high": None, "n": float("nan"),
        })
    return curves


def _model_members(summary_csv: Path) -> int | None:
    """Member count of the model, from the first row of the sibling ``scores_by_lead.csv``."""
    scores = summary_csv.with_name("scores_by_lead.csv")
    if not scores.exists():
        return None
    with scores.open(newline="") as f:
        row = next(csv.DictReader(f), None)
    n = int(_float(row or {}, "n_members", 0))
    return n or None


def plot_probabilistic_summary(
    summary_csv: str | Path,
    output_pdf: str | Path,
    *,
    title_prefix: str = "Probabilistic scores",
    reference_curves: str | Path | None = None,
    lane_config: dict | None = None,
) -> Path:
    """Create the multi-page lead-time PDF (and page PNGs) from ``summary_by_lead.csv``.

    ``lane_config`` (a resolved lane) tells the figure what the truth is; without it the
    source line says only "member 0 of the lane's target ensemble". When
    ``reference_summary_by_lead.csv`` sits next to ``summary_csv``, the input and target
    reference ensembles are drawn on every panel with the model.
    """
    summary_csv = Path(summary_csv)
    output_pdf = Path(output_pdf)
    rows = _read_csv(summary_csv)
    if not rows:
        raise ValueError(f"No rows found in {summary_csv}")

    refs: list[dict[str, str]] = []
    if reference_curves:
        ref_path = Path(reference_curves).expanduser()
        if ref_path.exists():
            refs = _read_csv(ref_path)

    ens_rows: list[dict[str, str]] = []
    ref_summary = summary_csv.with_name("reference_summary_by_lead.csv")
    if ref_summary.exists():
        ens_rows = _read_csv(ref_summary)
    ref_members = {}
    for r in ens_rows:
        n = int(_float(r, "n_members", 0))
        if n:
            ref_members[r["source"]] = max(n, ref_members.get(r["source"], 0))
    truth = truth_from_lane(lane_config)
    labels = ensemble_labels(_model_members(summary_csv), ref_members, truth)
    curves = summary_to_curves(rows, refs, model_label=labels["model"],
                               ensemble_rows=ens_rows, labels=labels)
    plot_probabilistic_scores(
        curves,
        source_local(truth),
        output_pdf,
        title=title_prefix if title_prefix and title_prefix != "Probabilistic scores" else None,
        n_noun="dates",
        band_label=BAND_LABEL,
    )
    return output_pdf
