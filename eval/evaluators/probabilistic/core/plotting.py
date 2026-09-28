"""Plot local probabilistic scores through the shared probabilistic figure.

The figure itself (layout, role styles, units, wording) lives in
``eval.plotting.probabilistic`` and is shared with the ``quaver`` evaluator; this module only
turns the local evaluator's CSV files into the tidy table that function expects.
"""
from __future__ import annotations

import csv
from pathlib import Path

from eval.plotting.probabilistic import SOURCE_LOCAL, plot_probabilistic_scores

_METRICS = ("fcrps", "crps", "spread", "rmse_ens_mean")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def summary_to_curves(rows: list[dict[str, str]], ref_rows: list[dict[str, str]] | None = None,
                      *, model_label: str = "model ensemble") -> list[dict]:
    """Tidy rows (see ``eval.plotting.probabilistic``) from ``summary_by_lead.csv`` rows.

    The model curve carries a 95 percent band (mean plus or minus 1.96 standard errors over
    dates) when the standard error is positive. Reference rows (from an exported quaver
    curve file) become "reference" series, one per label.
    """
    curves: list[dict] = []
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


def plot_probabilistic_summary(
    summary_csv: str | Path,
    output_pdf: str | Path,
    *,
    title_prefix: str = "Probabilistic scores",
    reference_curves: str | Path | None = None,
) -> Path:
    """Create the multi-page lead-time PDF (and page PNGs) from ``summary_by_lead.csv``."""
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

    curves = summary_to_curves(rows, refs)
    plot_probabilistic_scores(
        curves,
        SOURCE_LOCAL,
        output_pdf,
        title=title_prefix if title_prefix and title_prefix != "Probabilistic scores" else None,
        n_noun="dates",
    )
    return output_pdf
