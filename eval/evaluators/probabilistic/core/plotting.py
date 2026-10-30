"""Plot local probabilistic scores through the shared probabilistic figure.

The lead-time figure itself (layout, role styles, units, wording) lives in
``eval.plotting.probabilistic`` and is shared with the ``quaver`` evaluator; this module
turns the local evaluator's CSV files into the tidy table that function expects, for the
historical metrics (``plot_probabilistic_summary``) and for the per-band calibration read
(``plot_band_scores``). The rank histograms have their own small figure here
(``plot_rank_histograms``).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

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
    lane_config: dict | None = None,
) -> Path:
    """Create the multi-page lead-time PDF (and page PNGs) from ``summary_by_lead.csv``.

    ``lane_config`` (a resolved lane) tells the figure what the truth is; without it the
    source line says only "member 0 of the lane's target ensemble".
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

    curves = summary_to_curves(rows, refs)
    plot_probabilistic_scores(
        curves,
        source_local(truth_from_lane(lane_config)),
        output_pdf,
        title=title_prefix if title_prefix and title_prefix != "Probabilistic scores" else None,
        n_noun="dates",
        band_label=BAND_LABEL,
    )
    return output_pdf


# ------------------------------------------------------------------ rank histograms

_RANK_DOMAIN_ORDER = ("n.hem", "tropics", "s.hem", "europe", "global")


def plot_rank_histograms(hist_csv: str | Path, output_pdf: str | Path, *, title: str | None = None) -> Path | None:
    """One page per variable: rank histograms (rows = leads, columns = regions) from ``rank_histograms.csv``.

    Bars are bin frequencies, the grey line the flat expectation ``1/(m+1)``; each panel's
    title carries the outer-two-bin fraction against ``2/(m+1)`` and the shape word.
    Returns ``None`` when the CSV has no rows (rank histograms were off or no member set).
    """
    import matplotlib.pyplot as plt

    from eval.plotting.roles import MODEL_COLOR, NEUTRAL_COLOR
    from eval.plotting.style import FigureBook, eval_style
    from eval.plotting.variables import variable_spec

    hist_csv = Path(hist_csv)
    rows = _read_csv(hist_csv) if hist_csv.exists() else []
    if not rows:
        return None
    output_pdf = Path(output_pdf)
    variables = list(dict.fromkeys(r["weather_state"] for r in rows))
    with eval_style():
        with FigureBook(output_pdf, png=True) as book:
            for var in variables:
                sub = [r for r in rows if r["weather_state"] == var]
                leads = sorted({int(r["step"]) for r in sub})
                present = {r["domain"] for r in sub}
                domains = [d for d in _RANK_DOMAIN_ORDER if d in present] + sorted(present - set(_RANK_DOMAIN_ORDER))
                fig, axes = plt.subplots(len(leads), len(domains), squeeze=False,
                                         figsize=(3.2 * len(domains) + 0.6, 2.3 * len(leads) + 1.4))
                for i, lead in enumerate(leads):
                    for j, dom in enumerate(domains):
                        ax = axes[i, j]
                        r = next((x for x in sub if int(x["step"]) == lead and x["domain"] == dom), None)
                        if r is None:
                            ax.set_axis_off()
                            continue
                        n_bins = int(r["n_bins"])
                        counts = np.array([_float(r, f"bin_{k}", 0.0) for k in range(n_bins)])
                        freq = counts / counts.sum() if counts.sum() else counts
                        ax.bar(np.arange(n_bins), freq, color=MODEL_COLOR, width=0.85)
                        ax.axhline(1.0 / n_bins, color=NEUTRAL_COLOR, linewidth=1.2, zorder=3)
                        ax.set_title(f"{dom}, +{lead} h: outer {_float(r, 'outer_two_fraction'):.3f} "
                                     f"(flat {_float(r, 'outer_two_expected'):.3f}), {r.get('shape', '')}", fontsize=8)
                        ax.set_xticks([0, n_bins - 1])
                        ax.set_xticklabels(["below all", "above all"], fontsize=8)
                        if j == 0:
                            ax.set_ylabel("frequency")
                        if i == len(leads) - 1:
                            ax.set_xlabel(f"rank of the truth among {n_bins - 1} members")
                head = variable_spec(var).name if variable_spec(var).unit else str(var)
                m = int(sub[0]["n_members"])
                fig.suptitle(f"{head}: rank histograms, {m + 1} bins, ties broken at random "
                             f"(n = {sub[0]['n_dates']} dates)" + (f"\n{title}" if title else ""), fontsize=11)
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                book.add(fig, name=str(var))
    return output_pdf


# -------------------------------------------------------------------- band scores

BAND_LABELS = {"full": "full field", "large": "large part (above the cut)", "fine": "fine part (below the cut)"}
_BAND_METRICS = {"fcrps_mean": "fcrps", "spread_rms": "spread_rms", "rmse_ens_mean": "rmse_ens_mean",
                 "spread_skill_ratio_corrected": "ssr"}


def calibration_to_curves(rows: list[dict[str, str]], *, cut_km: float | None = None) -> list[dict]:
    """Tidy rows for the shared figure from ``calibration_by_lead.csv``: one model series per band.

    Fair CRPS carries its 95 percent band over dates; ``ssr`` is the CORRECTED spread/RMSE
    (``ratio * sqrt((m+1)/m)``, pooled); the input members' spread is an "input" series.
    """
    curves: list[dict] = []
    for r in rows:
        band = r.get("band", "full")
        label = BAND_LABELS.get(band, band)
        if cut_km and band != "full":
            label = label.replace("the cut", f"{cut_km:.0f} km")
        for column, metric in _BAND_METRICS.items():
            value = _float(r, column, float("nan"))
            if not np.isfinite(value):
                continue
            se = _float(r, "fcrps_stderr") if column == "fcrps_mean" else 0.0
            curves.append({
                "metric": metric, "variable": r["weather_state"], "domain": r["domain"],
                "lead_h": _float(r, "step"), "series_role": "model", "series_label": label,
                "value": value,
                "ci_low": value - 1.96 * se if se > 0.0 else None,
                "ci_high": value + 1.96 * se if se > 0.0 else None,
                "n": _float(r, "n_dates", float("nan")),
            })
        inp = _float(r, "spread_input_rms", float("nan"))
        if np.isfinite(inp):
            curves.append({
                "metric": "spread_rms", "variable": r["weather_state"], "domain": r["domain"],
                "lead_h": _float(r, "step"), "series_role": "input",
                "series_label": f"input members, {label}", "value": inp,
                "ci_low": None, "ci_high": None, "n": float("nan"),
            })
    return curves


def plot_band_scores(calibration_csv: str | Path, output_pdf: str | Path, *, lane_config: dict | None = None,
                     title: str | None = None, cut_km: float | None = None) -> Path | None:
    """Fair CRPS, pooled spread, pooled RMSE and corrected spread/RMSE by band and lead.

    Returns ``None`` when the calibration CSV holds only the full band (no scale split ran).
    """
    calibration_csv = Path(calibration_csv)
    rows = _read_csv(calibration_csv) if calibration_csv.exists() else []
    if len({r.get("band", "full") for r in rows}) < 2:
        return None
    curves = calibration_to_curves(rows, cut_km=cut_km)
    plot_probabilistic_scores(
        curves,
        source_local(truth_from_lane(lane_config)),
        Path(output_pdf),
        title=title,
        n_noun="dates",
        band_label="shaded band: 95 % confidence interval of the fair CRPS mean over dates",
        footnote="spread_rms and RMSE: roots of pooled mean squares; ssr: spread/RMSE x sqrt((m+1)/m), 1 = calibrated",
    )
    return Path(output_pdf)
