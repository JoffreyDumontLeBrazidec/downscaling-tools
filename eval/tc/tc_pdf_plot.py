"""
Plotting logic for TC PDF comparisons.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Iterable, Optional

import cmcrameri.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from .tc_events import TCEvent
    from .tc_vector_loading import CurveVectors, SupportMode, load_grib_event_curves
except ImportError:  # allow running as a script
    from tc_events import TCEvent
    from tc_vector_loading import CurveVectors, SupportMode, load_grib_event_curves

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*decode_timedelta will default to False.*",
    category=FutureWarning,
    module="cfgrib.xarray_plugin",
)

sns.set_theme(style="ticks", rc={"font.family": "DejaVu Sans"})
np.seterr(divide="ignore", invalid="ignore")

REFERENCE_STYLES = {
    "ENFO_O320_0001": {"label": "enfo_o320", "color": "black", "linestyle": "-.", "linewidth": 2},
    "ENFO_O48_0001": {"label": "enfo_o48", "color": "black", "linestyle": "-.", "linewidth": 2},
    "EEFO_O96_0001": {"label": "eefo_o96", "color": "red", "linestyle": "--", "linewidth": 2},
    "ENFO_O96_0001": {"label": "enfo_o96", "color": "red", "linestyle": "--", "linewidth": 2},
    "ENFO_O320_ip6y": {"label": "ip6y", "color": "orange", "linestyle": ":", "linewidth": 2},
}

IDALIA_EXTREME_MSLP_RANGE = (980.0, 990.0)
IDALIA_EXTREME_WIND_MIN = 25.0


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float64)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def _finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def _tail_summary(x: np.ndarray, *, tail: str) -> dict:
    x = _finite_1d(x)
    if x.size == 0:
        return {"n": 0}

    p = np.percentile(x, [0.1, 1, 5, 50, 95, 99, 99.5, 99.9])
    p01, p1, p5, p50, p95, p99, p995, p999 = p
    out = {
        "n": int(x.size),
        "min": float(x.min()),
        "max": float(x.max()),
        "p0.1": float(p01),
        "p1": float(p1),
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "p99.5": float(p995),
        "p99.9": float(p999),
    }

    eps = 1e-12
    if tail == "high":
        denom = max(p95 - p50, eps)
        out["tail_index"] = float((p99 - p95) / denom)
        out["extreme_index"] = float((p999 - p99) / denom)
        out["top0.1_mean"] = float(x[x >= p999].mean()) if np.any(x >= p999) else np.nan
    elif tail == "low":
        denom = max(p50 - p5, eps)
        out["tail_index"] = float((p5 - p1) / denom)
        out["extreme_index"] = float((p1 - p01) / denom)
        out["bottom0.1_mean"] = float(x[x <= p01].mean()) if np.any(x <= p01) else np.nan
    else:
        raise ValueError("tail must be 'low' or 'high'")
    return out


def _summary_stats(x: np.ndarray) -> dict:
    vals = _finite_1d(x)
    if vals.size == 0:
        return {"n": 0}
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "q01": float(np.quantile(vals, 0.01)),
        "q05": float(np.quantile(vals, 0.05)),
        "q50": float(np.quantile(vals, 0.50)),
        "q95": float(np.quantile(vals, 0.95)),
        "q99": float(np.quantile(vals, 0.99)),
    }


def _distribution_metrics(hist_ref: np.ndarray, hist_other: np.ndarray, bin_width: float) -> dict:
    pref = np.asarray(hist_ref, dtype=np.float64) * bin_width
    poth = np.asarray(hist_other, dtype=np.float64) * bin_width
    sref = float(np.sum(pref))
    soth = float(np.sum(poth))
    if sref <= 0.0 or soth <= 0.0:
        return {
            "l1_mass": math.nan,
            "total_variation": math.nan,
            "rmse_density": math.nan,
            "max_abs_density_diff": math.nan,
            "ks_hist": math.nan,
            "kl_ref_to_other": math.nan,
            "kl_other_to_ref": math.nan,
            "js_divergence": math.nan,
        }
    pref /= sref
    poth /= soth
    diff = pref - poth
    l1 = float(np.sum(np.abs(diff)))
    tv = 0.5 * l1
    max_abs_density_diff = float(np.max(np.abs(hist_ref - hist_other)))
    rmse_density = float(np.sqrt(np.mean((hist_ref - hist_other) ** 2)))
    cdf_ref = np.cumsum(pref)
    cdf_oth = np.cumsum(poth)
    ks_hist = float(np.max(np.abs(cdf_ref - cdf_oth)))
    eps = 1e-12
    kl_ref_to_other = float(np.sum(pref * np.log((pref + eps) / (poth + eps))))
    kl_other_to_ref = float(np.sum(poth * np.log((poth + eps) / (pref + eps))))
    mean_prob = 0.5 * (pref + poth)
    js = 0.5 * (
        np.sum(pref * np.log((pref + eps) / (mean_prob + eps)))
        + np.sum(poth * np.log((poth + eps) / (mean_prob + eps)))
    )
    return {
        "l1_mass": l1,
        "total_variation": tv,
        "rmse_density": rmse_density,
        "max_abs_density_diff": max_abs_density_diff,
        "ks_hist": ks_hist,
        "kl_ref_to_other": kl_ref_to_other,
        "kl_other_to_ref": kl_other_to_ref,
        "js_divergence": float(js),
    }


def _ratio_metrics(hist_ref: np.ndarray, hist_other: np.ndarray) -> dict:
    ratio = _safe_ratio(hist_other, hist_ref)
    valid = np.isfinite(ratio)
    if not np.any(valid):
        return {
            "valid_bins": 0,
            "ratio_mean": math.nan,
            "ratio_std": math.nan,
            "ratio_min": math.nan,
            "ratio_max": math.nan,
            "ratio_mae_to_1": math.nan,
            "ratio_max_abs_dev_from_1": math.nan,
        }
    values = ratio[valid]
    return {
        "valid_bins": int(values.size),
        "ratio_mean": float(np.mean(values)),
        "ratio_std": float(np.std(values)),
        "ratio_min": float(np.min(values)),
        "ratio_max": float(np.max(values)),
        "ratio_mae_to_1": float(np.mean(np.abs(values - 1.0))),
        "ratio_max_abs_dev_from_1": float(np.max(np.abs(values - 1.0))),
    }


def _extreme_fraction_mslp(vals: np.ndarray, mslp_range: tuple[float, float]) -> float:
    vals = _finite_1d(vals)
    if vals.size == 0:
        return math.nan
    lo, hi = mslp_range
    return float(np.mean((vals >= lo) & (vals <= hi)))


def _extreme_fraction_wind(vals: np.ndarray, wind_gt: float) -> float:
    vals = _finite_1d(vals)
    if vals.size == 0:
        return math.nan
    return float(np.mean(vals > wind_gt))


def _curve_label(curve_key: str, exp_labels: dict[str, str], *, oper_key: str) -> str:
    if curve_key in REFERENCE_STYLES:
        return REFERENCE_STYLES[curve_key]["label"]
    if curve_key == oper_key:
        return "OPER AN"
    if curve_key in exp_labels:
        return exp_labels[curve_key]
    return curve_key.replace("ENFO_O320_", "")


def _curve_style(
    curve_key: str,
    *,
    ml_palette: np.ndarray,
    ml_index: int,
) -> dict[str, object]:
    if curve_key in REFERENCE_STYLES:
        return dict(REFERENCE_STYLES[curve_key])
    return {
        "color": ml_palette[ml_index],
        "linestyle": "-",
        "linewidth": 3,
    }


def _variable_stats(
    vals: np.ndarray,
    *,
    hist_ref: np.ndarray,
    bins: np.ndarray,
    bin_width: float,
    tail: str,
) -> tuple[np.ndarray, dict]:
    hist, _ = np.histogram(vals, bins=bins, density=True)
    stats = {
        "summary": _summary_stats(vals),
        "tail": _tail_summary(vals, tail=tail),
        "vs_oper": {
            **_distribution_metrics(hist_ref, hist, bin_width),
            **_ratio_metrics(hist_ref, hist),
        },
    }
    return hist, stats


def _extreme_tail_table(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    rows: list[dict[str, object]] = []
    for exp, (msl_arr, wind_arr) in series.items():
        msl = _finite_1d(msl_arr)
        wind = _finite_1d(wind_arr)
        msl_hit = (msl >= IDALIA_EXTREME_MSLP_RANGE[0]) & (msl <= IDALIA_EXTREME_MSLP_RANGE[1])
        wind_hit = wind > IDALIA_EXTREME_WIND_MIN
        rows.append(
            {
                "exp": exp,
                "mslp_980_990_count": int(np.sum(msl_hit)),
                "mslp_980_990_fraction": float(np.mean(msl_hit)) if msl.size else math.nan,
                "wind_gt_25_count": int(np.sum(wind_hit)),
                "wind_gt_25_fraction": float(np.mean(wind_hit)) if wind.size else math.nan,
                "n_msl": int(msl.size),
                "n_wind": int(wind.size),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["mslp_980_990_fraction"]) if np.isfinite(row["mslp_980_990_fraction"]) else -1.0,
            float(row["wind_gt_25_fraction"]) if np.isfinite(row["wind_gt_25_fraction"]) else -1.0,
        ),
        reverse=True,
    )
    return {
        "thresholds": {
            "mslp_hpa_range": [IDALIA_EXTREME_MSLP_RANGE[0], IDALIA_EXTREME_MSLP_RANGE[1]],
            "wind_ms_gt": IDALIA_EXTREME_WIND_MIN,
        },
        "rows": rows,
    }


def plot_event_curves(
    cfg: TCEvent,
    *,
    curves: dict[str, CurveVectors],
    curve_order: Iterable[str],
    exp_labels: Optional[dict[str, str]] = None,
    return_stats: bool = False,
) -> plt.Figure | tuple[plt.Figure, dict]:
    exp_labels = exp_labels or {}
    oper_key = cfg.analysis
    curve_order = [curve_key for curve_key in curve_order if curve_key in curves and curve_key != oper_key]
    ml_like_keys = [curve_key for curve_key in curve_order if curve_key not in REFERENCE_STYLES]
    ml_palette = cm.batlow(np.linspace(0, 1, max(1, len(ml_like_keys))))
    ml_indices = {curve_key: idx for idx, curve_key in enumerate(ml_like_keys)}

    oper_curve = curves[oper_key]
    oper_msl = _finite_1d(oper_curve.msl)
    oper_wind = _finite_1d(oper_curve.wind)

    xbins_msl = np.arange(*cfg.mslp_bin_range)
    mids_msl = (xbins_msl[:-1] + xbins_msl[1:]) / 2.0
    xbins_wind = np.arange(*cfg.wind_bin_range)
    mids_wind = (xbins_wind[:-1] + xbins_wind[1:]) / 2.0

    hist_oper_msl, _ = np.histogram(oper_msl, bins=xbins_msl, density=True)
    hist_oper_wind, _ = np.histogram(oper_wind, bins=xbins_wind, density=True)

    s_oper_msl = _tail_summary(oper_msl, tail="low")
    s_oper_wind = _tail_summary(oper_wind, tail="high")
    logger.info(
        "MSLP OPER support [%.2f, %.2f] hPa | tail_index=%.3g extreme_index=%.3g",
        float(oper_msl.min()),
        float(oper_msl.max()),
        s_oper_msl.get("tail_index", np.nan),
        s_oper_msl.get("extreme_index", np.nan),
    )
    logger.info(
        "WIND OPER support [%.2f, %.2f] m/s | tail_index=%.3g extreme_index=%.3g",
        float(oper_wind.min()),
        float(oper_wind.max()),
        s_oper_wind.get("tail_index", np.nan),
        s_oper_wind.get("extreme_index", np.nan),
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    event_stats = {
        "event": cfg.name,
        "year": cfg.year,
        "month": cfg.month,
        "dates": list(cfg.dates),
        "analysis_dates": list(cfg.analysis_dates),
        "curve_order": list(curve_order),
        "variables": {
            "mslp_hpa": {
                "oper": {"summary": _summary_stats(oper_msl), "tail": s_oper_msl},
                "curves": {},
            },
            "wind10m_ms": {
                "oper": {"summary": _summary_stats(oper_wind), "tail": s_oper_wind},
                "curves": {},
            },
        },
    }
    extreme_series: dict[str, tuple[np.ndarray, np.ndarray]] = {
        oper_key: (oper_msl, oper_wind),
    }

    for curve_key in curve_order:
        curve = curves[curve_key]
        label = _curve_label(curve_key, exp_labels, oper_key=oper_key)
        style = _curve_style(
            curve_key,
            ml_palette=ml_palette,
            ml_index=ml_indices.get(curve_key, 0),
        )

        vals_msl = _finite_1d(curve.msl)
        hist_msl, msl_stats = _variable_stats(
            vals_msl,
            hist_ref=hist_oper_msl,
            bins=xbins_msl,
            bin_width=cfg.mslp_bin_range[2],
            tail="low",
        )
        logger.info(
            "MSLP %-24s support [%.2f, %.2f] | tail=%.3g ext=%.3g",
            curve_key,
            float(vals_msl.min()),
            float(vals_msl.max()),
            msl_stats["tail"].get("tail_index", np.nan),
            msl_stats["tail"].get("extreme_index", np.nan),
        )
        axs[0].plot(
            mids_msl,
            _safe_ratio(hist_msl, hist_oper_msl),
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
        event_stats["variables"]["mslp_hpa"]["curves"][curve_key] = {
            "label": label,
            **msl_stats,
        }

        vals_wind = _finite_1d(curve.wind)
        hist_wind, wind_stats = _variable_stats(
            vals_wind,
            hist_ref=hist_oper_wind,
            bins=xbins_wind,
            bin_width=cfg.wind_bin_range[2],
            tail="high",
        )
        logger.info(
            "WIND %-24s support [%.2f, %.2f] | tail=%.3g ext=%.3g",
            curve_key,
            float(vals_wind.min()),
            float(vals_wind.max()),
            wind_stats["tail"].get("tail_index", np.nan),
            wind_stats["tail"].get("extreme_index", np.nan),
        )
        axs[1].plot(
            mids_wind,
            _safe_ratio(hist_wind, hist_oper_wind),
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
        event_stats["variables"]["wind10m_ms"]["curves"][curve_key] = {
            "label": label,
            **wind_stats,
        }
        extreme_series[curve_key] = (vals_msl, vals_wind)

    axs[0].plot(
        mids_msl,
        np.ones_like(mids_msl),
        "--",
        linewidth=2,
        color="green",
        label="OPER AN",
    )
    axs[0].set_ylim(*cfg.mslp_ylim)
    axs[0].set_xlabel("Mean Sea Level Pressure (hPa)", fontsize=14)
    axs[0].set_ylabel("Normalized Probability Density", fontsize=14)
    axs[0].set_title("Normalized (by analysis) Distribution MSLP", fontsize=14)
    axs[0].legend()

    axs[1].plot(
        mids_wind,
        np.ones_like(mids_wind),
        "--",
        linewidth=2,
        color="green",
        label="OPER AN",
    )
    axs[1].set_ylim(*cfg.wind_ylim)
    axs[1].set_xlabel("10m wind speed (m/s)", fontsize=14)
    axs[1].set_ylabel("Normalized Probability Density", fontsize=14)
    axs[1].set_title("Normalized (by analysis) Distribution 10m Wind Speed", fontsize=14)
    axs[1].legend()

    fig.suptitle(cfg.plot_title)
    fig.tight_layout()
    event_stats["extreme_tail"] = _extreme_tail_table(extreme_series)

    if return_stats:
        return fig, event_stats
    return fig


def plot_event(
    cfg: TCEvent,
    *,
    dir_data_base: str,
    out_path: str,
    include_ml: Optional[Iterable[str]] = None,
    exclude_ml: Optional[Iterable[str]] = None,
    exp_labels: Optional[dict[str, str]] = None,
    return_stats: bool = False,
    support_mode: SupportMode = "regridded",
) -> plt.Figure | tuple[plt.Figure, dict]:
    del out_path

    exp_labels = exp_labels or {}
    ml_exps = list(include_ml) if include_ml is not None else list(cfg.list_expid_ml)
    if exclude_ml is not None:
        excluded = set(exclude_ml)
        ml_exps = [expid for expid in ml_exps if expid not in excluded]

    logger.info("Event: %s | support=%s | ML=%s", cfg.name, support_mode, ml_exps)
    curves = load_grib_event_curves(
        cfg,
        dir_data_base=dir_data_base,
        ml_exps=ml_exps,
        support_mode=support_mode,
    )
    curve_order = [*ml_exps, *cfg.reference_expids]
    return plot_event_curves(
        cfg,
        curves=curves,
        curve_order=curve_order,
        exp_labels=exp_labels,
        return_stats=return_stats,
    )
