# tc_pdf_plot.py
"""
Plotting logic for TC PDF comparisons.

Changes vs your current version:
- Much less logging: only the essentials + a short summary per experiment.
- Wind: no "below_oper" logging (only "above OPER" + fraction above).
- Default behavior: keep your full ML list, but you can pass exclude_ml/include_ml
  to quickly drop known-bad experiments (e.g. j0ys).
- Safer ratio plotting: bins where OPER histogram is zero are set to NaN (no spikes).
"""

import os
import warnings
import logging
import math
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri.cm as cm
import earthkit.data as ekd

try:
    from .tc_events import TCEvent
except ImportError:  # allow running as a script
    from tc_events import TCEvent

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*decode_timedelta will default to False.*",
    category=FutureWarning,
    module="cfgrib.xarray_plugin",
)

sns.set_theme(style="ticks", rc={"font.family": "DejaVu Sans"})
np.seterr(divide="ignore", invalid="ignore")
IDALIA_EXTREME_MSLP_RANGE = (980.0, 990.0)
IDALIA_EXTREME_WIND_MIN = 25.0


# -------------------------
# IO helpers
# -------------------------
def to_xarray_from_files(files):
    logger.debug("Loading %d GRIB files", len(files))
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError("Missing GRIB files:\n" + "\n".join(missing))
    return ekd.from_source("file", files).to_xarray(engine="cfgrib")


def _finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


# -------------------------
# Tail diagnostics (kept, but we log less)
# -------------------------
def _tail_summary(x: np.ndarray, *, tail: str) -> dict:
    """
    tail='low'  -> unusually low values (e.g., MSLP)
    tail='high' -> unusually high values (e.g., wind)
    """
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
        out["bottom0.1_mean"] = (
            float(x[x <= p01].mean()) if np.any(x <= p01) else np.nan
        )
    else:
        raise ValueError("tail must be 'low' or 'high'")

    return out


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Return num/den but NaN where den<=0."""
    out = np.full_like(num, np.nan, dtype=np.float64)
    m = den > 0
    out[m] = num[m] / den[m]
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
    # Convert densities to probability masses to compute robust divergences.
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
    m = 0.5 * (pref + poth)
    js = 0.5 * (
        np.sum(pref * np.log((pref + eps) / (m + eps)))
        + np.sum(poth * np.log((poth + eps) / (m + eps)))
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
    r = ratio[valid]
    return {
        "valid_bins": int(r.size),
        "ratio_mean": float(np.mean(r)),
        "ratio_std": float(np.std(r)),
        "ratio_min": float(np.min(r)),
        "ratio_max": float(np.max(r)),
        "ratio_mae_to_1": float(np.mean(np.abs(r - 1.0))),
        "ratio_max_abs_dev_from_1": float(np.max(np.abs(r - 1.0))),
    }


def _extreme_fraction_mslp(vals: np.ndarray, mslp_range: tuple[float, float]) -> float:
    v = _finite_1d(vals)
    if v.size == 0:
        return math.nan
    lo, hi = mslp_range
    return float(np.mean((v >= lo) & (v <= hi)))


def _extreme_fraction_wind(vals: np.ndarray, wind_gt: float) -> float:
    v = _finite_1d(vals)
    if v.size == 0:
        return math.nan
    return float(np.mean(v > wind_gt))


# -------------------------
# Main plotting
# -------------------------
def plot_event(
    cfg: TCEvent,
    *,
    dir_data_base: str,
    out_path: str,  # kept for API compatibility; not used here (caller may save fig)
    include_ml: Optional[Iterable[str]] = None,
    exclude_ml: Optional[Iterable[str]] = None,
    exp_labels: Optional[dict[str, str]] = None,
    return_stats: bool = False,
) -> plt.Figure | tuple[plt.Figure, dict]:
    """
    Plot per-event PDF ratios (ML + references) for:
      - MSLP (ratio vs OPER analysis)
      - 10m wind speed (ratio vs OPER analysis)

    Logging is intentionally short:
      - OPER support + a compact tail index
      - For each ML exp: support + fraction outside (MSLP) or fraction above (wind)
    """

    exp_labels = exp_labels or {}

    dir_data = os.path.join(dir_data_base, cfg.name)
    logger.info("Event: %s | data: %s", cfg.name, dir_data)

    # --- select ML experiments (preserve order)
    ml_exps = list(include_ml) if include_ml is not None else list(cfg.list_expid_ml)
    if exclude_ml is not None:
        excl = set(exclude_ml)
        ml_exps = [e for e in ml_exps if e not in excl]

    logger.info("ML exps (%d): %s", len(ml_exps), ml_exps)

    # --- load ML datasets
    all_datasets: dict[str, "xarray.Dataset"] = {}
    for exp in ml_exps:
        files = [
            os.path.join(dir_data, f"surface_pf_{exp}_{cfg.year}{cfg.month}{d}.grib")
            for d in cfg.dates
        ]
        all_datasets[exp] = to_xarray_from_files(files)

    # --- load references
    dataset_enfo_o320 = to_xarray_from_files(
        [
            os.path.join(
                dir_data,
                f"surface_pf_{cfg.expid_enfo_o320}_{cfg.year}{cfg.month}{d}.grib",
            )
            for d in cfg.dates
        ]
    )
    dataset_eefo_o96 = to_xarray_from_files(
        [
            os.path.join(
                dir_data,
                f"surface_pf_{cfg.expid_eefo_o96}_{cfg.year}{cfg.month}{d}.grib",
            )
            for d in cfg.dates
        ]
    )
    dataset_oper = to_xarray_from_files(
        [
            os.path.join(dir_data, f"surface_an_{cfg.analysis}_{d}.grib")
            for d in cfg.analysis_dates
        ]
    )
    # Mandatory reference requested by workflow policy.
    dataset_ip6y = to_xarray_from_files(
        [
            os.path.join(
                dir_data,
                f"surface_pf_ENFO_O320_ip6y_{cfg.year}{cfg.month}{d}.grib",
            )
            for d in cfg.dates
        ]
    )

    # --- plotting
    cmap = cm.batlow
    colors = cmap(np.linspace(0, 1, max(1, len(all_datasets))))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    extreme_parts: dict[str, dict[str, float]] = {}
    event_stats = {
        "event": cfg.name,
        "year": cfg.year,
        "month": cfg.month,
        "dates": list(cfg.dates),
        "analysis_dates": list(cfg.analysis_dates),
        "ml_exps": ml_exps,
        "variables": {
            "mslp_hpa": {"curves": {}},
            "wind10m_ms": {"curves": {}},
        },
    }

    # =====================
    # MSLP
    # =====================
    xbins_msl = np.arange(*cfg.mslp_bin_range)
    mids_msl = (xbins_msl[:-1] + xbins_msl[1:]) / 2

    oper_msl = _finite_1d(dataset_oper["msl"].values / 100.0)
    hist_oper_msl, _ = np.histogram(oper_msl, bins=xbins_msl, density=True)
    oper_min_msl, oper_max_msl = float(oper_msl.min()), float(oper_msl.max())

    s_oper_msl = _tail_summary(oper_msl, tail="low")
    extreme_parts["OPER_O320_0001"] = {
        "mslp_980_990_fraction": _extreme_fraction_mslp(oper_msl, IDALIA_EXTREME_MSLP_RANGE),
    }
    event_stats["variables"]["mslp_hpa"]["oper"] = {
        "summary": _summary_stats(oper_msl),
        "tail": s_oper_msl,
    }
    logger.info(
        "MSLP OPER support [%.2f, %.2f] hPa | tail_index=%.3g extreme_index=%.3g",
        oper_min_msl,
        oper_max_msl,
        s_oper_msl.get("tail_index", np.nan),
        s_oper_msl.get("extreme_index", np.nan),
    )

    for i, exp in enumerate(all_datasets.keys()):
        vals = _finite_1d(all_datasets[exp]["msl"].values / 100.0)
        vmin, vmax = float(vals.min()), float(vals.max())

        frac_below = float(np.mean(vals < oper_min_msl))
        frac_above = float(np.mean(vals > oper_max_msl))

        s = _tail_summary(vals, tail="low")
        logger.info(
            "MSLP %-14s support [%.2f, %.2f] | outside: below=%.2e above=%.2e | tail=%.3g ext=%.3g",
            exp,
            vmin,
            vmax,
            frac_below,
            frac_above,
            s.get("tail_index", np.nan),
            s.get("extreme_index", np.nan),
        )

        hist_exp, _ = np.histogram(vals, bins=xbins_msl, density=True)
        event_stats["variables"]["mslp_hpa"]["curves"][exp] = {
            "label": exp_labels.get(exp, exp),
            "summary": _summary_stats(vals),
            "tail": s,
            "vs_oper": {
                **_distribution_metrics(hist_oper_msl, hist_exp, cfg.mslp_bin_range[2]),
                **_ratio_metrics(hist_oper_msl, hist_exp),
            },
        }
        extreme_parts.setdefault(exp, {})["mslp_980_990_fraction"] = _extreme_fraction_mslp(
            vals,
            IDALIA_EXTREME_MSLP_RANGE,
        )
        axs[0].plot(
            mids_msl,
            _safe_ratio(hist_exp, hist_oper_msl),
            linestyle="-",
            linewidth=3,
            label=exp_labels.get(exp, exp),
            color=colors[i],
        )

    # references
    vals = _finite_1d(dataset_enfo_o320["msl"].values / 100.0)
    hist, _ = np.histogram(vals, bins=xbins_msl, density=True)
    event_stats["variables"]["mslp_hpa"]["curves"]["ENFO_O320_0001"] = {
        "label": "enfo_o320",
        "summary": _summary_stats(vals),
        "tail": _tail_summary(vals, tail="low"),
        "vs_oper": {
            **_distribution_metrics(hist_oper_msl, hist, cfg.mslp_bin_range[2]),
            **_ratio_metrics(hist_oper_msl, hist),
        },
    }
    extreme_parts["ENFO_O320_0001"] = {
        "mslp_980_990_fraction": _extreme_fraction_mslp(vals, IDALIA_EXTREME_MSLP_RANGE),
    }
    axs[0].plot(
        mids_msl,
        _safe_ratio(hist, hist_oper_msl),
        "-.",
        linewidth=2,
        color="black",
        label="enfo_o320",
    )

    vals = _finite_1d(dataset_eefo_o96["msl"].values / 100.0)
    hist, _ = np.histogram(vals, bins=xbins_msl, density=True)
    event_stats["variables"]["mslp_hpa"]["curves"]["EEFO_O96_0001"] = {
        "label": "eefo_o96",
        "summary": _summary_stats(vals),
        "tail": _tail_summary(vals, tail="low"),
        "vs_oper": {
            **_distribution_metrics(hist_oper_msl, hist, cfg.mslp_bin_range[2]),
            **_ratio_metrics(hist_oper_msl, hist),
        },
    }
    extreme_parts["EEFO_O96_0001"] = {
        "mslp_980_990_fraction": _extreme_fraction_mslp(vals, IDALIA_EXTREME_MSLP_RANGE),
    }
    axs[0].plot(
        mids_msl,
        _safe_ratio(hist, hist_oper_msl),
        "--",
        linewidth=2,
        color="red",
        label="eefo_o96",
    )
    vals = _finite_1d(dataset_ip6y["msl"].values / 100.0)
    hist, _ = np.histogram(vals, bins=xbins_msl, density=True)
    event_stats["variables"]["mslp_hpa"]["curves"]["ENFO_O320_ip6y"] = {
        "label": "ip6y",
        "summary": _summary_stats(vals),
        "tail": _tail_summary(vals, tail="low"),
        "vs_oper": {
            **_distribution_metrics(hist_oper_msl, hist, cfg.mslp_bin_range[2]),
            **_ratio_metrics(hist_oper_msl, hist),
        },
    }
    extreme_parts["ENFO_O320_ip6y"] = {
        "mslp_980_990_fraction": _extreme_fraction_mslp(vals, IDALIA_EXTREME_MSLP_RANGE),
    }
    axs[0].plot(
        mids_msl,
        _safe_ratio(hist, hist_oper_msl),
        ":",
        linewidth=2,
        color="orange",
        label="ip6y",
    )

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
    axs[0].set_title("Normalized (by AN O320) Distribution MSLP", fontsize=14)
    axs[0].legend()

    # =====================
    # WIND
    # =====================
    xbins_wind = np.arange(*cfg.wind_bin_range)
    mids_wind = (xbins_wind[:-1] + xbins_wind[1:]) / 2

    oper_wind = _finite_1d(
        np.sqrt(dataset_oper["u10"] ** 2 + dataset_oper["v10"] ** 2).values
    )
    hist_oper_wind, _ = np.histogram(oper_wind, bins=xbins_wind, density=True)
    oper_min_w, oper_max_w = float(oper_wind.min()), float(oper_wind.max())

    s_oper_w = _tail_summary(oper_wind, tail="high")
    extreme_parts.setdefault("OPER_O320_0001", {})["wind_gt_25_fraction"] = _extreme_fraction_wind(
        oper_wind,
        IDALIA_EXTREME_WIND_MIN,
    )
    event_stats["variables"]["wind10m_ms"]["oper"] = {
        "summary": _summary_stats(oper_wind),
        "tail": s_oper_w,
    }
    logger.info(
        "WIND OPER support [%.2f, %.2f] m/s | tail_index=%.3g extreme_index=%.3g",
        oper_min_w,
        oper_max_w,
        s_oper_w.get("tail_index", np.nan),
        s_oper_w.get("extreme_index", np.nan),
    )

    for i, exp in enumerate(all_datasets.keys()):
        vals = _finite_1d(
            np.sqrt(
                all_datasets[exp]["u10"] ** 2 + all_datasets[exp]["v10"] ** 2
            ).values
        )
        vmin, vmax = float(vals.min()), float(vals.max())

        above_oper = vmax > oper_max_w
        frac_above = float(np.mean(vals > oper_max_w))

        s = _tail_summary(vals, tail="high")
        # NOTE: no "below_oper" for wind (by request)
        logger.info(
            "WIND %-14s support [%.2f, %.2f] | above_OPER=%s frac_above=%.2e | tail=%.3g ext=%.3g",
            exp,
            vmin,
            vmax,
            str(bool(above_oper)),
            frac_above,
            s.get("tail_index", np.nan),
            s.get("extreme_index", np.nan),
        )

        hist, _ = np.histogram(vals, bins=xbins_wind, density=True)
        event_stats["variables"]["wind10m_ms"]["curves"][exp] = {
            "label": exp_labels.get(exp, exp),
            "summary": _summary_stats(vals),
            "tail": s,
            "vs_oper": {
                **_distribution_metrics(hist_oper_wind, hist, cfg.wind_bin_range[2]),
                **_ratio_metrics(hist_oper_wind, hist),
            },
        }
        extreme_parts.setdefault(exp, {})["wind_gt_25_fraction"] = _extreme_fraction_wind(
            vals,
            IDALIA_EXTREME_WIND_MIN,
        )
        axs[1].plot(
            mids_wind,
            _safe_ratio(hist, hist_oper_wind),
            "-",
            linewidth=3,
            color=colors[i],
            label=exp_labels.get(exp, exp),
        )

    # references
    vals = _finite_1d(
        np.sqrt(dataset_enfo_o320["u10"] ** 2 + dataset_enfo_o320["v10"] ** 2).values
    )
    hist, _ = np.histogram(vals, bins=xbins_wind, density=True)
    event_stats["variables"]["wind10m_ms"]["curves"]["ENFO_O320_0001"] = {
        "label": "enfo_o320",
        "summary": _summary_stats(vals),
        "tail": _tail_summary(vals, tail="high"),
        "vs_oper": {
            **_distribution_metrics(hist_oper_wind, hist, cfg.wind_bin_range[2]),
            **_ratio_metrics(hist_oper_wind, hist),
        },
    }
    extreme_parts.setdefault("ENFO_O320_0001", {})["wind_gt_25_fraction"] = _extreme_fraction_wind(
        vals,
        IDALIA_EXTREME_WIND_MIN,
    )
    axs[1].plot(
        mids_wind,
        _safe_ratio(hist, hist_oper_wind),
        "-.",
        linewidth=2,
        color="black",
        label="enfo_o320",
    )

    vals = _finite_1d(
        np.sqrt(dataset_eefo_o96["u10"] ** 2 + dataset_eefo_o96["v10"] ** 2).values
    )
    hist, _ = np.histogram(vals, bins=xbins_wind, density=True)
    event_stats["variables"]["wind10m_ms"]["curves"]["EEFO_O96_0001"] = {
        "label": "eefo_o96",
        "summary": _summary_stats(vals),
        "tail": _tail_summary(vals, tail="high"),
        "vs_oper": {
            **_distribution_metrics(hist_oper_wind, hist, cfg.wind_bin_range[2]),
            **_ratio_metrics(hist_oper_wind, hist),
        },
    }
    extreme_parts.setdefault("EEFO_O96_0001", {})["wind_gt_25_fraction"] = _extreme_fraction_wind(
        vals,
        IDALIA_EXTREME_WIND_MIN,
    )
    axs[1].plot(
        mids_wind,
        _safe_ratio(hist, hist_oper_wind),
        "--",
        linewidth=2,
        color="red",
        label="eefo_o96",
    )
    vals = _finite_1d(
        np.sqrt(dataset_ip6y["u10"] ** 2 + dataset_ip6y["v10"] ** 2).values
    )
    hist, _ = np.histogram(vals, bins=xbins_wind, density=True)
    event_stats["variables"]["wind10m_ms"]["curves"]["ENFO_O320_ip6y"] = {
        "label": "ip6y",
        "summary": _summary_stats(vals),
        "tail": _tail_summary(vals, tail="high"),
        "vs_oper": {
            **_distribution_metrics(hist_oper_wind, hist, cfg.wind_bin_range[2]),
            **_ratio_metrics(hist_oper_wind, hist),
        },
    }
    extreme_parts.setdefault("ENFO_O320_ip6y", {})["wind_gt_25_fraction"] = _extreme_fraction_wind(
        vals,
        IDALIA_EXTREME_WIND_MIN,
    )
    axs[1].plot(
        mids_wind,
        _safe_ratio(hist, hist_oper_wind),
        ":",
        linewidth=2,
        color="orange",
        label="ip6y",
    )

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
    axs[1].set_title("Normalized (by AN O320) Distribution 10m Wind Speed", fontsize=14)
    axs[1].legend()

    fig.suptitle(cfg.plot_title)
    plt.tight_layout()
    if cfg.name == "idalia":
        rows: list[dict[str, float | str]] = []
        for exp, part in extreme_parts.items():
            m = part.get("mslp_980_990_fraction", math.nan)
            w = part.get("wind_gt_25_fraction", math.nan)
            rows.append(
                {
                    "exp": exp,
                    "mslp_980_990_fraction": float(m),
                    "wind_gt_25_fraction": float(w),
                    "extreme_score": 0.0,
                    "extreme_repro_score": 0.0,
                    "mslp_repro_ratio_vs_ip6y": math.nan,
                    "wind_repro_ratio_vs_ip6y": math.nan,
                }
            )
        m_vals = [r["mslp_980_990_fraction"] for r in rows if np.isfinite(r["mslp_980_990_fraction"])]
        w_vals = [r["wind_gt_25_fraction"] for r in rows if np.isfinite(r["wind_gt_25_fraction"])]
        m_min, m_max = (min(m_vals), max(m_vals)) if m_vals else (math.nan, math.nan)
        w_min, w_max = (min(w_vals), max(w_vals)) if w_vals else (math.nan, math.nan)
        for r in rows:
            m = r["mslp_980_990_fraction"]
            w = r["wind_gt_25_fraction"]
            m_norm = (m - m_min) / (m_max - m_min) if np.isfinite(m) and m_max > m_min else 0.0
            w_norm = (w - w_min) / (w_max - w_min) if np.isfinite(w) and w_max > w_min else 0.0
            r["extreme_score"] = 0.5 * m_norm + 0.5 * w_norm

        # Reference-consistent reproduction score:
        # 1.0 means matching ip6y extreme occurrence rates exactly for both pressure/wind.
        # Falls smoothly as ratios diverge (under- or over-production are both penalized).
        ref_exp = "ENFO_O320_ip6y"
        ref_row = next((r for r in rows if r["exp"] == ref_exp), None)
        eps = 1e-12
        if ref_row is not None:
            ref_m = float(ref_row["mslp_980_990_fraction"])
            ref_w = float(ref_row["wind_gt_25_fraction"])
            for r in rows:
                m = float(r["mslp_980_990_fraction"])
                w = float(r["wind_gt_25_fraction"])
                m_ratio = (m + eps) / (ref_m + eps) if np.isfinite(m) else math.nan
                w_ratio = (w + eps) / (ref_w + eps) if np.isfinite(w) else math.nan
                m_repro = math.exp(-abs(math.log(max(m_ratio, eps)))) if np.isfinite(m_ratio) else 0.0
                w_repro = math.exp(-abs(math.log(max(w_ratio, eps)))) if np.isfinite(w_ratio) else 0.0
                r["mslp_repro_ratio_vs_ip6y"] = m_ratio
                r["wind_repro_ratio_vs_ip6y"] = w_ratio
                r["extreme_repro_score"] = 0.5 * m_repro + 0.5 * w_repro

        rows.sort(
            key=lambda x: (
                float(x.get("extreme_repro_score", 0.0)),
                float(x.get("extreme_score", 0.0)),
            ),
            reverse=True,
        )
        event_stats["extreme_tail"] = {
            "thresholds": {
                "mslp_hpa_range": [IDALIA_EXTREME_MSLP_RANGE[0], IDALIA_EXTREME_MSLP_RANGE[1]],
                "wind_ms_gt": IDALIA_EXTREME_WIND_MIN,
                "reference_exp_for_repro_score": ref_exp,
            },
            "rows": rows,
        }
    logger.info("Event %s complete", cfg.name)
    if return_stats:
        return fig, event_stats
    return fig
