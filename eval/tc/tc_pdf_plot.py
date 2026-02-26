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
) -> plt.Figure:
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

    # --- plotting
    cmap = cm.batlow
    colors = cmap(np.linspace(0, 1, max(1, len(all_datasets))))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # =====================
    # MSLP
    # =====================
    xbins_msl = np.arange(*cfg.mslp_bin_range)
    mids_msl = (xbins_msl[:-1] + xbins_msl[1:]) / 2

    oper_msl = _finite_1d(dataset_oper["msl"].values / 100.0)
    hist_oper_msl, _ = np.histogram(oper_msl, bins=xbins_msl, density=True)
    oper_min_msl, oper_max_msl = float(oper_msl.min()), float(oper_msl.max())

    s_oper_msl = _tail_summary(oper_msl, tail="low")
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
    axs[0].plot(
        mids_msl,
        _safe_ratio(hist, hist_oper_msl),
        "--",
        linewidth=2,
        color="red",
        label="eefo_o96",
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
    axs[1].plot(
        mids_wind,
        _safe_ratio(hist, hist_oper_wind),
        "--",
        linewidth=2,
        color="red",
        label="eefo_o96",
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
    logger.info("Event %s complete", cfg.name)
    return fig
