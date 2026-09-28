"""PDF ratio visualization — matplotlib rendering only."""
from __future__ import annotations

import logging
import warnings

import cmcrameri.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .plot_config import REFERENCE_STYLES, TCPlotConfig
from .stats import safe_ratio

logger = logging.getLogger(__name__)

NAMED_DISTRIBUTION_STYLES: dict[str, dict[str, object]] = {
    "ENFO_O1280_0001": {
        "label": "ENFO O1280",
        "color": "#E69F00",
        "linestyle": "--",
        "linewidth": 2.3,
    },
    "IEKM": {
        "label": "IEKM",
        "color": "#009E73",
        "linestyle": "-.",
        "linewidth": 2.3,
    },
}

MODEL_DISTRIBUTION_COLORS = [
    "#8E24AA",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
]

warnings.filterwarnings(
    "ignore",
    message=".*decode_timedelta will default to False.*",
    category=FutureWarning,
    module="cfgrib.xarray_plugin",
)

sns.set_theme(style="ticks", rc={"font.family": "DejaVu Sans"})
np.seterr(divide="ignore", invalid="ignore")


def _shorten_run_label(label: str) -> str:
    """Shorten a manual_<ckpt8>_..._<date>_<sampler> run label to '<ckpt6> <sampler>'.

    Examples:
        manual_cfec83a3_new_o96_o320_20260320_oldlike200k -> cfec83 oldlike200k
        manual_59e40596_new_o96_o320_20260422_pw20_t10_h7_l13 -> 59e405 pw20_t10_h7_l13
        anemoi_cfec83a3_new_o96_o320_20260323_karras40_direct -> cfec83 karras40_direct
    """
    import re
    m = re.match(r"(?:manual|anemoi)_([0-9a-f]{8})_\w+_o\d+_o\d+_\d{8}_(.+)", label)
    if m:
        return f"{m.group(1)[:6]} {m.group(2)}"
    # Fallback: if longer than 24 chars, take first 24
    if len(label) > 24:
        return label[:24]
    return label


def curve_label(curve_key: str, exp_labels: dict[str, str], *, oper_key: str) -> str:
    if curve_key in NAMED_DISTRIBUTION_STYLES:
        return str(NAMED_DISTRIBUTION_STYLES[curve_key]["label"])
    if curve_key in REFERENCE_STYLES:
        return REFERENCE_STYLES[curve_key]["label"]
    if curve_key == oper_key:
        return "OPER AN"
    if curve_key in exp_labels:
        return exp_labels[curve_key]
    return _shorten_run_label(curve_key)


def curve_style(
    curve_key: str,
    *,
    ml_palette: np.ndarray,
    ml_index: int,
) -> dict[str, object]:
    if curve_key == "OPER_O1280_0001":
        return {"color": "#111827", "linestyle": "-", "linewidth": 2.3}
    if curve_key in NAMED_DISTRIBUTION_STYLES:
        return dict(NAMED_DISTRIBUTION_STYLES[curve_key])
    if curve_key in REFERENCE_STYLES:
        return dict(REFERENCE_STYLES[curve_key])
    return {
        "color": MODEL_DISTRIBUTION_COLORS[ml_index % len(MODEL_DISTRIBUTION_COLORS)],
        "linestyle": "-",
        "linewidth": 3,
    }


def _clean_distribution_label(curve_key: str, exp_labels: dict[str, str], *, oper_key: str) -> str:
    if curve_key == oper_key:
        return "OPER O320" if "O320" in curve_key else "OPER AN"
    if curve_key in NAMED_DISTRIBUTION_STYLES:
        return str(NAMED_DISTRIBUTION_STYLES[curve_key]["label"])
    if curve_key == "ENFO_O320_0001":
        return "ENFO O320"
    if curve_key == "EEFO_O96_0001":
        return "EEFO O96"
    if curve_key in REFERENCE_STYLES:
        return str(REFERENCE_STYLES[curve_key]["label"]).replace("_", " ").upper()
    if curve_key in exp_labels:
        return exp_labels[curve_key]
    return _shorten_run_label(curve_key)


def _distribution_ml_palette(count: int) -> np.ndarray:
    color_cycle = np.asarray(MODEL_DISTRIBUTION_COLORS, dtype=object)
    if count <= len(color_cycle):
        return color_cycle[:count]
    repeats = int(np.ceil(count / len(color_cycle)))
    return np.tile(color_cycle, repeats)[:count]


def _distribution_style(curve_key: str, *, oper_key: str, ml_palette: np.ndarray, ml_index: int) -> dict[str, object]:
    if curve_key == oper_key:
        return {"color": "#000000", "linestyle": "-", "linewidth": 3.0}
    if curve_key in NAMED_DISTRIBUTION_STYLES:
        return dict(NAMED_DISTRIBUTION_STYLES[curve_key])
    if curve_key == "ENFO_O320_0001":
        return {"color": "#E69F00", "linestyle": "-.", "linewidth": 2.0}
    if curve_key in REFERENCE_STYLES:
        base = dict(REFERENCE_STYLES[curve_key])
        if curve_key.startswith("EEFO"):
            base.update({"color": "red", "linestyle": "--", "linewidth": 2.0})
        return base
    return {"color": ml_palette[ml_index], "linestyle": "-", "linewidth": 2.8}


def _apply_distribution_xlim(ax, var_data: dict, *, variable: str) -> None:
    xbins = np.asarray(var_data["bin_edges"], dtype=np.float64)
    if variable == "mslp_hpa":
        if "data_range_msl" in var_data:
            lo, hi = var_data["data_range_msl"]
            left = min(float(xbins[-1]), float(hi) + 5.0)
            right = max(float(xbins[0]), float(lo) - 5.0)
        else:
            left = float(xbins[-1])
            right = float(xbins[0])
        # Lower MSLP means stronger storms; invert so intensity increases rightward.
        ax.set_xlim(left, right)
    elif variable == "wind10m_ms":
        if "data_range_wind" in var_data:
            _lo, hi = var_data["data_range_wind"]
            ax.set_xlim(0, min(float(xbins[-1]), float(hi) + 2.0))
        else:
            ax.set_xlim(float(xbins[0]), float(xbins[-1]))


def _positive_for_log(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.where(arr > 0.0, arr, np.nan)


def _log_density_floor(*series: np.ndarray) -> float:
    positive = []
    for values in series:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr) & (arr > 0.0)]
        if arr.size:
            positive.append(arr)
    if not positive:
        return 1e-12
    return max(float(np.min(np.concatenate(positive))) * 0.1, 1e-12)


def _floor_for_log(values: np.ndarray, *, floor: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(arr) & (arr > 0.0), arr, floor)


def plot_pdf_distribution_overview(
    plot_config: TCPlotConfig,
    *,
    event_stats: dict,
    exp_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Render raw TC distributions with log-density styling for quick visual comparison."""
    exp_labels = exp_labels or {}
    oper_key = event_stats["analysis_key"]
    curve_order = event_stats["curve_order"]
    var_mslp = event_stats["variables"]["mslp_hpa"]
    var_wind = event_stats["variables"]["wind10m_ms"]

    mids_msl = np.asarray(var_mslp["bin_mids"])
    mids_wind = np.asarray(var_wind["bin_mids"])

    ml_like_keys = [
        k for k in curve_order
        if k not in REFERENCE_STYLES and k not in NAMED_DISTRIBUTION_STYLES and k != oper_key
    ]
    ml_palette = _distribution_ml_palette(len(ml_like_keys))
    ml_indices = {k: idx for idx, k in enumerate(ml_like_keys)}

    fig, axs = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)

    series = [oper_key, *curve_order]
    seen: set[str] = set()
    for key in series:
        if key in seen:
            continue
        seen.add(key)
        label = _clean_distribution_label(key, exp_labels, oper_key=oper_key)
        style = _distribution_style(
            key,
            oper_key=oper_key,
            ml_palette=ml_palette,
            ml_index=ml_indices.get(key, 0),
        )

        if key == oper_key:
            hist_msl = np.asarray(var_mslp["oper_histogram"])
            hist_wind = np.asarray(var_wind["oper_histogram"])
        else:
            hist_msl = np.asarray(var_mslp["curves"][key]["histogram"])
            hist_wind = np.asarray(var_wind["curves"][key]["histogram"])

        axs[0].plot(
            mids_msl,
            _positive_for_log(hist_msl),
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=0.96,
        )
        axs[1].plot(
            mids_wind,
            _positive_for_log(hist_wind),
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=0.96,
        )

    for ax, var_data, variable, xlabel, title in [
        (axs[0], var_mslp, "mslp_hpa", "hPa", "Mean sea level pressure (PDF)"),
        (axs[1], var_wind, "wind10m_ms", "m/s", "Wind speed (PDF)"),
    ]:
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("density", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        _apply_distribution_xlim(ax, var_data, variable=variable)

    title = plot_config.plot_title.replace("normed pdfs", "TC distributions")
    fig.suptitle(title, fontsize=16)
    return fig


def plot_pdf_ratios(
    plot_config: TCPlotConfig,
    *,
    event_stats: dict,
    exp_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Render pre-computed event stats as a PDF ratio figure.

    Takes the output of workflows.compute_event_stats().
    """
    exp_labels = exp_labels or {}
    oper_key = event_stats["analysis_key"]
    curve_order = event_stats["curve_order"]
    var_mslp = event_stats["variables"]["mslp_hpa"]
    var_wind = event_stats["variables"]["wind10m_ms"]

    oper_hist_msl = np.asarray(var_mslp["oper_histogram"])
    oper_hist_wind = np.asarray(var_wind["oper_histogram"])
    mids_msl = np.asarray(var_mslp["bin_mids"])
    mids_wind = np.asarray(var_wind["bin_mids"])

    ml_like_keys = [
        k for k in curve_order
        if k not in REFERENCE_STYLES and k not in NAMED_DISTRIBUTION_STYLES
    ]
    ml_palette = _distribution_ml_palette(max(1, len(ml_like_keys)))
    ml_indices = {k: idx for idx, k in enumerate(ml_like_keys)}

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    for key in curve_order:
        label = curve_label(key, exp_labels, oper_key=oper_key)
        style = curve_style(key, ml_palette=ml_palette, ml_index=ml_indices.get(key, 0))

        hist_msl = np.asarray(var_mslp["curves"][key]["histogram"])
        axs[0].plot(
            mids_msl,
            safe_ratio(hist_msl, oper_hist_msl),
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )

        hist_wind = np.asarray(var_wind["curves"][key]["histogram"])
        axs[1].plot(
            mids_wind,
            safe_ratio(hist_wind, oper_hist_wind),
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )

    # Auto-crop x-axis; MSLP is intentionally inverted to match TC intensity semantics.
    _apply_distribution_xlim(axs[0], var_mslp, variable="mslp_hpa")
    _apply_distribution_xlim(axs[1], var_wind, variable="wind10m_ms")

    for ax, mids, ylim, xlabel, title in [
        (axs[0], mids_msl, plot_config.mslp_ylim,
         "Mean Sea Level Pressure (hPa)", "Normalized (by analysis) Distribution MSLP"),
        (axs[1], mids_wind, plot_config.wind_ylim,
         "10m wind speed (m/s)", "Normalized (by analysis) Distribution 10m Wind Speed"),
    ]:
        ax.plot(mids, np.ones_like(mids), "--", linewidth=2, color="#111827", label="OPER AN")
        ydata_max = max(
            (np.nanmax(line.get_ydata()) for line in ax.get_lines() if line.get_ydata().size),
            default=0.0,
        )
        if np.isfinite(ydata_max) and ydata_max > ylim[1]:
            ax.set_yscale("symlog", linthresh=ylim[1])
            ax.set_ylim(0, None)
        else:
            ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Normalized Probability Density", fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.legend()

    fig.suptitle(plot_config.plot_title)
    fig.tight_layout()
    return fig


def plot_pdf_log(
    plot_config: TCPlotConfig,
    *,
    event_stats: dict,
    exp_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Render raw PDFs (no analysis normalisation) with log y-axis."""
    exp_labels = exp_labels or {}
    oper_key = event_stats["analysis_key"]
    curve_order = event_stats["curve_order"]
    var_mslp = event_stats["variables"]["mslp_hpa"]
    var_wind = event_stats["variables"]["wind10m_ms"]

    oper_hist_msl = np.asarray(var_mslp["oper_histogram"])
    oper_hist_wind = np.asarray(var_wind["oper_histogram"])
    mids_msl = np.asarray(var_mslp["bin_mids"])
    mids_wind = np.asarray(var_wind["bin_mids"])

    ml_like_keys = [
        k for k in curve_order
        if k not in REFERENCE_STYLES and k not in NAMED_DISTRIBUTION_STYLES
    ]
    ml_palette = _distribution_ml_palette(max(1, len(ml_like_keys)))
    ml_indices = {k: idx for idx, k in enumerate(ml_like_keys)}

    fig, axs = plt.subplots(1, 2, figsize=(13.8, 5))

    msl_series = [oper_hist_msl, *(np.asarray(var_mslp["curves"][key]["histogram"]) for key in curve_order)]
    wind_series = [oper_hist_wind, *(np.asarray(var_wind["curves"][key]["histogram"]) for key in curve_order)]
    msl_floor = _log_density_floor(*msl_series)
    wind_floor = _log_density_floor(*wind_series)

    oper_label = _clean_distribution_label(oper_key, exp_labels, oper_key=oper_key)
    axs[0].plot(mids_msl, _floor_for_log(oper_hist_msl, floor=msl_floor), "-", linewidth=3.0, color="#000000", label=oper_label)
    axs[1].plot(mids_wind, _floor_for_log(oper_hist_wind, floor=wind_floor), "-", linewidth=3.0, color="#000000", label=oper_label)

    for key in curve_order:
        label = curve_label(key, exp_labels, oper_key=oper_key)
        style = curve_style(key, ml_palette=ml_palette, ml_index=ml_indices.get(key, 0))

        hist_msl = np.asarray(var_mslp["curves"][key]["histogram"])
        axs[0].plot(mids_msl, _floor_for_log(hist_msl, floor=msl_floor), label=label,
                    color=style["color"], linestyle=style["linestyle"],
                    linewidth=style["linewidth"])

        hist_wind = np.asarray(var_wind["curves"][key]["histogram"])
        axs[1].plot(mids_wind, _floor_for_log(hist_wind, floor=wind_floor), label=label,
                    color=style["color"], linestyle=style["linestyle"],
                    linewidth=style["linewidth"])

    _apply_distribution_xlim(axs[0], var_mslp, variable="mslp_hpa")
    _apply_distribution_xlim(axs[1], var_wind, variable="wind10m_ms")

    for ax, floor, xlabel, title in [
        (axs[0], msl_floor, "Mean Sea Level Pressure (hPa)", "Mean sea level pressure (PDF)"),
        (axs[1], wind_floor, "10m wind speed (m/s)", "Wind speed (PDF)"),
    ]:
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
        ax.grid(False)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Probability Density", fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.legend()

    fig.suptitle(plot_config.plot_title.replace("normed pdfs", "TC distributions"))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.83, wspace=0.24)
    return fig


def plot_pdf_single_variable(
    plot_config: TCPlotConfig,
    *,
    event_stats: dict,
    variable: str,
    mode: str = "ratio",
    exp_labels: dict[str, str] | None = None,
    title_suffix: str = "",
) -> plt.Figure:
    """Render ONE variable's tropical-cyclone distribution on its own figure.

    ``plot_pdf_ratios`` and ``plot_pdf_log`` put sea-level pressure and 10 m wind
    speed side by side in one figure.  That is convenient for a quick look but it
    forces wind to share a caption and a title with pressure, and wind is not a
    secondary column on any lane: a change that deepens a cyclone without
    strengthening its wind is a different verdict from one that moves both.  This
    function draws a single variable so wind can carry its own figure, its own
    axis limits and its own caption.

    ``variable`` is a key of ``event_stats["variables"]``, normally ``mslp_hpa``
    or ``wind10m_ms``.  ``mode`` is ``"ratio"`` for curves normalised by the
    analysis, matching ``plot_pdf_ratios``, or ``"log"`` for raw densities on a
    logarithmic axis, matching ``plot_pdf_log``.
    """
    if mode not in ("ratio", "log"):
        raise ValueError(f"mode must be 'ratio' or 'log', got {mode!r}")
    exp_labels = exp_labels or {}
    oper_key = event_stats["analysis_key"]
    curve_order = event_stats["curve_order"]
    var_data = event_stats["variables"][variable]

    oper_hist = np.asarray(var_data["oper_histogram"])
    mids = np.asarray(var_data["bin_mids"])

    ml_like_keys = [
        k for k in curve_order
        if k not in REFERENCE_STYLES and k not in NAMED_DISTRIBUTION_STYLES
    ]
    ml_palette = _distribution_ml_palette(max(1, len(ml_like_keys)))
    ml_indices = {k: idx for idx, k in enumerate(ml_like_keys)}

    is_wind = variable.startswith("wind")
    xlabel = "10 m wind speed (m/s)" if is_wind else "Mean sea level pressure (hPa)"
    ylim = plot_config.wind_ylim if is_wind else plot_config.mslp_ylim

    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    if mode == "log":
        series = [oper_hist, *(np.asarray(var_data["curves"][k]["histogram"]) for k in curve_order)]
        floor = _log_density_floor(*series)
        ax.plot(mids, _floor_for_log(oper_hist, floor=floor), "-", linewidth=3.0,
                color="#000000",
                label=_clean_distribution_label(oper_key, exp_labels, oper_key=oper_key))

    for key in curve_order:
        label = curve_label(key, exp_labels, oper_key=oper_key)
        style = curve_style(key, ml_palette=ml_palette, ml_index=ml_indices.get(key, 0))
        hist = np.asarray(var_data["curves"][key]["histogram"])
        values = (safe_ratio(hist, oper_hist) if mode == "ratio"
                  else _floor_for_log(hist, floor=floor))
        ax.plot(mids, values, label=label, color=style["color"],
                linestyle=style["linestyle"], linewidth=style["linewidth"])

    _apply_distribution_xlim(ax, var_data, variable=variable)
    if mode == "ratio":
        ax.plot(mids, np.ones_like(mids), "--", linewidth=2, color="#111827", label="OPER AN")
        ydata_max = max(
            (np.nanmax(line.get_ydata()) for line in ax.get_lines() if line.get_ydata().size),
            default=0.0,
        )
        if np.isfinite(ydata_max) and ydata_max > ylim[1]:
            ax.set_yscale("symlog", linthresh=ylim[1])
            ax.set_ylim(0, None)
        else:
            ax.set_ylim(*ylim)
        ax.set_ylabel("Probability density divided by the analysis's", fontsize=12)
        kind = "normalised by the analysis"
    else:
        ax.set_yscale("log")
        ax.set_ylabel("Probability density", fontsize=12)
        kind = "raw density, logarithmic axis"

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(f"{'10 m wind speed' if is_wind else 'Sea level pressure'} — {kind}",
                 fontsize=12)
    ax.legend(fontsize=8.5)
    fig.suptitle((plot_config.plot_title + " " + title_suffix).strip())
    fig.tight_layout()
    return fig
