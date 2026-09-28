"""PDF ratio visualization — matplotlib rendering only.

Every curve is drawn by the role it plays (``eval.plotting.roles``): the truth is the bundle
target, i.e. ENFO (black, solid, thick); the downscaling run is the model (red; several runs
take ``sequence_style``); the model's own input is blue and dashed; the operational analysis
(OPER-AN), operational forecasts on other grids and other anchors are references with distinct
styles. On native-support lanes the curve the statistics are normalised by ("analysis key") IS the
bundle target and so is the truth; on regridded lanes it is the operational analysis, a
reference, and the ratio figures say so in their title and axis label.
Legends use readable names, never raw curve keys, and axes carry their units.
"""
from __future__ import annotations

import logging
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np

from eval.plotting import (
    SEQUENCE,
    REFERENCE_STYLES as HOUSE_REFERENCE_STYLES,
    axis_label,
    eval_style,
    pdf_label,
    readable_label,
    reference_style,
    role_style,
    sequence_style,
    shorten_run_label,
)

from .plot_config import REFERENCE_STYLES, TCPlotConfig
from .stats import safe_ratio

logger = logging.getLogger(__name__)

# Named anchors that are not in plot_config.REFERENCE_STYLES. A curve whose key equals the
# analysis key is always drawn as the truth instead.
NAMED_DISTRIBUTION_STYLES: dict[str, dict[str, object]] = {
    "ENFO_O1280_0001": {"label": "ENFO O1280", **reference_style(6)},
    "IEKM": {"label": "IEKM", **reference_style(3)},
}

# Colours for several model runs in one figure (Okabe-Ito based, no black, red or blue).
MODEL_DISTRIBUTION_COLORS = list(SEQUENCE)

warnings.filterwarnings(
    "ignore",
    message=".*decode_timedelta will default to False.*",
    category=FutureWarning,
    module="cfgrib.xarray_plugin",
)

np.seterr(divide="ignore", invalid="ignore")

_STYLE_KEYS = ("color", "linestyle", "linewidth", "zorder")

# Keys that name where predictions were read from rather than which model produced them.
_LEAKED_MODEL_KEYS = {"eval_inputs", "predictions", "prediction", "data", "y_pred", "model", "ml", ""}

_STREAM_KEY = re.compile(r"^(od_)?(enfo|eefo|iekm|oper)(_o\d+)?(_\w+)?$", re.I)


def _shorten_run_label(label: str) -> str:
    """Shorten a manual_<ckpt8>_..._<date>_<sampler> run label to '<ckpt6> <sampler>'.

    Examples:
        manual_cfec83a3_new_o96_o320_20260320_oldlike200k -> cfec83 oldlike200k
        manual_59e40596_new_o96_o320_20260422_pw20_t10_h7_l13 -> 59e405 pw20_t10_h7_l13
        anemoi_cfec83a3_new_o96_o320_20260323_karras40_direct -> cfec83 karras40_direct
    """
    m = re.match(r"(?:manual|anemoi)_([0-9a-f]{8})_\w+_o\d+_o\d+_\d{8}_(.+)", label)
    if m:
        return f"{m.group(1)[:6]} {m.group(2)}"
    # Fallback: if longer than 24 chars, take first 24
    if len(label) > 24:
        return label[:24]
    return label


# ---------------------------------------------------------------------------
# Roles, styles and labels
# ---------------------------------------------------------------------------

def curve_role(curve_key: str, *, analysis_key: str | None = None,
               curve_roles: dict[str, str] | None = None) -> str:
    """Role of a TC curve: ``truth``, ``model``, ``input`` or ``reference``.

    ``curve_roles`` overrides the guess for named keys. Otherwise: "target ..." / "truth ..."
    keys are the truth (the bundle target, ENFO), also when they are the analysis key; an
    analysis key that is not target-like (``OPER_O1280_0001``, ``OPER-AN O1280``) is a
    reference, never the truth; keys starting with "input" (or ``x_interp``) are the input;
    other stream/grid keys (``ENFO_O320_0001``, ``IEKM...``) are references; everything else
    is a model run. ``_figure_styles`` keeps at most one truth per figure.
    """
    if curve_roles and curve_key in curve_roles:
        return curve_roles[curve_key]
    low = str(curve_key).strip().lower()
    if low.startswith(("target", "truth")):
        return "truth"
    if analysis_key is not None and curve_key == analysis_key:
        return "reference"
    if low.startswith("input") or low in ("x", "x_interp", "x_interp_0"):
        return "input"
    if (curve_key in REFERENCE_STYLES or curve_key in NAMED_DISTRIBUTION_STYLES
            or _STREAM_KEY.match(low) or low.startswith(("target", "truth", "oper-an", "oper an"))):
        return "reference"
    return "model"


def _reference_style_map(ref_keys) -> dict[str, dict]:
    """Distinct house reference styles for the reference curves of one figure.

    Keys with a fixed entry (plot_config.REFERENCE_STYLES, NAMED_DISTRIBUTION_STYLES) keep it;
    the others take the first house reference style whose colour and dash are still unused.
    """
    out: dict[str, dict] = {}
    used: set[tuple[str, str]] = set()
    pending = []
    for key in ref_keys:
        fixed = REFERENCE_STYLES.get(key) or NAMED_DISTRIBUTION_STYLES.get(key)
        sig = (str(fixed["color"]).lower(), str(fixed["linestyle"])) if fixed else None
        if fixed and sig not in used:
            out[key] = {k: fixed[k] for k in _STYLE_KEYS if k in fixed}
            used.add(sig)
        else:
            pending.append(key)
    free = [s for s in HOUSE_REFERENCE_STYLES if (s.color.lower(), str(s.linestyle)) not in used]
    for n, key in enumerate(pending):
        style = free[n % len(free)] if free else HOUSE_REFERENCE_STYLES[n % len(HOUSE_REFERENCE_STYLES)]
        out[key] = style.kwargs()
    return out


def _figure_styles(keys, *, analysis_key, curve_roles=None) -> tuple[dict[str, dict], dict[str, str]]:
    """``({key: plot kwargs}, {key: role})`` for every curve drawn in one figure."""
    roles = {k: curve_role(k, analysis_key=analysis_key, curve_roles=curve_roles) for k in keys}
    truths = [k for k in keys if roles[k] == "truth"]
    if not truths:
        # no target curve: an ENFO anchor is the strong-tail truth of this project
        enfo = [k for k in keys if roles[k] == "reference" and re.match(r"^(od_)?enfo", str(k).lower())]
        if enfo:
            roles[enfo[0]] = "truth"
    else:
        for extra in truths[1:]:  # one truth per figure; the others are references
            roles[extra] = "reference"
    models = [k for k in keys if roles[k] == "model"]
    refs = [k for k in keys if roles[k] == "reference"]
    ref_map = _reference_style_map(refs)
    styles: dict[str, dict] = {}
    for key in keys:
        role = roles[key]
        if role in ("truth", "input"):
            styles[key] = role_style(role)
        elif role == "model":
            styles[key] = role_style("model") if len(models) == 1 else sequence_style(models.index(key))
        elif role == "baseline":
            styles[key] = role_style("baseline")
        else:
            styles[key] = ref_map[key]
    return styles, roles


def _plain_name(curve_key: str) -> str:
    """Readable name of a source key: stream/grid ids become 'ENFO O320', others stay."""
    if curve_key in REFERENCE_STYLES:
        return str(REFERENCE_STYLES[curve_key]["label"])
    if curve_key in NAMED_DISTRIBUTION_STYLES:
        return str(NAMED_DISTRIBUTION_STYLES[curve_key]["label"])
    if _STREAM_KEY.match(str(curve_key).lower()):
        return readable_label(curve_key)
    return str(curve_key).replace("_", " ").strip()


def _strip_word(name: str, *words: str) -> str:
    low = name.lower()
    for w in words:
        if low.startswith(w + " "):
            return name[len(w) + 1:].strip()
    return name


def _role_label(curve_key: str, role: str, exp_labels: dict[str, str], *, n_models: int = 1) -> str:
    if curve_key in exp_labels and exp_labels[curve_key]:
        return exp_labels[curve_key]
    if role == "truth":
        return f"truth ({_strip_word(_plain_name(curve_key), 'target', 'truth')})"
    if role == "input":
        rest = _strip_word(_plain_name(curve_key), "input")
        return "input" if rest.lower() in ("", "input") else f"input ({rest})"
    if role == "model":
        if str(curve_key).strip().lower() in _LEAKED_MODEL_KEYS:
            return "model"
        short = shorten_run_label(str(curve_key))
        return short if n_models > 1 else f"model ({short})"
    name = _plain_name(curve_key)
    if name.lower().startswith("target "):
        return f"{_strip_word(name, 'target')} target"
    if name.lower().startswith("oper-an"):
        return f"operational analysis ({name})"
    return name


def curve_label(curve_key: str, exp_labels: dict[str, str], *, oper_key: str,
                curve_roles: dict[str, str] | None = None, n_models: int = 1) -> str:
    """Readable legend label for a curve key (never the raw key)."""
    role = curve_role(curve_key, analysis_key=oper_key, curve_roles=curve_roles)
    return _role_label(curve_key, role, exp_labels or {}, n_models=n_models)


def curve_style(
    curve_key: str,
    *,
    ml_palette: np.ndarray | None = None,
    ml_index: int = 0,
    analysis_key: str | None = None,
    n_models: int = 1,
    curve_roles: dict[str, str] | None = None,
) -> dict[str, object]:
    """Line style of one curve by its role (``ml_palette`` is kept for old callers, unused)."""
    role = curve_role(curve_key, analysis_key=analysis_key, curve_roles=curve_roles)
    if role in ("truth", "input", "baseline"):
        return role_style(role)
    if role == "model":
        return role_style("model") if n_models <= 1 else sequence_style(ml_index)
    return _reference_style_map([curve_key])[curve_key]


def _clean_distribution_label(curve_key: str, exp_labels: dict[str, str], *, oper_key: str) -> str:
    return curve_label(curve_key, exp_labels, oper_key=oper_key)


def _distribution_ml_palette(count: int) -> np.ndarray:
    color_cycle = np.asarray(MODEL_DISTRIBUTION_COLORS, dtype=object)
    if count <= len(color_cycle):
        return color_cycle[:count]
    repeats = int(np.ceil(count / len(color_cycle)))
    return np.tile(color_cycle, repeats)[:count]


def _distribution_style(curve_key: str, *, oper_key: str, ml_palette: np.ndarray, ml_index: int) -> dict[str, object]:
    return curve_style(curve_key, analysis_key=oper_key, ml_index=ml_index,
                       n_models=len(ml_palette) if ml_palette is not None else 1)


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

_VARIABLES = {
    "mslp_hpa": ("msl", "hPa", "Mean sea level pressure"),
    "wind10m_ms": ("10ff", "m s⁻¹", "10 m wind speed"),
}


def _var_meta(variable: str) -> tuple[str, str, str]:
    """(variable-table key, unit, title) for an event_stats variable key."""
    if variable in _VARIABLES:
        return _VARIABLES[variable]
    return ("10ff", "m s⁻¹", "10 m wind speed") if variable.startswith("wind") else ("msl", "hPa", "Mean sea level pressure")


def _ordered_curves(event_stats: dict, *, include_truth: bool, curve_roles=None) -> tuple[list[str], dict, dict]:
    """Curve keys in legend order (truth, models, input, references) with styles and roles."""
    oper_key = event_stats["analysis_key"]
    all_keys = list(dict.fromkeys([oper_key] + list(event_stats["curve_order"])))
    # styles and roles are decided over every curve, the analysis included, so that the
    # analysis (the denominator of the ratio figures) always has a style and a role
    styles, roles = _figure_styles(all_keys, analysis_key=oper_key, curve_roles=curve_roles)
    keys = all_keys if include_truth else [k for k in all_keys if k != oper_key]
    rank = {"truth": 0, "model": 1, "input": 2, "baseline": 3, "reference": 4}
    keys = sorted(keys, key=lambda k: (rank.get(roles[k], 5), all_keys.index(k)))
    return keys, styles, roles


def _curve_count(var_data: dict, key: str, oper_key: str):
    summ = (var_data.get("oper") or {}) if key == oper_key else (var_data.get("curves", {}).get(key) or {})
    n = (summ.get("summary") or {}).get("n")
    return int(n) if isinstance(n, (int, float)) and n == n else None


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} million"
    return f"{n:,}".replace(",", " ")


def _legend(ax, var_data: dict, keys, oper_key: str, **kw) -> None:
    """Legend whose title states the sample count (one line when every curve has the same n)."""
    counts = {k: _curve_count(var_data, k, oper_key) for k in keys}
    known = {n for n in counts.values() if n is not None}
    handles, labels = ax.get_legend_handles_labels()
    title = None
    if len(known) == 1:
        title = f"n = {_fmt_count(known.pop())} grid-point values per curve"
    elif known:
        by_label = {}
        for line in ax.get_lines():
            k = getattr(line, "_tc_key", None)
            if k is not None and counts.get(k) is not None:
                by_label[line.get_label()] = counts[k]
        labels = [f"{lab} (n = {_fmt_count(by_label[lab])})" if lab in by_label else lab for lab in labels]
    kw.setdefault("fontsize", 8.5)
    ax.legend(handles, labels, title=title, title_fontsize=7.5, **kw)


def _plot_curve(ax, x, y, *, key, label, style, alpha=None):
    kw = dict(style)
    if alpha is not None:
        kw["alpha"] = alpha
    (line,) = ax.plot(x, y, label=label, **kw)
    line._tc_key = key  # used by _legend for per-curve counts
    return line


def _title(plot_config: TCPlotConfig, kind: str) -> str:
    """Figure title from the per-event plot title, with the old 'normed pdfs' wording replaced."""
    raw = (plot_config.plot_title or "").strip()
    if not raw:
        return kind[:1].upper() + kind[1:]
    if re.search(r"(?i)normed pdfs|TC distributions", raw):
        return re.sub(r"(?i)normed pdfs|TC distributions", kind, raw)
    return f"{raw}: {kind}"


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


def _hist(var_data: dict, key: str, oper_key: str) -> np.ndarray:
    if key == oper_key:
        return np.asarray(var_data["oper_histogram"])
    return np.asarray(var_data["curves"][key]["histogram"])


def _n_models(roles: dict) -> int:
    return sum(1 for r in roles.values() if r == "model")


def _density_axes(ax, variable: str) -> None:
    vkey, unit, title = _var_meta(variable)
    ax.set_xlabel(axis_label(vkey))
    ax.set_ylabel(pdf_label(unit))
    ax.set_title(title)


def _denominator(oper_key: str, roles: dict, styles: dict, exp_labels: dict) -> tuple[str, str, str, dict]:
    """What the ratio figures divide by: ``(label, phrase, short name, line style)``.

    On native-support lanes the analysis IS the truth ("truth (ENFO O1280)"); on regridded lanes
    it is the operational analysis ("operational analysis (OPER-AN O1280)"), a reference. The
    title and axis label are worded from this so they never call a reference "the truth".
    """
    role = roles.get(oper_key, "reference")
    label = _role_label(oper_key, role, exp_labels)
    short = re.sub(r"\s*\(.*\)\s*$", "", label) or label
    return label, f"the {label}", f"the {short}", styles[oper_key]


def _ratio_axes(ax, variable: str, denominator_short: str) -> None:
    vkey, _unit, title = _var_meta(variable)
    ax.set_xlabel(axis_label(vkey))
    ax.set_ylabel(f"Probability density ratio to {denominator_short}")
    ax.set_title(title)


def _ratio_ylim(ax, ylim) -> None:
    ydata_max = max(
        (np.nanmax(line.get_ydata()) for line in ax.get_lines() if np.size(line.get_ydata())),
        default=0.0,
    )
    if np.isfinite(ydata_max) and ydata_max > ylim[1]:
        ax.set_yscale("symlog", linthresh=ylim[1])
        ax.set_ylim(0, None)
    else:
        ax.set_ylim(*ylim)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_pdf_distribution_overview(
    plot_config: TCPlotConfig,
    *,
    event_stats: dict,
    exp_labels: dict[str, str] | None = None,
    curve_roles: dict[str, str] | None = None,
) -> plt.Figure:
    """Render raw TC distributions with log-density styling for quick visual comparison."""
    exp_labels = exp_labels or {}
    oper_key = event_stats["analysis_key"]
    keys, styles, roles = _ordered_curves(event_stats, include_truth=True, curve_roles=curve_roles)
    n_models = _n_models(roles)

    with eval_style():
        fig, axs = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)
        for ax, variable in zip(axs, ("mslp_hpa", "wind10m_ms")):
            var_data = event_stats["variables"][variable]
            mids = np.asarray(var_data["bin_mids"])
            for key in keys:
                _plot_curve(ax, mids, _positive_for_log(_hist(var_data, key, oper_key)), key=key,
                            label=_role_label(key, roles[key], exp_labels, n_models=n_models),
                            style=styles[key], alpha=0.96)
            ax.set_yscale("log")
            _density_axes(ax, variable)
            _legend(ax, var_data, keys, oper_key, loc="best")
            _apply_distribution_xlim(ax, var_data, variable=variable)
        fig.suptitle(_title(plot_config, "TC distributions"))
    return fig


def plot_pdf_ratios(
    plot_config: TCPlotConfig,
    *,
    event_stats: dict,
    exp_labels: dict[str, str] | None = None,
    curve_roles: dict[str, str] | None = None,
) -> plt.Figure:
    """Render pre-computed event stats as a PDF ratio figure.

    Takes the output of workflows.compute_event_stats(). Every curve is divided by the
    analysis curve, which is drawn as the constant 1 line and named for what it is: the truth
    on native-support lanes, the operational analysis on regridded lanes.
    """
    exp_labels = exp_labels or {}
    oper_key = event_stats["analysis_key"]
    keys, styles, roles = _ordered_curves(event_stats, include_truth=False, curve_roles=curve_roles)
    n_models = _n_models(roles)
    denom_label, denom_phrase, denom_short, denom_style = _denominator(oper_key, roles, styles, exp_labels)

    with eval_style():
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for ax, variable, ylim in ((axs[0], "mslp_hpa", plot_config.mslp_ylim),
                                   (axs[1], "wind10m_ms", plot_config.wind_ylim)):
            var_data = event_stats["variables"][variable]
            mids = np.asarray(var_data["bin_mids"])
            oper_hist = np.asarray(var_data["oper_histogram"])
            _plot_curve(ax, mids, np.ones_like(mids), key=oper_key, label=f"{denom_label} = 1",
                        style=denom_style)
            for key in keys:
                _plot_curve(ax, mids, safe_ratio(_hist(var_data, key, oper_key), oper_hist), key=key,
                            label=_role_label(key, roles[key], exp_labels, n_models=n_models),
                            style=styles[key])
            # Auto-crop x-axis; MSLP is intentionally inverted to match TC intensity semantics.
            _apply_distribution_xlim(ax, var_data, variable=variable)
            _ratio_ylim(ax, ylim)
            _ratio_axes(ax, variable, denom_short)
            _legend(ax, var_data, [oper_key, *keys], oper_key)
        fig.suptitle(_title(plot_config, f"TC distributions divided by {denom_phrase}"))
        fig.tight_layout()
    return fig


def plot_pdf_log(
    plot_config: TCPlotConfig,
    *,
    event_stats: dict,
    exp_labels: dict[str, str] | None = None,
    curve_roles: dict[str, str] | None = None,
) -> plt.Figure:
    """Render raw PDFs (no analysis normalisation) with log y-axis."""
    exp_labels = exp_labels or {}
    oper_key = event_stats["analysis_key"]
    keys, styles, roles = _ordered_curves(event_stats, include_truth=True, curve_roles=curve_roles)
    n_models = _n_models(roles)

    with eval_style():
        fig, axs = plt.subplots(1, 2, figsize=(13.8, 5))
        for ax, variable in zip(axs, ("mslp_hpa", "wind10m_ms")):
            var_data = event_stats["variables"][variable]
            mids = np.asarray(var_data["bin_mids"])
            floor = _log_density_floor(*(_hist(var_data, k, oper_key) for k in keys))
            for key in keys:
                _plot_curve(ax, mids, _floor_for_log(_hist(var_data, key, oper_key), floor=floor), key=key,
                            label=_role_label(key, roles[key], exp_labels, n_models=n_models),
                            style=styles[key])
            _apply_distribution_xlim(ax, var_data, variable=variable)
            ax.set_yscale("log")
            ax.set_ylim(bottom=floor)
            _density_axes(ax, variable)
            _legend(ax, var_data, keys, oper_key)
        fig.suptitle(_title(plot_config, "TC distributions"))
        fig.subplots_adjust(left=0.07, right=0.985, bottom=0.14, top=0.86, wspace=0.24)
    return fig


def plot_pdf_single_variable(
    plot_config: TCPlotConfig,
    *,
    event_stats: dict,
    variable: str,
    mode: str = "ratio",
    exp_labels: dict[str, str] | None = None,
    title_suffix: str = "",
    curve_roles: dict[str, str] | None = None,
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

    The figure uses no layout engine (only ``tight_layout``), so callers may still
    enlarge it and call ``subplots_adjust`` to add a caption.
    """
    if mode not in ("ratio", "log"):
        raise ValueError(f"mode must be 'ratio' or 'log', got {mode!r}")
    exp_labels = exp_labels or {}
    oper_key = event_stats["analysis_key"]
    var_data = event_stats["variables"][variable]
    oper_hist = np.asarray(var_data["oper_histogram"])
    mids = np.asarray(var_data["bin_mids"])
    keys, styles, roles = _ordered_curves(event_stats, include_truth=(mode == "log"), curve_roles=curve_roles)
    n_models = _n_models(roles)
    denom_label, denom_phrase, denom_short, denom_style = _denominator(oper_key, roles, styles, exp_labels)
    is_wind = variable.startswith("wind")
    ylim = plot_config.wind_ylim if is_wind else plot_config.mslp_ylim

    with eval_style():
        fig, ax = plt.subplots(figsize=(8.6, 5.6))
        if mode == "log":
            floor = _log_density_floor(*(_hist(var_data, k, oper_key) for k in keys))
            for key in keys:
                _plot_curve(ax, mids, _floor_for_log(_hist(var_data, key, oper_key), floor=floor), key=key,
                            label=_role_label(key, roles[key], exp_labels, n_models=n_models),
                            style=styles[key])
            ax.set_yscale("log")
            _density_axes(ax, variable)
            kind = "probability density, logarithmic axis"
            legend_keys = keys
        else:
            _plot_curve(ax, mids, np.ones_like(mids), key=oper_key, label=f"{denom_label} = 1",
                        style=denom_style)
            for key in keys:
                _plot_curve(ax, mids, safe_ratio(_hist(var_data, key, oper_key), oper_hist), key=key,
                            label=_role_label(key, roles[key], exp_labels, n_models=n_models),
                            style=styles[key])
            _ratio_ylim(ax, ylim)
            _ratio_axes(ax, variable, denom_short)
            kind = f"probability density divided by {denom_phrase}"
            legend_keys = [oper_key, *keys]

        _apply_distribution_xlim(ax, var_data, variable=variable)
        ax.set_title(f"{_var_meta(variable)[2]}: {kind}")
        _legend(ax, var_data, legend_keys, oper_key)
        fig.suptitle((_title(plot_config, "TC distributions") + " " + title_suffix).strip())
        fig.tight_layout()
    return fig
