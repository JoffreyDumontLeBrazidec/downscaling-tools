"""spectra_plot_pdf.py — Build a consolidated PDF of spectral curves.

Two modes are auto-detected:
  proxy   : spectra_curve_summary.json exists in spectra_dir
  ecmwf   : ampl_*.npy / wvn_*.npy files exist in subdirectories of spectra_dir

Figures follow the house style of ``eval.plotting``: truth black solid, model red solid,
input blue dashed, x axis "Total wavenumber ℓ" with the wavelength on a top axis. The stored
curves are spectral AMPLITUDES (``sqrt(sum_m |X_nm|^2)`` in the field's native unit) and the
figures show amplitudes, the quantity the scorer's relative L2 error is computed on, in the
variable's display unit (hPa for pressure, dam for geopotential height). Curves are
averaged as amplitudes. Every
multi-page PDF also gets one PNG per page in ``<name>_pages/``. The stored files are only read.

Usage:
    python spectra_plot_pdf.py --spectra-dir <dir> --out-pdf <path>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval.plotting import AXIS, FigureBook, eval_style, role_style, variable_spec  # noqa: E402
from eval.plotting.spec_helpers import (  # noqa: E402
    add_wavelength_axis,
    amplitude_in_display,
    count_phrase,
    spectral_amplitude_label,
)

# Preferred variable order for proxy mode
_VAR_ORDER = ["10u", "10v", "2t", "msl", "sp", "t_850", "z_500"]
_SCOPE_ORDER = ["full_field", "residual"]
_SCOPE_NAMES = {"full_field": "full field", "residual": "residual (field minus interpolated input)"}

_FIGSIZE = (8.0, 5.2)
_GUIDE_COLOR = "0.45"      # neutral grey for reading aids (score threshold, tolerance band)


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def _role_label(role: str, name: str | None) -> str:
    """"Truth (ENFO)" from ("truth", "ENFO"); just "Truth" when the name adds nothing."""
    title = role.capitalize()
    if not name or name.strip().lower() in (role, ""):
        return title
    return f"{title} ({name})"


def _sample_note(files: list[Path]) -> str:
    """"n = 10: 5 dates, 2 lead times" from the curve file names (plain "n = 10" otherwise)."""
    try:
        from eval._backends.spectra import naming
    except Exception:  # pragma: no cover - plotting must not depend on this
        naming = None
    n = len(files)
    parsed = [naming.parse(Path(f).name) for f in files] if naming else [None]
    if not files or any(p is None for p in parsed):
        return f"n = {n}"
    dates = {p["date"] for p in parsed}
    steps = {p["step"] for p in parsed}
    members = {p["member"] for p in parsed}
    parts = [count_phrase(len(dates), "date"), count_phrase(len(steps), "lead time")]
    if len(members) > 1:
        parts.append(count_phrase(len(members), "member"))
    return f"n = {n}: " + ", ".join(parts)


def _finish_spectrum_axes(ax, variable: str | None, *, ylabel: str | None = None) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(AXIS["wavenumber"])
    ax.set_ylabel(ylabel or spectral_amplitude_label(variable))
    ax.grid(True, which="major", color="0.86", linewidth=0.6)
    ax.grid(True, which="minor", color="0.93", linewidth=0.4)
    add_wavelength_axis(ax)


def _title_for(pname: str, what: str) -> str:
    spec = variable_spec(pname)
    return f"{spec.name}: {what}"


def _load_curve_stack(amp_dir: Path, param_name: str):
    """(wavenumbers, amplitude stack [n_curves, n_wvn], amplitude files) or None."""
    d = Path(amp_dir) / param_name
    if not d.exists():
        return None
    ampl_files = sorted(d.glob("ampl_*.npy"))
    if not ampl_files:
        return None
    wvn_files = sorted(d.glob("wvn_*.npy"))
    if wvn_files and len(ampl_files) != len(wvn_files):
        return None
    ampls = [np.load(f) for f in ampl_files]
    if len(set(len(a) for a in ampls)) > 1:
        return None
    if wvn_files:
        wvn = np.mean(np.stack([np.load(f) for f in wvn_files], axis=0), axis=0)
    else:
        wvn = np.arange(len(ampls[0]), dtype=float)
    return wvn, np.stack(ampls, axis=0), ampl_files


def _amplitude_stats(param: str, stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation over curves of the spectral amplitude (display unit)."""
    amp = amplitude_in_display(param, stack)
    return amp.mean(axis=0), amp.std(axis=0)


# ---------------------------------------------------------------------------
# Proxy mode
# ---------------------------------------------------------------------------

def build_pdf_proxy(spectra_dir: Path, out_pdf: Path) -> int:
    """Build PDF from spectra_curve_summary.json.  Returns page count.

    The retired HEALPix proxy stored an unnormalised power spectrum in native units, so the
    y axis says so instead of naming a unit.
    """
    summary_path = spectra_dir / "spectra_curve_summary.json"
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)

    run_label = summary.get("run_label", "")
    score_wvn_min = summary.get("score_wavenumber_min_exclusive", None)
    weather_states: dict = summary.get("weather_states", {})

    # Sort variables: known order first, then remaining alphabetically
    known = [v for v in _VAR_ORDER if v in weather_states]
    extras = sorted(v for v in weather_states if v not in _VAR_ORDER)
    var_order = known + extras

    pages = 0
    with eval_style(), FigureBook(out_pdf, png=True) as book:
        for var in var_order:
            vs = weather_states[var]
            if vs.get("status") != "ok":
                print(f"[WARN] Skipping variable '{var}': status={vs.get('status')}")
                continue
            scopes = vs.get("scopes", {})
            for scope in _SCOPE_ORDER:
                if scope not in scopes:
                    continue
                sc = scopes[scope]
                if sc.get("status") != "ok":
                    print(f"[WARN] Skipping {var}/{scope}: status={sc.get('status')}")
                    continue

                wvn = np.asarray(sc["wavenumbers"], dtype=float)
                pred = np.asarray(sc["prediction_mean"], dtype=float)
                truth = np.asarray(sc["truth_mean"], dtype=float)
                rl2 = sc.get("relative_l2_mean_curve", float("nan"))
                n_curves = sc.get("n_curves", "?")

                # Fix 4: guard against empty mask
                mask = wvn > 0
                if mask.sum() == 0:
                    print(f"[WARN] Skipping {var}/{scope}: no wavenumbers > 0")
                    continue

                # Fix 3: guard against all-NaN or all-zero data after masking
                pred_masked = pred[mask]
                truth_masked = truth[mask]
                if (
                    len(pred_masked) == 0
                    or len(truth_masked) == 0
                    or (np.all(np.isnan(pred_masked)) or np.all(pred_masked == 0))
                    or (np.all(np.isnan(truth_masked)) or np.all(truth_masked == 0))
                ):
                    print(f"[WARN] Skipping {var}/{scope}: no valid data after masking")
                    continue

                fig, ax = plt.subplots(figsize=_FIGSIZE)
                ax.plot(wvn[mask], truth_masked, label=f"Truth (n = {n_curves})",
                        **role_style("truth"))
                ax.plot(wvn[mask], pred_masked, label=f"Model (n = {n_curves})",
                        **role_style("model"))
                if score_wvn_min is not None:
                    ax.axvline(score_wvn_min, color=_GUIDE_COLOR, linestyle=":", linewidth=1.0,
                               label=f"Scored above ℓ = {score_wvn_min:.0f}")
                _finish_spectrum_axes(ax, var, ylabel=f"{AXIS['power']} (proxy, unnormalised)")
                ax.set_title(_title_for(var, f"power spectrum, {_SCOPE_NAMES.get(scope, scope)}"))
                ax.legend(loc="lower left")
                rl2_label = ("relative L2 distance: n/a" if (isinstance(rl2, float) and np.isnan(rl2))
                             else f"relative L2 distance: {rl2:.4f}")
                ax.text(0.98, 0.97, rl2_label, transform=ax.transAxes, ha="right", va="top",
                        fontsize=9, bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                              edgecolor="0.7"))
                if run_label:
                    fig.text(0.01, 0.005, f"run {run_label} · retired HEALPix proxy spectra",
                             fontsize=7, color="0.4", ha="left", va="bottom")
                fig.tight_layout()
                book.add(fig, name=f"{var}_{scope}")
                pages += 1

    return pages


# ---------------------------------------------------------------------------
# ECMWF npy mode
# ---------------------------------------------------------------------------

def build_pdf_ecmwf(spectra_dir: Path, out_pdf: Path) -> int:
    """Build PDF from ampl_*.npy / wvn_*.npy files in param subdirs.

    Each subdirectory that contains ampl_*.npy files becomes one page.
    Returns page count.
    """
    # Find param dirs: subdirs that contain at least one ampl_*.npy file
    param_dirs = sorted(
        d for d in spectra_dir.iterdir()
        if d.is_dir() and list(d.glob("ampl_*.npy"))
    )

    pages = 0
    with eval_style(), FigureBook(out_pdf, png=True) as book:
        for param_dir in param_dirs:
            ampl_files = sorted(param_dir.glob("ampl_*.npy"))
            wvn_files = sorted(param_dir.glob("wvn_*.npy"))

            # Fix 1: guard against wvn/ampl count mismatch
            if wvn_files and len(ampl_files) != len(wvn_files):
                print(
                    f"[WARN] Skipping {param_dir.name}: "
                    f"ampl file count ({len(ampl_files)}) != wvn file count ({len(wvn_files)})"
                )
                continue

            got = _load_curve_stack(spectra_dir, param_dir.name)
            if got is None:
                # Fix 2: guard against inconsistent amplitude array lengths
                print(f"[WARN] Skipping {param_dir.name}: inconsistent amplitude array lengths")
                continue
            wvn, stack, files = got
            mean, std = _amplitude_stats(param_dir.name, stack)

            fig, ax = plt.subplots(figsize=_FIGSIZE)
            mask = wvn > 0
            model = role_style("model")
            ax.plot(wvn[mask], mean[mask], label=f"Model mean ({_sample_note(files)})", **model)
            ax.fill_between(wvn[mask], np.maximum(mean[mask] - std[mask], 1e-30),
                            mean[mask] + std[mask], color=model["color"], alpha=0.15, lw=0,
                            label="Model ±1 standard deviation")
            _finish_spectrum_axes(ax, param_dir.name)
            ax.set_title(_title_for(param_dir.name, "amplitude spectrum"))
            ax.legend(loc="lower left")
            fig.tight_layout()
            book.add(fig, name=param_dir.name)
            pages += 1

    return pages


def _load_mean_curve(amp_dir: Path, param_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load and average all ampl/wvn npy files for a param directory."""
    d = amp_dir / param_name
    if not d.exists():
        return None
    ampl_files = sorted(d.glob("ampl_*.npy"))
    if not ampl_files:
        return None
    wvn_files = sorted(d.glob("wvn_*.npy"))
    ampls = [np.load(f) for f in ampl_files]
    if len(set(len(a) for a in ampls)) > 1:
        return None
    ampl_mean = np.mean(np.stack(ampls, axis=0), axis=0)
    if wvn_files:
        wvn = np.mean(np.stack([np.load(f) for f in wvn_files], axis=0), axis=0)
    else:
        wvn = np.arange(len(ampl_mean), dtype=float)
    return wvn, ampl_mean


def _load_mean_amplitude(amp_dir: Path | None, param_name: str):
    """(wavenumbers, mean amplitude in display units, curve files) of a reference, or None."""
    if amp_dir is None:
        return None
    got = _load_curve_stack(amp_dir, param_name)
    if got is None:
        return None
    wvn, stack, files = got
    mean, _ = _amplitude_stats(param_name, stack)
    return wvn, mean, files


def build_pdf_ecmwf_with_references(
    pred_amp_dir: Path,
    out_pdf: Path,
    *,
    truth_amp_dir: Path | None = None,
    input_amp_dir: Path | None = None,
    truth_label: str = "truth",
    input_label: str = "input",
) -> int:
    """Build PDF with prediction + optional truth/input reference curves.

    One page per parameter: the model's mean amplitude spectrum with a ±1 standard deviation band
    across its fields, the truth (black) and the input (blue dashed) mean spectra.
    """
    param_dirs = sorted(
        d for d in pred_amp_dir.iterdir()
        if d.is_dir() and list(d.glob("ampl_*.npy"))
    )

    pages = 0
    with eval_style(), FigureBook(out_pdf, png=True) as book:
        for param_dir in param_dirs:
            pname = param_dir.name
            got = _load_curve_stack(pred_amp_dir, pname)
            if got is None:
                continue
            wvn, stack, files = got
            pred_mean, pred_std = _amplitude_stats(pname, stack)

            fig, ax = plt.subplots(figsize=_FIGSIZE)
            mask = wvn > 0

            truth = _load_mean_amplitude(truth_amp_dir, pname)
            if truth is not None:
                rwvn, rpow, rfiles = truth
                rmask = rwvn > 0
                ax.plot(rwvn[rmask], rpow[rmask],
                        label=f"{_role_label('truth', truth_label)} ({_sample_note(rfiles)})",
                        **role_style("truth"))

            inp = _load_mean_amplitude(input_amp_dir, pname)
            if inp is not None:
                rwvn, rpow, rfiles = inp
                rmask = rwvn > 0
                ax.plot(rwvn[rmask], rpow[rmask],
                        label=f"{_role_label('input', input_label)} ({_sample_note(rfiles)})",
                        **role_style("input"))

            model = role_style("model")
            ax.plot(wvn[mask], pred_mean[mask], label=f"Model ({_sample_note(files)})", **model)
            ax.fill_between(
                wvn[mask],
                np.maximum(pred_mean[mask] - pred_std[mask], 1e-30),
                pred_mean[mask] + pred_std[mask],
                color=model["color"], alpha=0.15, lw=0, zorder=1,
                label="Model ±1 standard deviation",
            )

            _finish_spectrum_axes(ax, pname)
            ax.set_title(_title_for(pname, "mean amplitude spectrum"))
            ax.legend(loc="lower left")
            fig.tight_layout()
            book.add(fig, name=pname)
            pages += 1

    return pages


# ---------------------------------------------------------------------------
# Ratio mode: every curve divided by the truth curve
# ---------------------------------------------------------------------------

# Shaded tolerance band drawn around one, purely as a reading aid.
_RATIO_GUIDE_BAND = 0.10
# Widest and narrowest vertical window the ratio axis is allowed to open.
_RATIO_YLIM_MIN = 1.25
_RATIO_YLIM_MAX = 20.0


def _default_score_wavenumber_min() -> float | None:
    """Wavenumber above which spectra are scored, or None if unavailable."""
    try:
        from eval._backends.scoreboard.spectra import (
            SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE,
        )
    except Exception:  # pragma: no cover - plotting must not depend on this
        return None
    return float(SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE)


def _ratio_to_reference(
    num_wvn: np.ndarray,
    num_amp: np.ndarray,
    den_wvn: np.ndarray,
    den_amp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Divide one spectrum by another over the wavenumbers they share.

    Both curves are first restricted to strictly positive wavenumbers and
    strictly positive amplitudes, since the ratio is drawn on log axes. The
    denominator is then interpolated onto the numerator wavenumbers in
    log-log space and the two are divided pointwise. When the two grids
    already coincide, which is the normal case because total wavenumbers are
    integers, that interpolation returns the stored values unchanged; it only
    does real work when the driver was truncated differently from the target.
    Returns (wavenumbers, ratio), or None when there is no usable overlap.
    """
    num_wvn = np.asarray(num_wvn, dtype=float)
    num_amp = np.asarray(num_amp, dtype=float)
    den_wvn = np.asarray(den_wvn, dtype=float)
    den_amp = np.asarray(den_amp, dtype=float)
    if num_wvn.shape != num_amp.shape or den_wvn.shape != den_amp.shape:
        return None

    n_ok = np.isfinite(num_wvn) & np.isfinite(num_amp) & (num_wvn > 0) & (num_amp > 0)
    d_ok = np.isfinite(den_wvn) & np.isfinite(den_amp) & (den_wvn > 0) & (den_amp > 0)
    if int(n_ok.sum()) < 2 or int(d_ok.sum()) < 2:
        return None

    nwvn, namp = num_wvn[n_ok], num_amp[n_ok]
    dwvn, damp = den_wvn[d_ok], den_amp[d_ok]
    order = np.argsort(dwvn)
    dwvn, damp = dwvn[order], damp[order]

    lo = max(float(nwvn.min()), float(dwvn.min()))
    hi = min(float(nwvn.max()), float(dwvn.max()))
    keep = (nwvn >= lo) & (nwvn <= hi)
    if int(keep.sum()) < 2:
        return None
    nwvn, namp = nwvn[keep], namp[keep]

    den_here = np.exp(np.interp(np.log(nwvn), np.log(dwvn), np.log(damp)))
    return nwvn, namp / den_here


def _ratio_ylim(series: list[np.ndarray]) -> tuple[float, float]:
    """Pick a vertical window centred on one, in the log sense.

    The window is driven by the bulk of the data rather than its extremes, so
    that a prediction whose amplitude collapses at the truncation does not squash
    everything else into a flat line. Values outside the window run off the
    top or bottom of the plot, which is the honest way to show them.
    """
    pooled = np.concatenate([s for s in series if s.size]) if series else np.array([])
    pooled = pooled[np.isfinite(pooled) & (pooled > 0)]
    if pooled.size == 0:
        return 1.0 / _RATIO_YLIM_MIN, _RATIO_YLIM_MIN
    lo = float(np.percentile(pooled, 2.0))
    hi = float(np.percentile(pooled, 98.0))
    span = max(hi, 1.0 / lo if lo > 0 else _RATIO_YLIM_MIN)
    span = float(np.clip(span * 1.15, _RATIO_YLIM_MIN, _RATIO_YLIM_MAX))
    return 1.0 / span, span


_RATIO_TICKS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0,
                5.0, 10.0, 20.0)
_RATIO_TICKS_MEDIUM = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0)
_RATIO_TICKS_COARSE = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


def _ratio_ticks(lo: float, hi: float) -> list[float]:
    """Readable tick values (at most about eight) for a log ratio axis between ``lo`` and ``hi``."""
    for candidates in (_RATIO_TICKS, _RATIO_TICKS_MEDIUM, _RATIO_TICKS_COARSE):
        ticks = [t for t in candidates if lo <= t <= hi]
        if len(ticks) <= 8:
            break
    return ticks or [1.0]


def build_pdf_ecmwf_ratio(
    pred_amp_dir: Path,
    out_pdf: Path,
    *,
    truth_amp_dir: Path | None = None,
    input_amp_dir: Path | None = None,
    truth_label: str = "truth",
    input_label: str = "input",
    score_wavenumber_min: float | None = None,
) -> int:
    """Build a PDF of spectra expressed as a ratio to the truth spectrum.

    One page per parameter. The mean amplitude spectra of the prediction and of
    the model input are each divided by the mean amplitude spectrum of the truth,
    so a perfect match sits on the horizontal line at one, a curve above one
    carries too much amplitude at that scale and a curve below one carries too
    little. The truth itself is the flat line at one by construction.

    The truth curves are required: without them there is nothing to divide
    by, so the function returns zero pages. Returns the page count.
    """
    if truth_amp_dir is None:
        return 0

    param_dirs = sorted(
        d for d in pred_amp_dir.iterdir()
        if d.is_dir() and list(d.glob("ampl_*.npy"))
    )
    if score_wavenumber_min is None:
        score_wavenumber_min = _default_score_wavenumber_min()

    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    pages = 0
    with eval_style(), FigureBook(out_pdf, png=True) as book:
        for param_dir in param_dirs:
            pname = param_dir.name

            truth = _load_mean_amplitude(truth_amp_dir, pname)
            if truth is None:
                print(f"[WARN] Skipping ratio page for {pname}: no truth curve")
                continue
            twvn, tpow, tfiles = truth

            got = _load_curve_stack(pred_amp_dir, pname)
            if got is None:
                continue
            wvn, stack, files = got
            pred_mean, pred_std = _amplitude_stats(pname, stack)

            pred_ratio = _ratio_to_reference(wvn, pred_mean, twvn, tpow)
            if pred_ratio is None:
                print(f"[WARN] Skipping ratio page for {pname}: no prediction/truth overlap")
                continue

            fig, ax = plt.subplots(figsize=_FIGSIZE)
            drawn: list[np.ndarray] = []

            # Perfect agreement, and a tolerance band to read small departures against.
            ax.axhspan(
                1.0 - _RATIO_GUIDE_BAND, 1.0 + _RATIO_GUIDE_BAND,
                color=_GUIDE_COLOR, alpha=0.12, lw=0, zorder=0,
                label=f"±{_RATIO_GUIDE_BAND * 100:.0f} % of the truth amplitude",
            )
            truth_style = role_style("truth", linewidth=1.8)
            ax.axhline(1.0, label=f"{_role_label('truth', truth_label)} = 1 "
                                  f"({_sample_note(tfiles)})", **truth_style)

            # Input over truth: how far the driver already is from the target.
            inp = _load_mean_amplitude(input_amp_dir, pname)
            if inp is not None:
                got_in = _ratio_to_reference(inp[0], inp[1], twvn, tpow)
                if got_in is not None:
                    iwvn, iratio = got_in
                    ax.plot(iwvn, iratio,
                            label=f"{_role_label('input', input_label)} / truth "
                                  f"({_sample_note(inp[2])})",
                            **role_style("input"))
                    # The vertical window follows the model only: the input's collapse
                    # beyond its own truncation runs off the bottom instead of flattening
                    # the few-percent departures of the model that this page exists to show.

            # Prediction over truth, with the ±1 standard deviation band of the model amplitude
            # carried through the division.
            pwvn, pratio = pred_ratio
            band = _ratio_to_reference(wvn, np.maximum(pred_mean - pred_std, 1e-30), twvn, tpow)
            band_hi = _ratio_to_reference(wvn, pred_mean + pred_std, twvn, tpow)
            model = role_style("model")
            ax.plot(pwvn, pratio, label=f"Model / truth ({_sample_note(files)})", **model)
            if (
                band is not None and band_hi is not None
                and band[1].shape == pratio.shape
                and band_hi[1].shape == pratio.shape
            ):
                ax.fill_between(pwvn, band[1], band_hi[1], color=model["color"], alpha=0.15,
                                lw=0, zorder=1, label="Model ±1 standard deviation")
            drawn.append(pratio)

            if score_wavenumber_min is not None and score_wavenumber_min > 0:
                ax.axvline(
                    score_wavenumber_min, color=_GUIDE_COLOR, linestyle=":", linewidth=1.2,
                    label=f"Scored above ℓ = {score_wavenumber_min:.0f}",
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            lo, hi = _ratio_ylim(drawn)
            ax.set_ylim(lo, hi)
            ax.yaxis.set_major_locator(FixedLocator(_ratio_ticks(lo, hi)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(AXIS["wavenumber"])
            ax.set_ylabel(AXIS["amplitude_ratio"])
            ax.grid(True, which="major", color="0.86", linewidth=0.6)
            add_wavelength_axis(ax)
            ax.set_title(_title_for(pname, "amplitude spectrum relative to the truth"))
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
            fig.tight_layout()
            book.add(fig, name=pname)
            pages += 1

    return pages


# ---------------------------------------------------------------------------
# Top-level auto-detect
# ---------------------------------------------------------------------------

def build_pdf(spectra_dir: Path | str, out_pdf: Path | str) -> int:
    """Auto-detect mode and build consolidated PDF.

    Returns page count.
    Raises FileNotFoundError if neither proxy summary nor npy files are found.
    """
    spectra_dir = Path(spectra_dir)
    out_pdf = Path(out_pdf)

    summary_path = spectra_dir / "spectra_curve_summary.json"
    if summary_path.exists():
        return build_pdf_proxy(spectra_dir, out_pdf)

    # Check for ECMWF npy subdirs
    has_npy = any(
        list(d.glob("ampl_*.npy"))
        for d in spectra_dir.iterdir()
        if d.is_dir()
    ) if spectra_dir.exists() else False

    if has_npy:
        return build_pdf_ecmwf(spectra_dir, out_pdf)

    raise FileNotFoundError(
        f"No spectra data found in {spectra_dir}: "
        "expected spectra_curve_summary.json (proxy mode) or "
        "ampl_*.npy files in subdirectories (ECMWF mode)."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a consolidated spectra PDF from proxy summary or ECMWF npy files."
    )
    parser.add_argument("--spectra-dir", required=True, type=Path, help="Directory with spectra data.")
    parser.add_argument("--out-pdf", required=True, type=Path, help="Output PDF path.")
    args = parser.parse_args()

    n = build_pdf(args.spectra_dir, args.out_pdf)
    print(f"Wrote consolidated PDF ({n} pages): {args.out_pdf}")


if __name__ == "__main__":
    main()
