"""spectra_plot_pdf.py — Build a consolidated PDF of spectral curves.

Two modes are auto-detected:
  proxy   : spectra_curve_summary.json exists in spectra_dir
  ecmwf   : ampl_*.npy / wvn_*.npy files exist in subdirectories of spectra_dir

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
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import numpy as np  # noqa: E402


# Preferred variable order for proxy mode
_VAR_ORDER = ["10u", "10v", "2t", "msl", "sp", "t_850", "z_500"]
_SCOPE_ORDER = ["full_field", "residual"]


# ---------------------------------------------------------------------------
# Proxy mode
# ---------------------------------------------------------------------------

def build_pdf_proxy(spectra_dir: Path, out_pdf: Path) -> int:
    """Build PDF from spectra_curve_summary.json.  Returns page count."""
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
    with PdfPages(out_pdf) as pdf:
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

                fig, ax = plt.subplots(figsize=(8, 5))
                # Only plot positive wavenumbers for log-log
                ax.loglog(wvn[mask], pred_masked, label="prediction", color="tab:blue")
                ax.loglog(wvn[mask], truth_masked, label="truth", color="tab:orange", linestyle="--")

                if score_wvn_min is not None:
                    ax.axvline(score_wvn_min, color="gray", linestyle=":", linewidth=0.8, label=f"score ell>{score_wvn_min:.0f}")

                ax.set_xlabel("Wavenumber ℓ")
                ax.set_ylabel("Spectral amplitude")
                ax.set_title(f"{run_label}  |  {var}  |  {scope}  (n={n_curves})")
                ax.legend(fontsize=8)
                rl2_label = "RL2=N/A" if (isinstance(rl2, float) and np.isnan(rl2)) else f"RL2={rl2:.4f}"
                ax.text(
                    0.98, 0.98,
                    rl2_label,
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.7),
                )
                ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
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
    with PdfPages(out_pdf) as pdf:
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

            # Load all amplitude arrays; use matching wvn if available
            ampls = [np.load(f) for f in ampl_files]
            if wvn_files:
                wvns = [np.load(f) for f in wvn_files]
                wvn = np.mean(np.stack(wvns, axis=0), axis=0)
            else:
                wvn = np.arange(len(ampls[0]), dtype=float)

            # Fix 2: guard against inconsistent amplitude array lengths
            ampl_lengths = [len(a) for a in ampls]
            if len(set(ampl_lengths)) > 1:
                print(f"[WARN] Skipping {param_dir.name}: inconsistent amplitude array lengths")
                continue

            ampl_stack = np.stack(ampls, axis=0)
            ampl_mean = np.mean(ampl_stack, axis=0)
            ampl_std = np.std(ampl_stack, axis=0)

            fig, ax = plt.subplots(figsize=(8, 5))
            mask = wvn > 0
            ax.loglog(wvn[mask], ampl_mean[mask], label="mean", color="tab:blue")
            ax.fill_between(
                wvn[mask],
                np.maximum(ampl_mean[mask] - ampl_std[mask], 1e-30),
                ampl_mean[mask] + ampl_std[mask],
                alpha=0.25,
                color="tab:blue",
                label="±1σ",
            )
            ax.set_xlabel("Wavenumber ℓ")
            ax.set_ylabel("Spectral amplitude")
            ax.set_title(f"{param_dir.name}  (n={len(ampl_files)} files)")
            ax.legend(fontsize=8)
            ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
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


def build_pdf_ecmwf_with_references(
    pred_amp_dir: Path,
    out_pdf: Path,
    *,
    truth_amp_dir: Path | None = None,
    input_amp_dir: Path | None = None,
    truth_label: str = "truth",
    input_label: str = "input",
) -> int:
    """Build PDF with prediction + optional truth/input reference curves."""
    param_dirs = sorted(
        d for d in pred_amp_dir.iterdir()
        if d.is_dir() and list(d.glob("ampl_*.npy"))
    )

    pages = 0
    with PdfPages(out_pdf) as pdf:
        for param_dir in param_dirs:
            pname = param_dir.name
            ampl_files = sorted(param_dir.glob("ampl_*.npy"))
            wvn_files = sorted(param_dir.glob("wvn_*.npy"))
            if wvn_files and len(ampl_files) != len(wvn_files):
                continue

            ampls = [np.load(f) for f in ampl_files]
            if len(set(len(a) for a in ampls)) > 1:
                continue
            if wvn_files:
                wvn = np.mean(np.stack([np.load(f) for f in wvn_files], axis=0), axis=0)
            else:
                wvn = np.arange(len(ampls[0]), dtype=float)

            pred_mean = np.mean(np.stack(ampls, axis=0), axis=0)
            pred_std = np.std(np.stack(ampls, axis=0), axis=0)

            fig, ax = plt.subplots(figsize=(8, 5))
            mask = wvn > 0

            # Truth reference
            if truth_amp_dir:
                ref = _load_mean_curve(truth_amp_dir, pname)
                if ref is not None:
                    rwvn, rampl = ref
                    rmask = rwvn > 0
                    ax.loglog(rwvn[rmask], rampl[rmask], label=f"truth ({truth_label})", color="tab:orange", linestyle="--", linewidth=2)

            # Input reference
            if input_amp_dir:
                ref = _load_mean_curve(input_amp_dir, pname)
                if ref is not None:
                    rwvn, rampl = ref
                    rmask = rwvn > 0
                    ax.loglog(rwvn[rmask], rampl[rmask], label=f"input ({input_label})", color="#888888", linestyle="--", linewidth=2)

            # Prediction
            ax.loglog(wvn[mask], pred_mean[mask], label="prediction", color="tab:blue", linewidth=2)
            ax.fill_between(
                wvn[mask],
                np.maximum(pred_mean[mask] - pred_std[mask], 1e-30),
                pred_mean[mask] + pred_std[mask],
                alpha=0.2,
                color="tab:blue",
            )

            ax.set_xlabel("Wavenumber ℓ")
            ax.set_ylabel("Spectral amplitude")
            ax.set_title(f"{pname}  (n={len(ampl_files)})")
            ax.legend(fontsize=8)
            ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
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
    that a prediction whose power collapses at the truncation does not squash
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

    One page per parameter. The prediction and the model input are each
    divided by the truth curve, so a perfect match sits on the horizontal
    line at one, a curve above one carries too much power at that scale and a
    curve below one carries too little. The truth itself is the flat line at
    one by construction and is not drawn as a separate series.

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

    pages = 0
    with PdfPages(out_pdf) as pdf:
        for param_dir in param_dirs:
            pname = param_dir.name

            truth = _load_mean_curve(truth_amp_dir, pname)
            if truth is None:
                print(f"[WARN] Skipping ratio page for {pname}: no truth curve")
                continue
            twvn, tampl = truth

            ampl_files = sorted(param_dir.glob("ampl_*.npy"))
            wvn_files = sorted(param_dir.glob("wvn_*.npy"))
            if wvn_files and len(ampl_files) != len(wvn_files):
                continue
            ampls = [np.load(f) for f in ampl_files]
            if len(set(len(a) for a in ampls)) > 1:
                continue
            if wvn_files:
                wvn = np.mean(np.stack([np.load(f) for f in wvn_files], axis=0), axis=0)
            else:
                wvn = np.arange(len(ampls[0]), dtype=float)
            stack = np.stack(ampls, axis=0)
            pred_mean = np.mean(stack, axis=0)
            pred_std = np.std(stack, axis=0)

            pred_ratio = _ratio_to_reference(wvn, pred_mean, twvn, tampl)
            if pred_ratio is None:
                print(f"[WARN] Skipping ratio page for {pname}: no prediction/truth overlap")
                continue

            from matplotlib.ticker import FuncFormatter

            fig, ax = plt.subplots(figsize=(8, 5))
            drawn: list[np.ndarray] = []

            # Input over truth: how far the driver already is from the target.
            if input_amp_dir:
                ref = _load_mean_curve(input_amp_dir, pname)
                if ref is not None:
                    got = _ratio_to_reference(ref[0], ref[1], twvn, tampl)
                    if got is not None:
                        iwvn, iratio = got
                        ax.plot(
                            iwvn, iratio,
                            label=f"input ({input_label}) / truth",
                            color="#888888", linestyle="--", linewidth=2,
                        )
                        drawn.append(iratio)

            # Prediction over truth, with the same spread band as the
            # absolute plot carried through the division.
            pwvn, pratio = pred_ratio
            band = _ratio_to_reference(wvn, np.maximum(pred_mean - pred_std, 1e-30), twvn, tampl)
            band_hi = _ratio_to_reference(wvn, pred_mean + pred_std, twvn, tampl)
            ax.plot(pwvn, pratio, label="prediction / truth", color="tab:blue", linewidth=2)
            if (
                band is not None and band_hi is not None
                and band[1].shape == pratio.shape
                and band_hi[1].shape == pratio.shape
            ):
                ax.fill_between(pwvn, band[1], band_hi[1], alpha=0.2, color="tab:blue")
            drawn.append(pratio)

            # Perfect agreement, and a tolerance band to read small
            # departures against.
            ax.axhline(1.0, color="tab:orange", linestyle="-", linewidth=1.5,
                       label=f"truth ({truth_label})")
            ax.axhspan(
                1.0 - _RATIO_GUIDE_BAND, 1.0 + _RATIO_GUIDE_BAND,
                color="tab:orange", alpha=0.10, zorder=0,
                label=f"+/-{_RATIO_GUIDE_BAND * 100:.0f}%",
            )
            if score_wavenumber_min is not None and score_wavenumber_min > 0:
                ax.axvline(
                    score_wavenumber_min, color="gray", linestyle=":", linewidth=0.8,
                    label=f"scored above l={score_wavenumber_min:.0f}",
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(*_ratio_ylim(drawn))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
            ax.set_xlabel("Wavenumber l")
            ax.set_ylabel("Spectral amplitude / truth amplitude")
            ax.set_title(f"{pname}  ratio to truth  (n={len(ampl_files)})")
            ax.legend(fontsize=8)
            ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
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
