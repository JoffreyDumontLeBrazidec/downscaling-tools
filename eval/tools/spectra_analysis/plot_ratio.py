"""Plot the prediction/truth spectral AMPLITUDE ratio against wavenumber.

Reading a deficit off a log-log spectrum is unreliable: a 15% shortfall is a
0.07-decade offset, which is a couple of percent of a four-decade axis, and it
looks bigger or smaller depending on how flat the curve is and how tall the
panel is. The ratio on a linear axis around 1.0 shows exactly where each field
departs and by how much, with no eyeballing.

Reads the spectra_ecmwf evaluator's own .npy output, so it inherits that
evaluator's proper spectral transform rather than any HEALPix approximation.
The stored curves are amplitudes and the figure shows the smoothed amplitude ratio
(the amplitude, not its square, is what the spectra scorer compares) in the house style of ``eval.plotting``, as PNG and PDF.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GROUPS = [
    ("10u_sfc", "10 m zonal wind"),
    ("10v_sfc", "10 m meridional wind"),
    ("2t_sfc", "2 m temperature"),
    ("msl_sfc", "mean sea level pressure"),
    ("t_850", "temperature at 850 hPa"),
    ("z_500", "geopotential at 500 hPa"),
]


def _stack(root: str, grp: str):
    files = sorted(glob.glob(f"{root}/{grp}/ampl_*.npy"))
    if not files:
        return None, None, 0
    amp = np.mean([np.load(f) for f in files], axis=0)
    wvn = np.load(sorted(glob.glob(f"{root}/{grp}/wvn_*.npy"))[0])
    return wvn, amp, len(files)


def _smooth_ratio(w, num, den, width=12):
    """Log-spaced running mean of the power, then square-rooted: the amplitude ratio."""
    r = np.full(len(w), np.nan)
    for i in range(len(w)):
        lo = max(1, int(i / (1.0 + 1.0 / width)))
        hi = min(len(w), int(i * (1.0 + 1.0 / width)) + 1)
        if hi > lo:
            a = np.nansum(num[lo:hi] ** 2)
            b = np.nansum(den[lo:hi] ** 2)
            if b > 0:
                r[i] = np.sqrt(a / b)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-spectra", required=True)
    p.add_argument("--truth-spectra", required=True)
    p.add_argument("--input-spectra", default=None)
    p.add_argument("--label", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from eval.plotting import AXIS, eval_style, role_style, save_figure, variable_spec

    with eval_style():
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), squeeze=False, sharey=True)
        handles = None
        for idx, (grp, _title) in enumerate(GROUPS):
            ax = axes[idx // 3][idx % 3]
            name = variable_spec(grp).name
            w, am, nm = _stack(a.model_spectra, grp)
            _, at, nt = _stack(a.truth_spectra, grp)
            if w is None or at is None:
                ax.set_title(name + " (no data)")
                continue
            n = min(len(w), len(am), len(at))
            w, am, at = w[:n], am[:n], at[:n]
            keep = w >= 2

            ax.axhline(1.0, label="Truth = 1", **role_style("truth", linewidth=1.6))
            ax.semilogx(w[keep], _smooth_ratio(w, am, at)[keep],
                        label="Model / truth", **role_style("model"))
            if a.input_spectra:
                _, ai, _ = _stack(a.input_spectra, grp)
                if ai is not None:
                    ai = ai[:n]
                    ax.semilogx(w[keep], _smooth_ratio(w, ai, at)[keep],
                                label="Coarse input / truth", **role_style("input"))
            ax.axvline(320, color="0.45", lw=1.0, ls=":", label="O320 truncation (ℓ = 320)")
            ax.set_ylim(0.0, 1.8)
            ax.set_xlim(2, max(w))
            ax.set_title(f"{name} (n = {nm} fields)")
            ax.set_xlabel(AXIS["wavenumber"])
            if idx % 3 == 0:
                ax.set_ylabel(AXIS["amplitude_ratio"])
            if handles is None:
                handles = ax.get_legend_handles_labels()

        if handles is not None:
            fig.legend(*handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(
            "Spectral amplitude ratio to the truth" + (f": {a.label}" if a.label else "")
            + "\n1 = the right amount of variance at that scale; dotted line = O320 truncation")
        fig.tight_layout(rect=(0, 0.035, 1, 0.96))
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # writes <out stem>.png (150 dpi) and <out stem>.pdf, as before
        save_figure(fig, out, close=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
