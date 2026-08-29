"""Plot prediction/truth spectral amplitude RATIO against wavenumber.

Reading a deficit off a log-log spectrum is unreliable: a 15% shortfall is a
0.07-decade offset, which is a couple of percent of a four-decade axis, and it
looks bigger or smaller depending on how flat the curve is and how tall the
panel is. The ratio on a linear axis around 1.0 shows exactly where each field
departs and by how much, with no eyeballing.

Reads the spectra_ecmwf evaluator's own .npy output, so it inherits that
evaluator's proper spectral transform rather than any HEALPix approximation.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    """Log-spaced running mean of the POWER ratio, then square-rooted."""
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

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.0), squeeze=False)
    for idx, (grp, title) in enumerate(GROUPS):
        ax = axes[idx // 3][idx % 3]
        w, am, nm = _stack(a.model_spectra, grp)
        _, at, nt = _stack(a.truth_spectra, grp)
        if w is None or at is None:
            ax.set_title(title + " (missing)")
            continue
        n = min(len(w), len(am), len(at))
        w, am, at = w[:n], am[:n], at[:n]
        keep = w >= 2

        ax.semilogx(w[keep], _smooth_ratio(w, am, at)[keep], color="tab:blue", lw=1.8,
                    label="prediction / truth")
        if a.input_spectra:
            _, ai, _ = _stack(a.input_spectra, grp)
            if ai is not None:
                ai = ai[:n]
                ax.semilogx(w[keep], _smooth_ratio(w, ai, at)[keep], color="0.55", lw=1.2,
                            ls="--", label="coarse input / truth")
        ax.axhline(1.0, color="k", lw=0.9, ls=":")
        ax.axvline(320, color="tab:red", lw=0.9, ls=":", alpha=0.7)
        ax.set_ylim(0.0, 1.35)
        ax.set_xlim(2, max(w))
        ax.set_title("%s   (n=%d)" % (title, nm), fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("wavenumber")
        if idx % 3 == 0:
            ax.set_ylabel("amplitude ratio")
        if idx == 0:
            ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "Spectral amplitude ratio to truth  --  %s\n"
        "1.0 = exactly the right amount of structure at that scale; "
        "red dotted line = O320 truncation" % a.label, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
