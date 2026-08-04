#!/usr/bin/env python3
"""TC extreme distributions with member/file-level statistical units."""
from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from interp.core.geometry import box_mask_km, detect_min_center, norm_lon

PRED_DIR = "/home/ecm5702/scratch/eval/manual_eecdb127_new_o320_o1280_20260503_manual_eval/data/predictions"
OUT_DIR = "/home/ecm5702/perm/interp/presentation"
RADIUS_KM = 500.0
EVENTS = {"Franklin": (10.0, 45.0, 283.0, 315.0), "Idalia": (12.0, 34.0, 260.0, 283.0)}
CURVES = [("x_interp", "input (EEFO O320 → O1280)", "#d9531e"),
          ("y_pred", "model (eecdb127 pw30)", "#1f77b4"),
          ("y", "truth (ENFO O1280)", "#e377c2")]

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "legend.fontsize": 8.5})


def collect(window):
    la0, la1, lo0, lo1 = window
    files = sorted(glob.glob(os.path.join(PRED_DIR, "predictions_*.nc")))
    blocks = {var: {"msl": [], "wind": []} for var, _, _ in CURVES}
    centers = []
    lat_b = lon_b = win_sub = None
    for fp in files:
        with xr.open_dataset(fp) as ds:
            ws = ds["weather_state"].values.tolist()
            i_msl, i_u, i_v = ws.index("msl"), ws.index("10u"), ws.index("10v")
            if lat_b is None:
                lat_h = np.asarray(ds["lat_hres"].values, dtype=np.float64)
                lon_h = np.asarray(ds["lon_hres"].values, dtype=np.float64)
                band = (lat_h >= la0 - 6.0) & (lat_h <= la1 + 6.0)
                ib = np.where(band)[0]
                lo, hi = int(ib.min()), int(ib.max()) + 1
                lat_b, lon_b = lat_h[lo:hi], lon_h[lo:hi]
                lonn = lon_b % 360.0
                win_sub = ((lat_b >= la0) & (lat_b <= la1) & (lonn >= lo0) & (lonn <= lo1))
            else:
                lo, hi = lo, hi
            arrays = {var: np.asarray(ds[var].isel(sample=0).values[:, lo:hi, :])
                      for var, _, _ in CURVES}
        truth_mean = (arrays["y"][:, :, i_msl] / 100.0).mean(axis=0)
        j = int(np.argmin(np.where(win_sub, truth_mean, np.inf)))
        clat, clon = float(lat_b[j]), float(lon_b[j] % 360.0)
        centers.append((clat, clon))
        box = box_mask_km(lat_b, lon_b, clat, clon, RADIUS_KM)
        for var, _, _ in CURVES:
            arr = arrays[var]
            msl = arr[:, box, i_msl] / 100.0
            wind = np.sqrt(arr[:, box, i_u] ** 2 + arr[:, box, i_v] ** 2)
            # One statistical unit per member/file: the TC-relevant box extreme.
            blocks[var]["msl"].append(np.min(msl, axis=1))
            blocks[var]["wind"].append(np.max(wind, axis=1))
    return {var: {field: np.concatenate(values) for field, values in fields.items()}
            for var, fields in blocks.items()}, centers, len(files)


def block_bootstrap_median(blocks, reps=2000):
    blocks = np.asarray(blocks)
    rng = np.random.default_rng(20260714)
    picks = rng.integers(0, len(blocks), size=(reps, len(blocks)))
    boot = np.median(blocks[picks].reshape(reps, -1), axis=1)
    return np.percentile(boot, [2.5, 50, 97.5])


def plot_event(event, window, out):
    data, centers, n_files = collect(window)
    fig, (ax_msl, ax_wind) = plt.subplots(1, 2, figsize=(13, 5))
    for field, ax, xlabel, invert in (("msl", ax_msl, "member/file box minimum (hPa)", True),
                                      ("wind", ax_wind, "member/file box maximum (m/s)", False)):
        for var, label, color in CURVES:
            blocks = []
            # Reconstruct the 25 blocks from the flat arrays for block bootstrap.
            n_members = len(data[var][field]) // n_files
            flat = data[var][field]
            blocks = flat.reshape(n_files, n_members)
            lo, med, hi = block_bootstrap_median(blocks)
            vals = np.sort(flat)
            y = np.arange(1, len(vals) + 1, dtype=float) / len(vals)
            ax.step(vals, y, where="post", color=color, lw=2.0,
                    label=f"{label}\nmedian {med:.1f} [{lo:.1f}, {hi:.1f}]")
            ax.axvline(med, color=color, lw=0.8, alpha=0.35)
        if invert:
            ax.invert_xaxis()
        ax.set_xlabel(xlabel); ax.set_ylabel("empirical CDF")
        ax.set_ylim(0, 1.02); ax.grid(True)
        ax.legend(loc="lower right" if invert else "lower right")
    clat = np.mean([c[0] for c in centers]); clon = np.mean([norm_lon(c[1]) for c in centers])
    n = n_files * 10
    fig.suptitle(f"{event} TC extreme distributions — storm-following {RADIUS_KM:.0f} km box\n"
                 f"n={n:,} member/file extremes per curve ({n_files} files × 10 members; O1280 support; "
                 f"mean center {clat:.1f}N {clon:.1f}E)", fontsize=12)
    fig.text(0.5, 0.01, "Block-bootstrap 95% intervals resample the 25 date/step files; spatial grid cells are not treated as independent.",
             ha="center", fontsize=8.5, color="#444", style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    png = os.path.join(out, f"eecdb127_tc_dist_{event.lower()}_stormbox.png")
    fig.savefig(png); return fig, data, png


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = os.path.join(OUT_DIR, "eecdb127_tc_distributions_stormbox.pdf")
    with PdfPages(pdf) as writer:
        for event, window in EVENTS.items():
            fig, data, png = plot_event(event, window, OUT_DIR)
            writer.savefig(fig); plt.close(fig)
            print(f"{event}: n={len(data['y']['msl']):,} member/file extremes; wrote {png}")
    print("wrote", pdf)


if __name__ == "__main__":
    main()
