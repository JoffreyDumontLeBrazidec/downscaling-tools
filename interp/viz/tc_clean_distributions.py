#!/usr/bin/env python3
"""Clean, apples-to-apples TC distribution PDFs for eecdb127 (o320->o1280),
on a STORM-FOLLOWING box (does NOT confound Franklin and Idalia).

Every curve is built from the SAME prediction bundle (same init dates, steps,
10 members, O1280 grid).  For each (init date, step) the storm center is found
as argmin(truth msl) inside a per-storm search window, and a fixed-radius
great-circle box is taken around it.  The SAME box is applied to input / model /
truth, so all three share identical support per field; Franklin and Idalia use
disjoint search windows so their boxes never overlap.

Curves (all on the O1280 grid):
  input  = x_interp  (EEFO O320 input, interpolated to O1280; == EEFO_O320)
  model  = y_pred    (eecdb127 pw30)
  truth  = y         (== ENFO O1280, verified member-by-member)
"""
from __future__ import annotations
import glob, os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from interp.core.geometry import detect_min_center, box_mask_km, norm_lon

PRED_DIR = ("/home/ecm5702/scratch/eval/manual_eecdb127_new_o320_o1280_"
            "20260503_manual_eval/data/predictions")
OUT_DIR = "/home/ecm5702/perm/interp/presentation"
RADIUS_KM = 500.0

# Per-storm SEARCH WINDOW (lat0,lat1,lon0,lon1; lon in 0..360). Disjoint in lon
# at 283E (=-77W): Franklin is open-Atlantic (~290-300E), Idalia is Gulf/SE-US
# (~273-282E). Storm center = argmin(truth msl) inside the window, per field.
EVENTS = {
    "Franklin": (10.0, 45.0, 283.0, 315.0),
    "Idalia":   (12.0, 34.0, 260.0, 283.0),
}
CURVES = [  # (bundle var, label, color)
    ("x_interp", "input  (EEFO O320 → O1280)", "#d9531e"),
    ("y_pred",   "model  (eecdb127 pw30)",     "#1f77b4"),
    ("y",        "truth  (ENFO O1280)",        "#e377c2"),
]
MSL_BINS  = np.arange(915.0, 1026.0, 1.0)
WIND_BINS = np.arange(0.0, 70.0, 0.5)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "legend.fontsize": 9,
})


def collect(window):
    """Pool msl (hPa) and wind10m (m/s) over a storm-following 500 km box.

    Center is argmin(truth-mean msl) inside `window`, recomputed per file; the
    same box mask is applied to all three streams.
    """
    la0, la1, lo0, lo1 = window
    files = sorted(glob.glob(os.path.join(PRED_DIR, "predictions_*.nc")))
    acc = {v: {"msl": [], "wind": []} for v, _, _ in CURVES}
    centers = []
    lo = hi = lat_b = lon_b = win_sub = None
    for fp in files:
        ds = xr.open_dataset(fp)
        ws = ds["weather_state"].values.tolist()
        i_msl, i_u, i_v = ws.index("msl"), ws.index("10u"), ws.index("10v")
        if lo is None:
            lat_h = np.asarray(ds["lat_hres"].values, dtype=np.float64)
            lon_h = np.asarray(ds["lon_hres"].values, dtype=np.float64)  # 0..360
            # contiguous latitude band covering the window (+6 deg so a 500 km
            # box near the window edge stays fully inside the band).
            band = (lat_h >= la0 - 6.0) & (lat_h <= la1 + 6.0)
            ib = np.where(band)[0]; lo, hi = int(ib.min()), int(ib.max()) + 1
            lat_b, lon_b = lat_h[lo:hi], lon_h[lo:hi]
            lonn = lon_b % 360.0
            win_sub = ((lat_b >= la0) & (lat_b <= la1) &
                       (lonn >= lo0) & (lonn <= lo1))  # centering window in band
        # read all three streams' band once (contiguous, all ws), slice in RAM
        bands = {v: np.asarray(ds[v].isel(sample=0).values[:, lo:hi, :])
                 for v, _, _ in CURVES}
        ds.close()
        # storm center = argmin(truth-mean msl) within the window
        y_mean = (bands["y"][:, :, i_msl] / 100.0).mean(axis=0)
        j = int(np.argmin(np.where(win_sub, y_mean, np.inf)))
        clat, clon = float(lat_b[j]), float(lon_b[j] % 360.0)
        centers.append((clat, clon))
        boxsub = box_mask_km(lat_b, lon_b, clat, clon, RADIUS_KM)  # (band,) bool
        for var, _, _ in CURVES:
            arr = bands[var]
            msl = arr[:, :, i_msl][:, boxsub] / 100.0
            u = arr[:, :, i_u][:, boxsub]
            v = arr[:, :, i_v][:, boxsub]
            acc[var]["msl"].append(msl.reshape(-1))
            acc[var]["wind"].append(np.sqrt(u ** 2 + v ** 2).reshape(-1))
        del bands
    out = {}
    for var, _, _ in CURVES:
        out[var] = {"msl": np.concatenate(acc[var]["msl"]),
                    "wind": np.concatenate(acc[var]["wind"])}
    out["_centers"] = centers
    return out


def density(vals, bins):
    h, edges = np.histogram(vals, bins=bins, density=True)
    return 0.5 * (edges[:-1] + edges[1:]), h


def print_stats(event, data):
    cs = data["_centers"]
    clat = np.mean([c[0] for c in cs]); clon = np.mean([norm_lon(c[1]) for c in cs])
    print(f"\n##STATS {event}  storm-following R={RADIUS_KM:.0f}km  "
          f"(10 members / 5 dates / 5 steps; mean center {clat:.1f}N {clon:.1f}E)")
    print(f"{'stream':28s} {'n':>10s} {'min':>7s} {'max':>7s} {'mean':>7s} {'std':>6s} "
          f"{'p5':>7s} {'p50':>7s} {'p95':>7s}")
    for var, label, _ in CURVES:
        m = data[var]["msl"]
        p5, p50, p95 = np.percentile(m, [5, 50, 95])
        print(f"{label:28s} {m.size:>10,d} {m.min():7.1f} {m.max():7.1f} {m.mean():7.1f} "
              f"{m.std():6.1f} {p5:7.1f} {p50:7.1f} {p95:7.1f}")


def plot_event(ax_msl, ax_wind, data, event):
    n = {v: data[v]["msl"].size for v, _, _ in CURVES}
    assert len(set(n.values())) == 1, f"support mismatch! {n}"
    for var, label, color in CURVES:
        mids, h = density(data[var]["msl"], MSL_BINS)
        ax_msl.semilogy(mids, h, color=color, lw=2.0, label=label)
        mids, h = density(data[var]["wind"], WIND_BINS)
        ax_wind.semilogy(mids, h, color=color, lw=2.0, label=label)
    ax_msl.set_title(f"{event} — mean sea-level pressure")
    ax_msl.set_xlabel("hPa"); ax_msl.set_ylabel("density"); ax_msl.invert_xaxis()
    ax_wind.set_title(f"{event} — 10 m wind speed")
    ax_wind.set_xlabel("m/s"); ax_wind.set_ylabel("density")
    ax_msl.legend(loc="upper right"); ax_wind.legend(loc="upper right")
    return next(iter(n.values()))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUT_DIR, "eecdb127_tc_distributions_stormbox.pdf")
    with PdfPages(pdf_path) as pdf:
        for event, window in EVENTS.items():
            data = collect(window)
            print_stats(event, data)
            fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
            npts = plot_event(a, b, data, event)
            fig.suptitle(f"{event} TC distributions — storm-following "
                         f"{RADIUS_KM:.0f} km box (10 members · 5 dates · 5 steps · "
                         f"O1280 · n={npts:,}/curve)", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            png = os.path.join(OUT_DIR, f"eecdb127_tc_dist_{event.lower()}_stormbox.png")
            fig.savefig(png); pdf.savefig(fig); plt.close(fig)
            print(f"{event}: n={npts:,} per curve -> {png}")
    print("wrote", pdf_path)
    print("DONE_TC_STORMBOX")


if __name__ == "__main__":
    main()
