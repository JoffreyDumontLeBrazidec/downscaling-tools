#!/usr/bin/env python3
"""TD-227 temporal-consistency read: per-band lead-to-lead difference and lag-1 correlation.

For one member's trajectory over leads 12-72 h (the ``.npz`` files of ``predict_sequence.py``,
or any arrays shaped (L, grid, vars) with lat/lon), the fields of consecutive leads are
band-passed on a regular 0.09 degree regrid of the box into three bands,
31-56 km, 56-125 km and above 125 km (Gaussian smoothers, half-widths set by the band
edges), and two statistics are computed per band and variable:

- ``tdiff``: the area-weighted mean absolute difference between lead L and lead L+12 h of the
  band-passed field, averaged over the five consecutive pairs; reported for the prediction and
  the truth and as the ratio prediction / truth. Above 1 the trajectory flickers more than the
  truth; below 1 it is smoother than the truth (both are defects: the target is to MATCH the
  truth's own lead-to-lead change, Andrae et al. 2024).
- ``lag1``: the area-weighted correlation between lead L and lead L+12 h of the band-passed
  field, averaged over the pairs; prediction and truth.

Optionally a second prediction set (``--control``) is read with the same instruments, so a
stride-1 (averaged) and a stride-K (isolated windows) run of the same model, or the single-time
reference lane's files, sit in one table.

Output: a Markdown table and a JSON file. Variables read by default: 10u, 10v, 2t, msl, tp.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

R_EARTH_KM = 6371.0
BANDS = {"31-56km": (31.0, 56.0), "56-125km": (56.0, 125.0), ">125km": (125.0, None)}
DEFAULT_VARS = ["10u", "10v", "2t", "msl", "tp"]


def _unit_vectors(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


class BoxRegrid:
    """Nearest-neighbour regrid of the box points onto a regular lat-lon grid, with Gaussian band-pass."""

    def __init__(self, lat, lon, deg=0.09, margin_deg=0.7):
        self.glat = np.arange(lat.min() + margin_deg, lat.max() - margin_deg + 1e-9, deg)
        self.glon = np.arange(lon.min() + margin_deg, lon.max() - margin_deg + 1e-9, deg)
        la, lo = np.meshgrid(self.glat, self.glon, indexing="ij")
        tree = cKDTree(_unit_vectors(lat, lon))
        _, self.idx = tree.query(_unit_vectors(la.ravel(), lo.ravel()), k=1, workers=-1)
        self.idx = self.idx.reshape(la.shape)
        self.dlat_km = deg * np.pi / 180.0 * R_EARTH_KM
        self.dlon_km = self.dlat_km * np.cos(np.deg2rad(self.glat))  # per row
        self.weights = np.cos(np.deg2rad(la))  # area weights on the regular grid

    def to_grid(self, field):
        return field[self.idx]

    def _smooth(self, g, scale_km):
        """Gaussian smoother with half-width ``scale_km`` (sigma = scale / 2) on the regular grid."""
        sigma_lat = scale_km / 2.0 / self.dlat_km
        sigma_lon = scale_km / 2.0 / float(np.mean(self.dlon_km))
        return ndimage.gaussian_filter(g, sigma=(sigma_lat, sigma_lon), mode="nearest")

    def bandpass(self, field):
        g = self.to_grid(field)
        out = {}
        for name, (lo_km, hi_km) in BANDS.items():
            s_lo = self._smooth(g, lo_km)
            s_hi = self._smooth(g, hi_km) if hi_km is not None else 0.0
            out[name] = s_lo - s_hi  # scales between lo and hi survive
        return out


def _wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def _wcorr(a, b, w):
    am, bm = _wmean(a, w), _wmean(b, w)
    cov = _wmean((a - am) * (b - bm), w)
    va, vb = _wmean((a - am) ** 2, w), _wmean((b - bm) ** 2, w)
    return float(cov / np.sqrt(va * vb + 1e-30))


def trajectory_stats(arr, regrid, variables, wanted):
    """arr: (L, grid, vars). Returns {var: {band: {tdiff, lag1}}} averaged over consecutive pairs."""
    stats = {}
    for var in wanted:
        if var not in variables:
            continue
        j = variables.index(var)
        bands = [regrid.bandpass(arr[t, :, j]) for t in range(arr.shape[0])]
        stats[var] = {}
        for band in BANDS:
            td, lg = [], []
            for t in range(arr.shape[0] - 1):
                a, b = bands[t][band], bands[t + 1][band]
                td.append(_wmean(np.abs(b - a), regrid.weights))
                lg.append(_wcorr(a, b, regrid.weights))
            stats[var][band] = {"tdiff": float(np.mean(td)), "lag1": float(np.mean(lg))}
    return stats


def load_set(paths):
    """Average the statistics over files (members, blocks). Each file: pred, truth, variables, lat, lon."""
    preds, truths, regrid, variables = [], [], None, None
    for p in paths:
        z = np.load(p, allow_pickle=True)
        if regrid is None:
            regrid = BoxRegrid(z["lat"], z["lon"])
            variables = [str(v) for v in z["variables"]]
        preds.append(z["pred"])
        truths.append(z["truth"])
    return preds, truths, regrid, variables


def average_stats(list_of_stats):
    out = {}
    for var in list_of_stats[0]:
        out[var] = {}
        for band in BANDS:
            out[var][band] = {
                key: float(np.mean([s[var][band][key] for s in list_of_stats])) for key in ("tdiff", "lag1")
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="npz files of predict_sequence.py (one trajectory each)")
    parser.add_argument("--control", nargs="*", default=None, help="npz files of a second set (same blocks)")
    parser.add_argument("--vars", default=",".join(DEFAULT_VARS))
    parser.add_argument("--out", default=None, help="prefix for <out>.md and <out>.json")
    args = parser.parse_args()
    wanted = args.vars.split(",")

    preds, truths, regrid, variables = load_set(args.files)
    pred_stats = average_stats([trajectory_stats(p, regrid, variables, wanted) for p in preds])
    truth_stats = average_stats([trajectory_stats(t, regrid, variables, wanted) for t in truths])
    ctrl_stats = None
    if args.control:
        cpreds, _, _, _ = load_set(args.control)
        ctrl_stats = average_stats([trajectory_stats(p, regrid, variables, wanted) for p in cpreds])

    lines = ["| variable | band | tdiff truth | tdiff pred | ratio pred/truth | lag1 truth | lag1 pred |"
             + (" tdiff control | ratio control/truth | lag1 control |" if ctrl_stats else ""),
             "|---|---|---|---|---|---|---|" + ("---|---|---|" if ctrl_stats else "")]
    for var in pred_stats:
        for band in BANDS:
            t, p = truth_stats[var][band], pred_stats[var][band]
            row = (f"| {var} | {band} | {t['tdiff']:.4g} | {p['tdiff']:.4g} | {p['tdiff'] / (t['tdiff'] + 1e-30):.3f} "
                   f"| {t['lag1']:.3f} | {p['lag1']:.3f} |")
            if ctrl_stats:
                c = ctrl_stats[var][band]
                row += f" {c['tdiff']:.4g} | {c['tdiff'] / (t['tdiff'] + 1e-30):.3f} | {c['lag1']:.3f} |"
            lines.append(row)
    table = "\n".join(lines)
    print(f"{len(preds)} trajectories, {preds[0].shape[0]} leads, regrid {len(regrid.glat)} x {len(regrid.glon)} at 0.09 deg")
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out + ".md").write_text(table + "\n")
        Path(args.out + ".json").write_text(json.dumps({"pred": pred_stats, "truth": truth_stats, "control": ctrl_stats}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
