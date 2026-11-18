#!/usr/bin/env python3
"""Ensemble spread by scale: variance of member anomalies (member minus the 10-member mean) per 5-degree tile and wavelength band,
for truth members, model draws and input members. Same field files and tiles as accumulate.py.
usage: spread.py --fields DIR --orog FILE --out CSV"""
import argparse, glob, os, re, csv
import numpy as np
from scipy.spatial import cKDTree
import accumulate as A

SB = [(28.0, 45.0), (45.0, 90.0), (90.0, 200.0), (200.0, 400.0), (400.0, 1e9)]
ap = argparse.ArgumentParser(); ap.add_argument("--fields", required=True); ap.add_argument("--orog", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
O = np.load(a.orog); lat = O["lat"].astype(float); lon = np.where(O["lon"] > 180, O["lon"] - 360, O["lon"]).astype(float)
tree = cKDTree(A.xyz(lat, lon))
tiles = [A.Tile(la, lo, tree) for la in np.arange(17.5, 65, 5.0) for lo in np.arange(-72.5, 35, 5.0)]
masks = []
for T in tiles:
    k = np.hypot(T.KX, T.KY)
    with np.errstate(divide="ignore"):
        wl = np.where(k > 0, 1.0 / k, 1e12)
    masks.append([T.w * ((wl >= lo_) & (wl < hi_)) for lo_, hi_ in SB])
rows = []
for f in sorted(glob.glob(os.path.join(a.fields, os.environ.get("FB_ROW", "GLB450e2_m") + "_*_L*.npz"))):
    mm = re.search(r"_(\d{8})_L(\d{3})\.npz$", f); Z = np.load(f)
    D = dict(truth=Z["truth"], model=Z["pred"], input=Z["inp"])
    for s, X in D.items():
        ws = np.hypot(X[:, :, 0], X[:, :, 1]); msl = X[:, :, 2]
        for name, F in (("ws", ws), ("msl", msl)):
            An = F - F.mean(0, keepdims=True)
            for ti, T in enumerate(tiles):
                acc = np.zeros(len(SB)); tot = 0.0
                for m in range(An.shape[0]):
                    g = An[m][T.idx]
                    tot += float((g[T.core] ** 2).mean())
                    P = np.abs(np.fft.rfft2((g - g.mean()) * T.win)) ** 2
                    for b in range(len(SB)):
                        acc[b] += float((P * masks[ti][b]).sum())
                M = An.shape[0]; c = M / (M - 1.0)      # unbiased member variance
                r = dict(date=mm.group(1), lead=int(mm.group(2)), tile=ti, lat0=T.lat0, lon0=T.lon0, src=s, var=name, total=c * tot / M)
                for b in range(len(SB)):
                    r["b%d_%d" % (SB[b][0], min(SB[b][1], 9999))] = c * acc[b] / M
                rows.append(r)
    print("FILE_DONE", os.path.basename(f), flush=True)
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# compact summary under 1 MB: sums over tiles per date, source, variable
import pandas as pd
t = pd.read_csv(a.out); num = [c for c in t.columns if c == "total" or c.startswith("b")]
t.groupby(["date", "lead", "src", "var"])[num].sum().reset_index().to_csv(a.out.replace(".csv", "_dates.csv"), index=False, float_format="%.6g")
t.groupby(["tile", "lat0", "lon0", "src", "var"])[num].mean().reset_index().to_csv(a.out.replace(".csv", "_tiles.csv"), index=False, float_format="%.6g")
print("SPREAD_DONE", len(rows))
