#!/usr/bin/env python3
"""Figures and tables from accumulate.py outputs. House colours: truth black, input blue dashed, model red."""
import sys, os, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

BT = ["17_28", "28_45", "45_90", "90_200"]
C = dict(truth="k", model="tab:red", input="tab:blue"); LS = dict(truth="-", model="-", input="--")


def grid(out):
    g = np.load(os.path.join(out, "tiles_grid0125.npz"))
    ny, nx = g["cnt"].shape
    lat = 10 + (np.arange(ny) + 0.5) * 0.125; lon = -80 + (np.arange(nx) + 0.5) * 0.125
    return g, lat, lon


def fill(a, cnt):
    a = np.where(cnt > 0, a, np.nan)
    from scipy import ndimage
    bad = ~np.isfinite(a)
    if bad.any():
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        a = a[tuple(idx)]
    return a


def ratio_maps(out, png, extent=None, title=""):
    g, lat, lon = grid(out)
    cnt = g["cnt"]; lsm = fill(g["lsm"], cnt)
    fig, axs = plt.subplots(3, 2, figsize=(17, 13), constrained_layout=True)
    panels = []
    for b in range(4):
        r = fill(g["band2_model_%d" % b], cnt) / fill(g["band2_truth_%d" % b], cnt)
        panels.append((np.sqrt(r), "amplitude ratio model / truth, wind speed, %s km" % BT[b].replace("_", "-")))
    n = g["n"]
    mt = fill(g["sum_ws_truth"], cnt) / n[0].max(); mm = fill(g["sum_ws_model"], cnt) / n[1].max()
    panels.append((mm - mt, "mean wind speed, model - truth [m/s]"))
    sk = (fill(g["speckle_model"], cnt) + 1e-3 * n[1].max()) / (fill(g["speckle_truth"], cnt) + 1e-3 * n[0].max())
    panels.append((sk, "speckle pixel frequency, model / truth"))
    for ax, (a, t) in zip(axs.ravel(), panels):
        if "mean wind" in t:
            im = ax.pcolormesh(lon, lat, a, cmap="RdBu_r", norm=TwoSlopeNorm(0, -1.5, 1.5), rasterized=True)
        else:
            im = ax.pcolormesh(lon, lat, a, cmap="RdBu_r", norm=LogNorm(1 / 2.0, 2.0), rasterized=True)
        ax.contour(lon, lat, lsm, [0.5], colors="k", linewidths=0.3)
        ax.set_title(t, fontsize=11)
        if extent: ax.set_xlim(extent[2], extent[3]); ax.set_ylim(extent[0], extent[1])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title)
    fig.savefig(png); plt.close(fig)


def class_table(out):
    """band amplitude ratios by surface class, SE = delete-one-date jackknife."""
    t = pd.read_csv(os.path.join(out, "tile_dates.csv"))
    g, lat, lon = grid(out)
    cnt = g["cnt"]; lsm = fill(g["lsm"], cnt); orog = fill(g["orog"], cnt)
    rows = []
    for (la, lo), _ in t.groupby(["lat0", "lon0"]):
        m = (np.abs(lat[:, None] - la) <= 2.5) & (np.abs(lon[None, :] - lo) <= 2.5)
        rows.append(dict(lat0=la, lon0=lo, land=float(lsm[m].mean()), orog_sd=float(orog[m].std()), orog_mean=float(orog[m].mean())))
    k = pd.DataFrame(rows)
    def cls(r):
        if r.lat0 in (12.5, 67.5) or r.lon0 in (-77.5, 37.5): return "edge"
        if r.land < 0.05: return "open sea"
        if r.land < 0.6: return "coast / mixed"
        return "mountain land" if r.orog_sd > 350 else "flat land"
    k["cls"] = k.apply(cls, axis=1)
    t = t.merge(k, on=["lat0", "lon0"])
    return t, k


def jk_ratio(df, num, den, unit="date"):
    """ratio of sums with delete-one-unit jackknife SE; df has columns unit, num, den"""
    s = df.groupby(unit)[[num, den]].sum()
    tot = s.sum(); r = tot[num] / tot[den]
    lo = (tot[num] - s[num]) / (tot[den] - s[den])
    n = len(s)
    return r, float(np.sqrt((n - 1) / n * ((lo - r) ** 2).sum())) if n > 1 else np.nan


def band_table(out):
    t, k = class_table(out)
    res = []
    for cl in ["open sea", "coast / mixed", "flat land", "mountain land", "all but edge"]:
        d = t[t.cls != "edge"] if cl == "all but edge" else t[t.cls == cl]
        if d.empty: continue
        for q, lab, root in (("ws_var_", "wind speed", True), ("ke_pow_", "u,v", True), ("vort_pow_", "vorticity", True), ("div_pow_", "divergence", True), ("msl_pow_", "msl", True)):
            for b in BT:
                w = d.pivot_table(index=["date", "lead", "tile"], columns="src", values=q + b).reset_index()
                for s in ("model", "input"):
                    r, se = jk_ratio(w, s, "truth")
                    res.append(dict(cls=cl, ntiles=d.tile.nunique(), var=lab, band=b, src=s, amp_ratio=np.sqrt(r), se=se / (2 * np.sqrt(r))))
    return pd.DataFrame(res), k


def spectra_fig(out, png, title=""):
    z = np.load(os.path.join(out, "spectra.npz"))
    S = z["spectra"]; e = z["edges"]; wl = np.sqrt(e[:-1] * e[1:]); names = list(z["names"]); src = list(z["src"])
    t, k = class_table(out)
    k = k.reset_index(drop=True)
    fig, axs = plt.subplots(2, 4, figsize=(19, 8.5), constrained_layout=True)
    for j, cl in enumerate(["open sea", "coast / mixed", "flat land", "mountain land"]):
        ti = np.where(k.cls.values == cl)[0]
        ax = axs[0, j]
        P = {s: S[src.index(s)][ti].sum(0) for s in src}
        for s in src:
            ax.loglog(wl, np.sqrt(P[s][0] / max(len(ti), 1) / z["n"][src.index(s)].max()), color=C[s], ls=LS[s], label=s)
        ax.axvline(31.3, color="grey", lw=0.8, ls=":"); ax.invert_xaxis(); ax.set_title("%s (%d tiles): wind-speed amplitude per bin [m/s]" % (cl, len(ti)), fontsize=10)
        ax.set_xlabel("wavelength km"); ax.legend(fontsize=8)
        ax = axs[1, j]
        for nm, col in (("ws", "tab:red"), ("vort", "tab:purple"), ("div", "tab:green"), ("msl", "tab:orange")):
            i = names.index(nm)
            ax.semilogx(wl, np.sqrt(P["model"][i] / P["truth"][i]), color=col, label=nm + " model")
        ax.semilogx(wl, np.sqrt(P["input"][0] / P["truth"][0]), color="tab:blue", ls="--", label="ws input")
        ax.axhline(1, color="k", lw=0.6); ax.axvline(31.3, color="grey", lw=0.8, ls=":"); ax.set_ylim(0.3, 3); ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_title("amplitude ratio to truth", fontsize=10); ax.set_xlabel("wavelength km"); ax.legend(fontsize=8)
    fig.suptitle(title + "  (dotted line: 31.3 km, the truth model's spectral truncation)")
    fig.savefig(png); plt.close(fig)


def crop_grid(z, src, m, var, lat0, lat1, lon0, lon1, d=0.05):
    from scipy.spatial import cKDTree
    la = np.arange(lat0, lat1, d); lo = np.arange(lon0, lon1, d / np.cos(np.radians(0.5 * (lat0 + lat1))))
    LO, LA = np.meshgrid(lo, la)
    c = np.cos(np.radians(0.5 * (lat0 + lat1)))
    tr = cKDTree(np.c_[z["lon"] * c, z["lat"]])
    dd, ii = tr.query(np.c_[LO.ravel() * c, LA.ravel()])
    def get(s, mm):
        if s == "ctl":
            uv = z["ctl_uv"].astype("f4"); p = z["ctl_msl"]
        else:
            uv = z[s + "_uv"][mm].astype("f4"); p = z[s + "_msl"][mm]
        if p.dtype == np.int16:      # reduce.py packing: (value + 10000) * 10 Pa
            p = (p.astype("f4") + 10000.0) * 10.0
        return uv[:, 0], uv[:, 1], p
    u, v, p = get(src, m)
    f = dict(u=u, v=v, msl=p, ws=np.hypot(u, v))[var]
    a = f[ii].reshape(LA.shape); a[dd.reshape(LA.shape) > 0.12] = np.nan
    return lo, la, a, z["lsm"].astype("f4")[ii].reshape(LA.shape)


def sheet(crop, png, ext, var="ws", fine=False, members=(0, 1), vmax=None, title="", sig_km=40.0, msl_contours=True):
    """rows of panels: input m1, truth members, model members (same colour scale)."""
    from scipy import ndimage
    z = np.load(crop)
    cols = [("input", members[0])] + [("truth", m) for m in members] + [("model", m) for m in members]
    fig, axs = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 4.9), constrained_layout=True)
    arrs = []
    for s, m in cols:
        lo, la, a, lsm = crop_grid(z, s, m, var, *ext)
        if fine:
            px = 0.05 * 111.2
            a = a - ndimage.gaussian_filter(np.nan_to_num(a, nan=np.nanmean(a)), sig_km / px)
        _, _, p, _ = crop_grid(z, s, m, "msl", *ext)
        arrs.append((lo, la, a, lsm, p))
    if vmax is None:
        vmax = np.nanpercentile(np.abs(np.concatenate([x[2].ravel() for x in arrs])), 99.8)
    for ax, (s, m), (lo, la, a, lsm, p) in zip(axs, cols, arrs):
        if fine:
            im = ax.pcolormesh(lo, la, a, cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True)
        else:
            im = ax.pcolormesh(lo, la, a, cmap="turbo", vmin=0, vmax=vmax, rasterized=True)
        ax.contour(lo, la, lsm, [0.5], colors="k", linewidths=0.6)
        if msl_contours:
            ax.contour(lo, la, p / 100.0, np.arange(900, 1060, 4), colors="w" if not fine else "grey", linewidths=0.4)
        ax.set_title("%s member %d" % (s, m + 1), fontsize=11); ax.set_aspect(1.0 / np.cos(np.radians(0.5 * (ext[0] + ext[1]))))
    fig.colorbar(im, ax=axs, shrink=0.8, label=("fine part (minus %d km smoothing) " % sig_km if fine else "") + "10 m wind speed m/s")
    fig.suptitle(title)
    fig.savefig(png); plt.close(fig)
