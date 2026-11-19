#!/usr/bin/env python
"""Map sheets for the realism t2m read: columns input | truth | model | model minus truth, rows value and
fine (judge 0.6-degree high-pass) per variable, member 1 (input x_interp member 1, truth pf member 1,
model draw 1). Truth pf 1 and input 1 are not paired members (see README). Orography contours every
500 m (O1280 zhr). Colour rule 25: 2t vik white at 273.15 K, msl/sp lipari; fine and difference vik.
Usage: maps_t2m.py OUTDIR name:region:date:lead:var1,var2 [...]
"""
import sys, os
import numpy as np
import netCDF4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm, Normalize
import cartopy.crs as ccrs, cartopy.feature as cfeature
import cmcrameri.cm as cmc
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import realism_t2m as R
LABEL = os.environ.get("FB_LABEL", "GL512 450k EMA-2000")

EXT = {"alps": (4, 16, 43, 49), "norway": (4, 20, 58, 69), "iceland": (-25, -13, 63, 67),
       "s_greenland": (-55, -40, 59.5, 69), "corsica_sardinia": (7, 11, 38, 44), "pyrenees": (-2, 3.5, 41.5, 43.5)}
FINE = {"2t": 3.0, "2d": 3.0, "skt": 4.0, "msl": 80.0, "sp": 4000.0}       # Pa for msl/sp (plotted in hPa below)
UNIT = {"2t": ("K", 1.0, 0.0), "2d": ("K", 1.0, 0.0), "skt": ("K", 1.0, 0.0), "msl": ("hPa", 0.01, 0.0), "sp": ("hPa", 0.01, 0.0)}


CTRL_VARS = ("2d", "skt", "tcw")   # no pf member truth: truth column = 9 km control y, read from the 10-output rows


def load_case(st, M, date, lead, var):
    row = os.environ.get("FB_ROW10", "GLB450e2_m10") if var in CTRL_VARS else R.ROW
    fn = f"{R.X}/pred/{row}_L{lead:03d}/predictions_{date}_step{lead:03d}.nc"
    with netCDF4.Dataset(fn) as d:
        d.set_auto_mask(False)
        states = [str(x) for x in d["weather_state"][:]]
        vi = states.index(var)
        mod = np.asarray(d["y_pred"][0, 0, :, vi], np.float64)
        inp = np.asarray(d["x_interp_members"][0, 0, :, vi], np.float64)
        if var in CTRL_VARS:
            return {"input": inp, "truth": np.asarray(d["y"][0, :, vi], np.float64), "model": mod}
    zi, zo, r = R.store_row(lead, date)
    tru = zo["data"][r, list(zo.attrs["variables"]).index(var), 0, :].astype(np.float64)
    return {"input": inp, "truth": tru, "model": mod}


def sheet(st, M, out, name, region, date, lead, vars_):
    ext = EXT[region]
    m = (st.lat >= ext[2] - .3) & (st.lat <= ext[3] + .3) & (st.lon >= ext[0] - .3) & (st.lon <= ext[1] + .3)
    tri = mtri.Triangulation(st.lon[m], st.lat[m])
    nrow = 2 * len(vars_)
    aspect = (ext[1] - ext[0]) * np.cos(np.radians(0.5 * (ext[2] + ext[3]))) / (ext[3] - ext[2])
    pw = 3.3; ph = pw / aspect + 0.45
    fig, axs = plt.subplots(nrow, 4, figsize=(4 * pw + 1.6, nrow * ph), subplot_kw={"projection": ccrs.PlateCarree()},
                            squeeze=False, layout="constrained")
    vt = f"{(lead % 24):02d} UTC"
    for j, var in enumerate(vars_):
        f = load_case(st, M, date, lead, var)
        u, sc, _ = UNIT[var]
        hp = {k: v - st.lowpass(v) for k, v in f.items()}
        for row, kind in ((2 * j, "value"), (2 * j + 1, "fine")):
            src = f if kind == "value" else hp
            vals = {k: src[k][m] * sc for k in ("input", "truth", "model")}
            vals["diff"] = vals["model"] - vals["truth"]
            if kind == "value":
                allv = np.concatenate([vals[k] for k in ("input", "truth", "model")])
                lo, hi = np.percentile(allv, [0.5, 99.5])
                if var in ("2t", "2d", "skt"):
                    lo, hi = min(lo, 272.65), max(hi, 273.65); norm = TwoSlopeNorm(273.15, lo, hi); cmap = cmc.vik
                else:
                    norm = Normalize(lo, hi); cmap = cmc.lipari
            else:
                a = FINE[var] * sc; norm = Normalize(-a, a); cmap = cmc.vik
            dmax = max(np.percentile(np.abs(vals["diff"]), 99), 1e-6)
            for c, k in enumerate(("input", "truth", "model", "diff")):
                ax = axs[row, c]; ax.set_extent(ext, crs=ccrs.PlateCarree())
                nm, cm = (Normalize(-dmax, dmax), cmc.vik) if k == "diff" else (norm, cmap)
                pc = ax.tripcolor(tri, vals[k], cmap=cm, norm=nm, shading="gouraud", transform=ccrs.PlateCarree(),
                                  rasterized=True)
                ax.tricontour(tri, st.zhr[m], levels=np.arange(500, 5000, 500), colors="0.35", linewidths=0.3,
                              transform=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, edgecolor="k")
                lab = {"input": "input x_interp m1", "truth": "truth ctrl" if var in CTRL_VARS else "truth pf m1", "model": os.environ.get("FB_ROW", "GLB450e2") + " draw 1",
                       "diff": "model minus truth"}[k]
                v = vals[k]
                ax.set_title(f"{var} {kind}: {lab}\nmin {v.min():.1f} max {v.max():.1f} {u}", fontsize=7.5)
                if c == 2:
                    cb = fig.colorbar(pc, ax=list(axs[row, :3]), shrink=0.85, pad=0.01); cb.ax.tick_params(labelsize=6)
                if c == 3:
                    cb = fig.colorbar(pc, ax=ax, shrink=0.85, pad=0.01); cb.ax.tick_params(labelsize=6)
    fig.suptitle(f"{name}: {region} {date} +{lead} h (valid {vt}); {LABEL} ({R.ROW}); "
                 f"fine = 0.6-degree high-pass; grey = O1280 orography every 500 m; truth/input member 1 not paired",
                 fontsize=8.5)
    p = f"{out}/{name}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    rgb = Image.open(p).convert("RGB")
    for ncol in (256, 128, 64):
        rgb.quantize(colors=ncol, method=Image.Quantize.MEDIANCUT).save(p, optimize=True)
        if os.path.getsize(p) < 900 * 1024:
            break
    print("MAP", p, os.path.getsize(p) // 1024, "KB", flush=True)


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    st, M = R.Stat(), R.load_matrix()
    for spec in sys.argv[2:]:
        name, region, date, lead, vs = spec.split(":")
        sheet(st, M, out, name, region, date, int(lead), vs.split(","))
