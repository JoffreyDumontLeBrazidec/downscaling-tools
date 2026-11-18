#!/usr/bin/env python3
"""Reduce accumulate.py outputs to files under 1 MB (docs pre-commit limit) and render the full-resolution maps.
usage: reduce.py <out_dir>   -> writes <out_dir>/small/ (commit this folder)."""
import sys, os, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plots
import matplotlib.pyplot as plt

out = sys.argv[1]; sm = os.path.join(out, "small"); os.makedirs(os.path.join(sm, "crops"), exist_ok=True)
LIM = 990_000

def save(path, **kw):
    np.savez_compressed(path, **kw)
    sz = os.path.getsize(path)
    print("WROTE", path, sz, "OVER_LIMIT" if sz > LIM else "", flush=True)
    return sz

# 1 tables
T, k = plots.band_table(out)
T.to_csv(os.path.join(sm, "band_table.csv"), index=False); k.to_csv(os.path.join(sm, "tile_class.csv"), index=False)
t = pd.read_csv(os.path.join(out, "tile_dates.csv"))
num = [c for c in t.columns if c not in ("date", "lead", "tile", "lat0", "lon0", "src", "members")]
t.groupby(["tile", "lat0", "lon0", "src"])[num].mean().reset_index().to_csv(os.path.join(sm, "tile_summary.csv"), index=False, float_format="%.6g")
# per date and class sums (for jackknife in the cloud)
tt, _ = plots.class_table(out)
tt.groupby(["date", "lead", "cls", "src"])[num].sum().reset_index().to_csv(os.path.join(sm, "class_dates.csv"), index=False, float_format="%.6g")

# 2 grid at 0.25 degree, float16, bundles
g = np.load(os.path.join(out, "tiles_grid0125.npz"))
cnt = g["cnt"].astype(np.float64)
def co(a):
    a = np.where(cnt > 0, a.astype(np.float64), 0.0) * (cnt > 0)
    ny, nx = a.shape
    s = (a * 1.0).reshape(ny // 2, 2, nx // 2, 2); c = (cnt > 0).reshape(ny // 2, 2, nx // 2, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s.sum((1, 3)) / c.sum((1, 3))).astype(np.float32)
meta = dict(dg=0.25, lat0=10.0, lon0=-80.0, n=g["n"], bands=g["bands"], src=g["src"], files=g["files"])
save(os.path.join(sm, "grid025_static.npz"), lsm=co(g["lsm"]).astype(np.float16), orog=co(g["orog"]), **meta)
for s in ("truth", "model", "input"):
    save(os.path.join(sm, "grid025_%s_ws.npz" % s), sum_ws=co(g["sum_ws_" + s]), sum_ws2=co(g["sum_ws2_" + s]))
    save(os.path.join(sm, "grid025_%s_speckle.npz" % s), speckle=co(g["speckle_" + s]))
    for b in range(4):
        save(os.path.join(sm, "grid025_%s_band%d.npz" % (s, b)), band2=co(g["band2_%s_%d" % (s, b)]))

# 3 profiles
p = np.load(os.path.join(out, "profiles.npz"))
save(os.path.join(sm, "profiles_wind.npz"), rb=p["rb"], key=p["key"], ws=p["ws"].astype(np.float16), vt=p["vt"].astype(np.float16), vr=p["vr"].astype(np.float16))
save(os.path.join(sm, "profiles_msl.npz"), key=p["key"], msl_hpa=(p["msl"] / 100.0).astype(np.float32), vgrad=p["vgrad"].astype(np.float16) if "vgrad" in p else np.zeros(1))

# 4 crops with fewer members
for f in sorted(glob.glob(os.path.join(out, "crops", "*.npz"))):
    z = dict(np.load(f)); dst = os.path.join(sm, "crops", os.path.basename(f))
    for nm in (2, 1):
        zz = {k: (v[:nm] if k.endswith(("_uv", "_msl")) and not k.startswith("ctl") and v.ndim >= 2 else v) for k, v in z.items()}
        zz = {k: (np.round(v / 10.0).astype(np.int16) if k.endswith("_msl") else v) for k, v in zz.items()}   # msl in units of 10 Pa, int16 offset below
        for k in list(zz):
            if k.endswith("_msl"):
                zz[k] = (np.round(z[k][:nm] / 10.0 if not k.startswith("ctl") else z[k] / 10.0) - 10000).astype(np.int16)
        if save(dst, msl_unit="(value+10000)*10 Pa", **zz) <= LIM:
            break

# 5 full-resolution maps
def png(fn, **kw):
    for dpi in (130, 100, 80):
        plots.plt.rcParams["savefig.dpi"] = dpi
        plots.ratio_maps(out, fn, **kw)
        if os.path.getsize(fn) <= LIM: break
    print("PNG", fn, os.path.getsize(fn), flush=True)
tag = os.path.basename(os.path.normpath(out))
png(os.path.join(sm, "maps_box_%s.jpg" % tag), title="%s against truth members, %s" % (os.environ.get("FB_ROW", "GLB450e2_m"), tag))
png(os.path.join(sm, "maps_europe_%s.jpg" % tag), extent=(34, 64, -14, 32), title="Europe, %s" % tag)
png(os.path.join(sm, "maps_watl_%s.jpg" % tag), extent=(12, 46, -78, -30), title="West Atlantic, %s" % tag)
png(os.path.join(sm, "maps_med_%s.jpg" % tag), extent=(28, 48, -8, 38), title="Mediterranean, %s" % tag)
plots.spectra_fig(out, os.path.join(sm, "spectra_%s.png" % tag), title=tag)
print("REDUCE_DONE", tag)
