"""Cut small box crops out of a plot-only prediction file (x = native ENS input, y = iekm, y_pred = model).

usage: python extract_crops.py OUT.npz PRED.nc MEMBER LONMIN LONMAX LATMIN LATMAX [--tc]
--tc: recentre the box on the iekm msl minimum inside the given box, then crop +-7 deg lon, +-6 deg lat.
Writes float32 arrays: lat_lres, lon_lres, x_<state>, lat_hres, lon_hres, y_<state>, p_<state>.
"""
import sys
import numpy as np
import xarray as xr

out, path, member = sys.argv[1], sys.argv[2], int(sys.argv[3])
lo0, lo1, la0, la1 = (float(v) for v in sys.argv[4:8])
tc = "--tc" in sys.argv
ds = xr.open_dataset(path)
states = [str(s) for s in ds["weather_state"].values]
want = [s for s in ("10u", "10v", "2t", "msl", "tp") if s in states]


def member_arr(name, m):
    d = ds[name]
    if "sample" in d.dims:
        d = d.isel(sample=0)
    if "ensemble_member" in d.dims:
        d = d.sel(ensemble_member=m)
    d = d.transpose(..., "weather_state")
    return np.asarray(d.values)


def wrap(lon):
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


latl, lonl = np.asarray(ds["lat_lres"].values, float), wrap(ds["lon_lres"].values)
lath, lonh = np.asarray(ds["lat_hres"].values, float), wrap(ds["lon_hres"].values)
X, Y, P = member_arr("x", member), member_arr("y", member), member_arr("y_pred", member)

if tc:
    sel = (lonh >= lo0) & (lonh <= lo1) & (lath >= la0) & (lath <= la1)
    i = np.flatnonzero(sel)[np.nanargmin(Y[sel, states.index("msl")])]
    c_lon, c_lat = lonh[i], lath[i]
    lo0, lo1, la0, la1 = c_lon - 7, c_lon + 7, c_lat - 6, c_lat + 6
    print(f"TC centre (iekm msl min) {c_lat:.2f} N {c_lon:.2f} E, {Y[i, states.index('msl')]:.0f}")

m = 0.5
kl = (lonl >= lo0 - m) & (lonl <= lo1 + m) & (latl >= la0 - m) & (latl <= la1 + m)
kh = (lonh >= lo0 - m) & (lonh <= lo1 + m) & (lath >= la0 - m) & (lath <= la1 + m)
res = dict(lat_lres=latl[kl], lon_lres=lonl[kl], lat_hres=lath[kh], lon_hres=lonh[kh],
           box=np.array([lo0, lo1, la0, la1]), member=member, file=path, states=np.array(want))
for s in want:
    j = states.index(s)
    res["x_" + s] = X[kl, j]
    res["y_" + s] = Y[kh, j]
    res["p_" + s] = P[kh, j]
res = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype.kind == "f" else v) for k, v in res.items()}
np.savez_compressed(out, **res)
print(out, "lres", kl.sum(), "hres", kh.sum(), "states", want,
      "iekm identical across members 1,2:", np.array_equal(member_arr("y", 1), member_arr("y", 2), equal_nan=True))
