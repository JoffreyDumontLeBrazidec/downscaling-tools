#!/usr/bin/env python3
"""Storm-centred structure per member on the XSET-1 row files (discs of 850 km around each best-track fix).
usage: xset_tc.py --cases tc_cases.csv --rows ROOT --row ENS9 --row GL512_500k --out xset_tc_members.csv
Files: ROOT/<ROW>/predictions_<date>_step<LLL>.nc with y_pred(sample, ensemble_member, grid_point, weather_state),
lat_hres, lon_hres, weather_state names holding 10u, 10v, msl. ADAPT the three lines marked LAYOUT if the names differ.
Centre of each member = msl minimum within 300 km of the best-track fix (as xset_score_events.py); structure() of accumulate.py."""
import argparse, os, csv
import numpy as np, netCDF4
from scipy.spatial import cKDTree
import accumulate as A

ap = argparse.ArgumentParser(); ap.add_argument("--cases", required=True); ap.add_argument("--rows", required=True)
ap.add_argument("--row", action="append", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
cases = list(csv.DictReader(open(a.cases)))
out = []; cache = {}
for row in a.row:
    for c in cases:
        fn = os.path.join(a.rows, row, "predictions_%s_step%03d.nc" % (c["date"], int(c["lead"])))
        if not os.path.exists(fn):
            print("MISSING", fn); continue
        if fn not in cache:
            cache.clear()
            with netCDF4.Dataset(fn) as d:
                names = [str(x) for x in d["weather_state"][:]]                                   # LAYOUT
                lat = np.asarray(d["lat_hres"][:], float); lon = np.asarray(d["lon_hres"][:], float)   # LAYOUT
                iu, iv, ip = names.index("10u"), names.index("10v"), names.index("msl")
                yp = np.asarray(d["y_pred"][0][:, :, [iu, iv, ip]], np.float32)                     # LAYOUT [member, point, 3]
            lon = np.where(lon > 180, lon - 360, lon)
            cache[fn] = (lat, lon, yp)
        lat, lon, yp = cache[fn]
        blat, blon = float(c["lat"]), float(c["lon"]); blon = blon - 360 if blon > 180 else blon
        sel = np.where(A.gc_km(blat, blon, lat, lon) <= 840.0)[0]
        if len(sel) < 2000:
            print("FEW_POINTS", c["sid"], c["date"], c["lead"], row, len(sel)); continue
        la, lo = lat[sel], lon[sel]; tree = cKDTree(A.xyz(la, lo))
        for m in range(yp.shape[0]):
            u, v, p = yp[m, sel, 0], yp[m, sel, 1], yp[m, sel, 2]
            if not np.isfinite(p).all():
                continue
            clat, clon, edge = A.centre_near(la, lo, p, tree, blat, blon, 300.0)
            try:
                res, prof, _ = A.structure(la, lo, u, v, p, tree, clat, clon)
            except Exception as e:
                print("STRUCT_FAIL", c["sid"], row, m, e); continue
            res.update(sid=c["sid"], name=c["name"], basin=c["basin"], date=c["date"], lead=c["lead"], row=row, member=m + 1, clat=clat, clon=clon,
                       on_edge=int(edge), bt_lat=blat, bt_lon=blon, bt_wind_kt=c["wind_kt"], bt_pres_hpa=c["pres_hpa"], bt_rmw_nmi=c["usa_rmw_nmi"],
                       dist2land_km=c["dist2land_km"], d_from_bt_km=float(A.gc_km(clat, clon, blat, blon)))
            out.append(res)
    print("ROW_DONE", row, len(out), flush=True)
keys = []
for r in out:
    for k in r:
        if k not in keys: keys.append(k)
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
print("XSET_TC_DONE", len(out))
