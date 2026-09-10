"""Full-year 2026 gates (copied from the summer verify_full.py, names, counts and probe dates changed): shared synthetic axis,
member count, valid-time pairing checked against a DIRECT MARS retrieval on one version 1 start (January) and one
version 2 start (July)."""
import json, os, subprocess, sys, tempfile, numpy as np
from anemoi.datasets import open_dataset
S = "/home/ecm5702/scratch/data/aifscrps_2026_full_20260910/training"
IN = f"{S}/downscaling-ai-pf-enfo-0001-mars-o320-2026-2026-12h-6h-v1-aifscrps.zarr"
TG = f"{S}/downscaling-od-an-oper-0001-mars-o1280-2026-2026-12h-6h-v1-aifscrps-validtime.zarr"
FO = f"{S}/downscaling-od-an-oper-0001-mars-o1280-2026-2026-12h-6h-v1-aifscrps-validtime-forcings.zarr"
ok = True
def check(c, msg):
    global ok; print(("PASS " if c else "FAIL ") + msg, flush=True); ok = ok and bool(c)
zi, zt, zf = open_dataset(IN), open_dataset(TG), open_dataset(FO)
for n, z in (("input", zi), ("target", zt), ("forcings", zf)): print(n, z.shape, "vars", len(z.variables))
mi, mt, mf = (json.load(open(f"{p}/.zattrs"))["fake_forecasts"] for p in (IN, TG, FO))
check(mi == mt == mf, "identical fake_forecasts mapping across the three stores")
check(len(mi) == 1008, f"1008 synthetic dates = 504 inits x 2 leads (got {len(mi)})")
check(zi.shape[2] == 50 and zt.shape[2] == 1 and zf.shape[2] == 1, f"ensemble dims 50/1/1 (got {zi.shape[2]}/{zt.shape[2]}/{zf.shape[2]})")
check(zi.shape[1] == 68 and zt.shape[1] == 68, "68 variables in input and target")
check(list(zi.variables) == list(zt.variables), "same variable order in input and target")
check(zi.shape[0] == zt.shape[0] == zf.shape[0] == 1008, "same length")
# valid-time pairing: (init 2026-06-07 12 UTC, lead 12) -> analysis 2026-07-08 00 UTC ; input member 7
inv = {tuple(v): k for k, v in mi.items()}
key = inv[("2026-07-07T12:00:00", 12)]
dates = [str(d).replace(" ", "T")[:19] for d in zt.dates]; i = dates.index(key[:19])
def mars_field(req, short):
    import eccodes as ec
    with tempfile.TemporaryDirectory() as td:
        tgt = f"{td}/f.grib"; r = req + f', target="{tgt}"'
        subprocess.run(["mars"], input=r.encode(), check=True, capture_output=True)
        with open(tgt, "rb") as f:
            h = ec.codes_grib_new_from_file(f); v = ec.codes_get_values(h); ec.codes_release(h)
    return v
t2 = zt.name_to_index["2t"]
an = mars_field("retrieve,class=od,stream=oper,expver=0001,type=an,date=20260708,time=0000,step=0,levtype=sfc,param=167,grid=O1280", "2t")
d = np.abs(zt[i, t2, 0, :].astype(np.float64) - an).max()
check(d < 1e-3, f"target (init 07-07 12Z, +12h) == analysis 2026-07-08 00 UTC retrieved directly from MARS (max abs diff {d:.2e} K)")
ai = mars_field("retrieve,class=ai,model=aifs-ens,stream=enfo,expver=0001,type=pf,number=7,date=20260707,time=1200,step=12,levtype=sfc,param=167,grid=O320", "2t")
d2 = np.abs(zi[i, zi.name_to_index["2t"], 6, :].astype(np.float64) - ai).max()
check(d2 < 1e-3, f"input ensemble slot 6 == AIFS member 7 (init 07-07 12Z, +12h) from MARS (max abs diff {d2:.2e} K)")
check(np.isfinite(zt[i]).all() and np.isfinite(zi[i]).all() and np.isfinite(zf[i]).all(), "no NaN in the checked sample")
m1 = zi[i, zi.name_to_index["msl"], 0, :]; m2 = zi[i, zi.name_to_index["msl"], 1, :]
check(not np.array_equal(m1, m2), "members differ (msl member 1 vs 2)")
ins = zf.name_to_index["insolation"]; k6 = dates.index(inv[("2026-07-07T12:00:00", 6)][:19])
check(not np.array_equal(zf[i, ins, 0, :], zf[k6, ins, 0, :]), "insolation differs between the 6 h and 12 h samples of one init (valid-time based)")
# second probe on a VERSION 1 start: (init 2026-01-20 00 UTC, lead 6) -> analysis 2026-01-20 06 UTC ; input member 13
key1 = inv[("2026-01-20T00:00:00", 6)]; i1 = dates.index(key1[:19])
an1 = mars_field("retrieve,class=od,stream=oper,expver=0001,type=an,date=20260120,time=0600,step=0,levtype=sfc,param=167,grid=O1280", "2t")
d3 = np.abs(zt[i1, t2, 0, :].astype(np.float64) - an1).max()
check(d3 < 1e-3, f"target (init 01-20 00Z, +6h, version 1 period) == analysis 2026-01-20 06 UTC from MARS (max abs diff {d3:.2e} K)")
ai1 = mars_field("retrieve,class=ai,model=aifs-ens,stream=enfo,expver=0001,type=pf,number=13,date=20260120,time=0000,step=6,levtype=sfc,param=167,grid=O320", "2t")
d4 = np.abs(zi[i1, zi.name_to_index["2t"], 12, :].astype(np.float64) - ai1).max()
check(d4 < 1e-3, f"input ensemble slot 12 == AIFS version 1 member 13 (init 01-20 00Z, +6h) from MARS (max abs diff {d4:.2e} K)")
check(np.isfinite(zt[i1]).all() and np.isfinite(zi[i1]).all() and np.isfinite(zf[i1]).all(), "no NaN in the January sample")
# the two halves of the year are different model versions: both sides of the boundary must be present
check(("2026-05-12T00:00:00", 12) in inv and ("2026-05-12T12:00:00", 6) in inv, "both sides of the 12 May version boundary are present")
st = zt.statistics; print("target stats mean 2t", float(st["mean"][t2]), "stdev", float(st["stdev"][t2]))
print("ALL_PASS" if ok else "SOME_FAIL"); sys.exit(0 if ok else 1)
