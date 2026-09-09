"""Section 5 of the design note: the analysis value at each station point.

For a one-in-eight subsample of the 446 valid times (every eighth synthetic
index) this reads 2t, 2d, 10u, 10v and msl out of the built target store, which
is the O1280 analysis at the valid time, and takes the value at the nearest
O1280 point of every station that reported at that time.  10ff is derived from
the two wind components the same way quaver derives it.

Each read pulls a whole 34-variable chunk, about 0.9 GB, so this belongs in a
batch job and never on the login node.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import zarr

ROOT = Path("/home/ecm5702/agent-work/20260909-station-head-adapter")
TARGET = ("/home/ecm5702/scratch/data/anemoi_datasets_aifsens2_20260907/"
          "downscaling-od-an-oper-0001-mars-o1280-2026-2026-12h-6h-v1-aifsens2-validtime.zarr")
STATIC = "/ec/ai/project/ai-ml/datasets/aifs-od-an-oper-0001-mars-o1280-2016-2023-6h-v1.zarr"
OUT = ROOT / "outputs" / "analysis_at_stations_subsample.parquet"
STRIDE = 8
TIME_BUDGET_S = 3 * 3600

VAR_INDEX = {"2t": 3, "2d": 2, "10u": 0, "10v": 1, "msl": 4}

g = zarr.open(TARGET, mode="r")
import json
tattrs = json.load(open(TARGET + "/.zattrs"))
for k, i in VAR_INDEX.items():
    assert tattrs["variables"][i] == k, (k, i, tattrs["variables"][i])

# the station table carries indices into the static store grid; the two grids
# must be the same O1280 grid for those indices to be usable here.
gs = zarr.open(STATIC, mode="r")
la_t = np.asarray(g["latitudes"][:])
la_s = np.asarray(gs["latitudes"][:])
lo_t = np.asarray(g["longitudes"][:])
lo_s = np.asarray(gs["longitudes"][:])
assert la_t.shape == la_s.shape and np.allclose(la_t, la_s) and np.allclose(lo_t, lo_s), \
    "the target store grid is not the static store grid; nearest_index cannot be reused"
print("grids agree:", la_t.size, "points", flush=True)

static_st = pd.read_parquet(ROOT / "outputs" / "static_stations.parquet",
                            columns=["stnid", "nearest_index"])
nearest = dict(zip(static_st["stnid"], static_st["nearest_index"]))

man = pd.read_parquet(ROOT / "outputs" / "pairing_manifest.parquet")
sub = man[man["synthetic_index"] % STRIDE == 0]
print("subsample size:", len(sub), flush=True)

obs = {}
for p in ["2t", "2d", "10ff", "msl"]:
    f = ROOT / "outputs" / "stations_2026" / f"{p}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        obs[p] = d[d["synthetic_index"].isin(sub["synthetic_index"])]
        print("observations for", p, len(obs[p]), flush=True)

rows = []
t0 = time.time()
done = 0
for si in sub["synthetic_index"].tolist():
    if time.time() - t0 > TIME_BUDGET_S:
        print("TIME BUDGET REACHED after %d of %d valid times" % (done, len(sub)), flush=True)
        break
    block = np.asarray(g["data"][si, 0:5, 0, :], dtype=np.float32)
    fields = {k: block[i] for k, i in VAR_INDEX.items()}
    fields["10ff"] = np.hypot(fields["10u"], fields["10v"])
    for p, d in obs.items():
        dd = d[d["synthetic_index"] == si]
        if not len(dd):
            continue
        idx = dd["stnid"].map(nearest)
        ok = idx.notna()
        dd = dd[ok]
        ii = idx[ok].to_numpy(dtype=np.int64)
        out = dd[["synthetic_index", "init", "lead_h", "valid", "parameter",
                  "stnid", "latitude", "longitude", "elevation", "value"]].copy()
        out["analysis_value"] = fields[p][ii]
        rows.append(out)
    done += 1
    print("si=%4d %d/%d t=%.1fs" % (si, done, len(sub), time.time() - t0), flush=True)

res = pd.concat(rows, ignore_index=True)
res.to_parquet(OUT, index=False)
print("written", OUT, "rows", len(res), "valid times", res["valid"].nunique(), flush=True)

print("\ndeparture statistics, observation minus analysis:", flush=True)
for p, d in res.groupby("parameter"):
    dep = (d["value"] - d["analysis_value"]).to_numpy(dtype=float)
    dep = dep[np.isfinite(dep)]
    print("  %-5s n=%9d  median %+10.3f  p5 %+10.3f  p95 %+10.3f"
          % (p, dep.size, np.median(dep), np.percentile(dep, 5), np.percentile(dep, 95)),
          flush=True)
