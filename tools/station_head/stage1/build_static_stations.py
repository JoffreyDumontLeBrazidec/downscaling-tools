"""Static features of every station seen in the 2026 retrieval, against the O1280 grid.

Reuses the recipe of
/home/ecm5702/agent-work/20260908-aifs-data-search/scratch/station_terrain.py:
the static fields (land-sea mask, sub-grid orography standard deviation, slope,
surface geopotential) come from the O1280 analysis archive, at variable indices
that this script re-checks against the store variables attribute before using them.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
from scipy.spatial import cKDTree

ROOT = Path("/home/ecm5702/agent-work/20260909-station-head-adapter")
STORE = "/ec/ai/project/ai-ml/datasets/aifs-od-an-oper-0001-mars-o1280-2016-2023-6h-v1.zarr"
OUT = ROOT / "outputs" / "static_stations.parquet"
G_CONST = 9.80665
KNN = 12
EARTH_R = 6371.0

IDX = {"lsm": 10, "sdor": 25, "slor": 31, "z": 87}
attrs = json.load(open(STORE + "/.zattrs"))
for name, i in IDX.items():
    assert attrs["variables"][i] == name, (name, i, attrs["variables"][i])
print("variable indices verified against the store attribute:", IDX, flush=True)


def xyz(la, lo):
    la = np.radians(np.asarray(la, dtype=float))
    lo = np.radians(np.asarray(lo, dtype=float))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def chord_to_km(d):
    return 2.0 * np.arcsin(np.clip(np.asarray(d) / 2.0, 0.0, 1.0)) * EARTH_R


# ---- the stations seen in any retrieval ---------------------------------
frames = []
for f in sorted((ROOT / "outputs" / "stations_2026").glob("*.parquet")):
    frames.append(pd.read_parquet(f, columns=["stnid", "latitude", "longitude",
                                              "elevation", "parameter"]))
obs = pd.concat(frames, ignore_index=True)
print("observation rows read:", len(obs), flush=True)
st = (obs.groupby("stnid")
        .agg(latitude=("latitude", "median"), longitude=("longitude", "median"),
             elevation=("elevation", "median"), n_reports=("parameter", "size"))
        .reset_index())
print("distinct stations:", len(st), flush=True)

# ---- the grid and its static fields -------------------------------------
g = zarr.open(STORE, mode="r")
glat = np.asarray(g["latitudes"][:])
glon = np.asarray(g["longitudes"][:])
print("grid points", glat.size, flush=True)
static = {k: np.asarray(g["data"][0, i, 0, :], dtype=np.float32) for k, i in IDX.items()}
for k, v in static.items():
    print("read", k, float(np.nanmin(v)), float(np.nanmax(v)), flush=True)

tree = cKDTree(xyz(glat, glon))
pts = xyz(st["latitude"].to_numpy(), st["longitude"].to_numpy())
dk, jk = tree.query(pts, k=KNN)
d1, j1 = dk[:, 0], jk[:, 0]
nb = tree.query_ball_point(pts, r=15.0 / EARTH_R)
lsm_frac = np.array([static["lsm"][np.asarray(n)].mean() if len(n) else np.nan for n in nb])

sdor = static["sdor"][j1]
slor = static["slor"][j1]
zmod = static["z"][j1] / G_CONST
el = st["elevation"].to_numpy(dtype=float)

terrain = np.where(sdor > 100.0, "mountain", np.where(sdor > 30.0, "hilly", "flat"))
lat = st["latitude"].to_numpy()
lon180 = ((st["longitude"].to_numpy() + 180.0) % 360.0) - 180.0
region = np.where((lat >= 35) & (lat <= 75) & (lon180 >= -12.5) & (lon180 <= 42.5), "europe",
          np.where(lat >= 20, "n.hem.other",
          np.where(lat > -20, "tropics", "s.hem")))

st["nearest_index"] = j1.astype(np.int64)
st["nearest_distance_km"] = chord_to_km(d1)
st["model_height_m"] = zmod
st["sdor"] = sdor
st["slor"] = slor
st["lsm_frac_15km"] = lsm_frac
st["terrain_class"] = terrain
st["coastal"] = (lsm_frac > 0.05) & (lsm_frac < 0.95)
st["region"] = region
st["station_minus_model_height_m"] = el - zmod
st["knn_indices"] = [row.astype(np.int64).tolist() for row in jk]
st["knn_distances_km"] = [chord_to_km(row).tolist() for row in dk]


def hash_holdout(s: str) -> bool:
    return int(hashlib.sha1(str(s).encode()).hexdigest(), 16) % 100 < 5


st["holdout_station"] = st["stnid"].map(hash_holdout)
method = "hash: sha1(stnid) modulo 100 below 5"


def strata_table(frame):
    rows = []
    for col in ["terrain_class", "region"]:
        for v, sub in frame.groupby(col):
            rows.append(dict(stratum=f"{col}={v}", n=len(sub),
                             n_holdout=int(sub["holdout_station"].sum()),
                             pct_holdout=100.0 * sub["holdout_station"].mean()))
    return pd.DataFrame(rows)


tab = strata_table(st)
print(tab.to_string(index=False), flush=True)
worst = (tab["pct_holdout"] - 5.0).abs().max()
print("largest stratum departure from 5 per cent: %.2f points" % worst, flush=True)

if worst > 1.0:
    print("stratified draw instead, seed 20260909", flush=True)
    rng = np.random.default_rng(20260909)
    flag = np.zeros(len(st), dtype=bool)
    for _, sub in st.groupby(["terrain_class", "region"]):
        idx = sub.index.to_numpy()
        k = int(round(0.05 * len(idx)))
        if k:
            flag[rng.choice(idx, size=k, replace=False)] = True
    st["holdout_station"] = flag
    method = "stratified draw by (terrain_class, region), numpy default_rng seed 20260909"
    tab = strata_table(st)
    print(tab.to_string(index=False), flush=True)
    worst = (tab["pct_holdout"] - 5.0).abs().max()
    print("largest stratum departure after the draw: %.2f points" % worst, flush=True)

print("HOLDOUT_METHOD:", method, flush=True)
print("held out overall: %d of %d = %.2f %%"
      % (st["holdout_station"].sum(), len(st), 100 * st["holdout_station"].mean()), flush=True)

st.to_parquet(OUT, index=False)
tab.to_csv(ROOT / "outputs" / "holdout_strata.csv", index=False)
(ROOT / "outputs" / "holdout_method.txt").write_text(method + "\n")
print("written", OUT, flush=True)
