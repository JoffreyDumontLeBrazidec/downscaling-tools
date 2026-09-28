"""The analysis value at every station, for all 446 valid times.

Stage 1 did this for a one-in-eight subsample of the calendar and measured that a
whole valid time costs about one and a half seconds, so the full calendar is a
ten-minute batch job. This script is the same computation over every synthetic
index. It reads 2 m temperature, 2 m dewpoint, the two 10 m wind components and
mean sea level pressure out of the built target store, which is the O1280 analysis
at the valid time, and takes the value at the nearest O1280 point of every station
that reported. The 10 m wind speed is formed from the two components the way
quaver forms it.

The output keeps only the join keys and the two values, because the initialisation,
the lead, the valid time and the station coordinates are all recoverable from the
pairing manifest and the static station table; that keeps the file small enough to
live on perm beside the rest of the adapter. One parquet file per parameter is
written so that the job never holds the whole table in memory.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

STAGE1 = Path("/home/ecm5702/perm/station-head-adapter/stage1_20260909/outputs")
TARGET = ("/home/ecm5702/scratch/data/anemoi_datasets_aifsens2_20260907/"
          "downscaling-od-an-oper-0001-mars-o1280-2026-2026-12h-6h-v1-aifsens2-validtime.zarr")
STATIC_STORE = "/ec/ai/project/ai-ml/datasets/aifs-od-an-oper-0001-mars-o1280-2016-2023-6h-v1.zarr"
OUT_DIR = Path("/home/ecm5702/perm/station-head-adapter/analysis_at_stations")
VAR_INDEX = {"2t": 3, "2d": 2, "10u": 0, "10v": 1, "msl": 4}
PARAMETERS = ["2t", "2d", "10ff", "msl"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = zarr.open(TARGET, mode="r")
    tattrs = json.load(open(TARGET + "/.zattrs"))
    for name, i in VAR_INDEX.items():
        assert tattrs["variables"][i] == name, (name, i, tattrs["variables"][i])

    gs = zarr.open(STATIC_STORE, mode="r")
    assert np.allclose(np.asarray(g["latitudes"][:]), np.asarray(gs["latitudes"][:]))
    assert np.allclose(np.asarray(g["longitudes"][:]), np.asarray(gs["longitudes"][:]))
    print("grids agree:", g["latitudes"].shape[0], "points", flush=True)

    static_st = pd.read_parquet(STAGE1 / "static_stations.parquet",
                                columns=["stnid", "nearest_index"])
    nearest = dict(zip(static_st["stnid"], static_st["nearest_index"]))

    man = pd.read_parquet(STAGE1 / "pairing_manifest.parquet")
    indices = sorted(man["synthetic_index"].tolist())
    print("valid times:", len(indices), flush=True)

    obs = {}
    for p in PARAMETERS:
        d = pd.read_parquet(STAGE1 / "stations_2026" / f"{p}.parquet",
                            columns=["synthetic_index", "stnid", "value"])
        d["nearest_index"] = d["stnid"].map(nearest)
        d = d[d["nearest_index"].notna()].copy()
        d["nearest_index"] = d["nearest_index"].astype(np.int64)
        obs[p] = {si: sub for si, sub in d.groupby("synthetic_index")}
        print("observations for", p, len(d), flush=True)

    parts: dict[str, list[pd.DataFrame]] = {p: [] for p in PARAMETERS}
    t0 = time.time()
    for n, si in enumerate(indices, 1):
        block = np.asarray(g["data"][si, 0:5, 0, :], dtype=np.float32)
        fields = {k: block[i] for k, i in VAR_INDEX.items()}
        fields["10ff"] = np.hypot(fields["10u"], fields["10v"])
        for p in PARAMETERS:
            dd = obs[p].get(si)
            if dd is None or not len(dd):
                continue
            out = dd[["synthetic_index", "stnid", "value"]].copy()
            out["analysis_value"] = fields[p][dd["nearest_index"].to_numpy()]
            parts[p].append(out)
        if n % 25 == 0:
            print("%d/%d t=%.1fs" % (n, len(indices), time.time() - t0), flush=True)

    for p in PARAMETERS:
        res = pd.concat(parts[p], ignore_index=True)
        res["parameter"] = p
        path = OUT_DIR / f"{p}.parquet"
        res.to_parquet(path, index=False)
        dep = (res["value"] - res["analysis_value"]).to_numpy(dtype=float)
        dep = dep[np.isfinite(dep)]
        print("%-5s rows=%9d valid_times=%4d median %+10.3f p5 %+10.3f p95 %+10.3f -> %s"
              % (p, len(res), res["synthetic_index"].nunique(), np.median(dep),
                 np.percentile(dep, 5), np.percentile(dep, 95), path), flush=True)


if __name__ == "__main__":
    main()
