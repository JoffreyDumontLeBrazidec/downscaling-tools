"""Gather the station neighbourhood features out of a stage 2a prediction file.

For one case, that is one initialisation and one lead, this reads the prediction
file the harness wrote and, for every station of the static table and every
member, records the predicted value and the interpolated AIFS input value at the
twelve output points nearest the station. The neighbour indices are built on the
longitude and latitude arrays of the prediction file itself, never on the ordering
of any store, because the two need not agree.

The neighbour index is the same for every case, since every prediction file is on
the same O1280 output grid, so it is built once and cached. Each case checks the
cache against the coordinates of its own file before using it, by comparing the
number of points and the sum of the coordinates, and rebuilds if they disagree.

Everything that does not vary from case to case, namely the distances, the terrain
class, the model height and the hold-out flag, stays in the static station table
and is joined on the station identifier at training time. The per-case file
therefore holds only what the case itself produced, plus the station observations
of that valid time, joined through the pairing manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

STAGE1 = Path("/home/ecm5702/perm/station-head-adapter/stage1_20260909/outputs")
PERM = Path("/home/ecm5702/perm/station-head-adapter")
CACHE = PERM / "manifests" / "station_knn_on_prediction_grid.npz"
GATHER_ROOT = PERM / "gathered"
PRED_ROOT = Path("/home/ecm5702/scratch/eval/station_head_2a/predictions")
K = 12
OBS_PARAMETERS = ["2t", "2d", "10ff", "msl", "tp"]


def _unit_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=1)


def _grid_signature(lon: np.ndarray, lat: np.ndarray) -> dict:
    return {
        "n": int(lon.size),
        "lon_sum": float(np.sum(np.asarray(lon, dtype=np.float64))),
        "lat_sum": float(np.sum(np.asarray(lat, dtype=np.float64))),
    }


def build_or_load_knn(lon_hres: np.ndarray, lat_hres: np.ndarray,
                      stations: pd.DataFrame, rebuild: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, distances_km) of shape (n_stations, K) in file ordering."""
    sig = _grid_signature(lon_hres, lat_hres)
    if CACHE.exists() and not rebuild:
        z = np.load(CACHE, allow_pickle=True)
        cached = json.loads(str(z["signature"]))
        same_ids = bool(np.array_equal(z["stnid"], stations["stnid"].to_numpy().astype(str)))
        if cached == sig and same_ids:
            return z["indices"], z["distances_km"]
        print("neighbour cache does not match this file, rebuilding", flush=True)
    from scipy.spatial import cKDTree

    tree = cKDTree(_unit_xyz(lon_hres, lat_hres))
    chord, idx = tree.query(_unit_xyz(stations["longitude"].to_numpy(),
                                      stations["latitude"].to_numpy()), k=K)
    # chord length on the unit sphere back to a great-circle distance in kilometres
    dist_km = 2.0 * 6371.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, indices=idx.astype(np.int64), distances_km=dist_km.astype(np.float64),
             stnid=stations["stnid"].to_numpy().astype(str), signature=json.dumps(sig))
    print("neighbour index built and cached at " + str(CACHE), flush=True)
    return idx.astype(np.int64), dist_km.astype(np.float64)


def check_against_static(idx: np.ndarray, stations: pd.DataFrame) -> tuple[int, int]:
    """Compare the nearest point found here with the nearest_index of the static table."""
    mine = idx[:, 0]
    theirs = stations["nearest_index"].to_numpy(dtype=np.int64)
    agree = int(np.sum(mine == theirs))
    return agree, int(mine.size)


def gather_case(pred_file: Path, out_file: Path, stations: pd.DataFrame,
                manifest_row: pd.Series, obs: dict[str, pd.DataFrame]) -> Path:
    ds = netCDF4.Dataset(str(pred_file))
    states = [s.strip() for s in str(ds.getncattr("output_weather_states")).split(",")]
    lon = np.asarray(ds.variables["lon_hres"][:])
    lat = np.asarray(ds.variables["lat_hres"][:])
    idx, dist_km = build_or_load_knn(lon, lat, stations)
    agree, total = check_against_static(idx, stations)
    print("nearest-index agreement with the static table: %d of %d stations" % (agree, total),
          flush=True)

    n_members = ds.dimensions["ensemble_member"].size
    members = np.asarray(ds.variables["ensemble_member"][:]).reshape(-1)
    n_st = len(stations)
    flat = idx.reshape(-1)
    order = np.argsort(flat)
    flat_sorted = flat[order]

    frames = []
    for mi in range(n_members):
        block = {"member": np.full(n_st, int(members[mi]), dtype=np.int32),
                 "stnid": stations["stnid"].to_numpy()}
        for vi, state in enumerate(states):
            for label, var in (("pred", "y_pred"), ("xinterp", "x_interp")):
                if var not in ds.variables:
                    continue
                field = np.asarray(ds.variables[var][0, mi, :, vi], dtype=np.float32)
                vals = field[flat_sorted]
                out = np.empty_like(vals)
                out[order] = vals
                out = out.reshape(n_st, K)
                for j in range(K):
                    block["%s_%s_k%02d" % (label, state, j)] = out[:, j]
        frames.append(pd.DataFrame(block))
    ds.close()

    df = pd.concat(frames, ignore_index=True)
    df["synthetic_index"] = int(manifest_row["synthetic_index"])
    df["init"] = pd.Timestamp(manifest_row["init"])
    df["lead_h"] = int(manifest_row["lead_h"])
    df["valid"] = pd.Timestamp(manifest_row["valid"])

    for p in OBS_PARAMETERS:
        d = obs.get(p)
        if d is None:
            continue
        sub = d[d["synthetic_index"] == int(manifest_row["synthetic_index"])]
        m = dict(zip(sub["stnid"], sub["value"]))
        df["obs_" + p] = df["stnid"].map(m).astype("float32")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_name("." + out_file.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_file)
    print("wrote %s rows=%d cols=%d bytes=%d" % (out_file, len(df), df.shape[1],
                                                 out_file.stat().st_size), flush=True)
    return out_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="case_id, e.g. 2026051212_step006")
    ap.add_argument("--out-root", default=str(GATHER_ROOT))
    ap.add_argument("--prediction", default=None,
                    help="explicit prediction file, instead of the one the case list implies")
    args = ap.parse_args()

    cases = pd.read_csv("/home/ecm5702/agent-work/20260909-station-head-adapter/notes/cases_firstcut.csv")
    row = cases[cases["case_id"] == args.case]
    if not len(row):
        raise SystemExit("no such case: " + args.case)
    row = row.iloc[0]
    init_key = "%s%02d" % (row["date"], int(row["time"]))
    pred = (Path(args.prediction) if args.prediction else
            PRED_ROOT / init_key / "predictions" / ("predictions_%s_step%03d.nc" % (row["date"], int(row["lead_h"]))))
    if not pred.exists():
        raise SystemExit("no prediction file: " + str(pred))

    stations = pd.read_parquet(STAGE1 / "static_stations.parquet")
    obs = {}
    for p in OBS_PARAMETERS:
        f = STAGE1 / "stations_2026" / (p + ".parquet")
        if f.exists():
            obs[p] = pd.read_parquet(f, columns=["synthetic_index", "stnid", "value"])

    out = Path(args.out_root) / (args.case + ".parquet")
    gather_case(pred, out, stations, row, obs)


if __name__ == "__main__":
    main()
