"""Turn the gathered stage 2a case tables into the arrays the station head trains on.

One gathered parquet file holds, for one case (one initialisation and one lead) and
for every station of the static table and every ensemble member, the predicted and
the interpolated value of ten weather states at the twelve output points nearest the
station. This module turns one such file into a compact set of numpy arrays for one
target variable, applying every rule the design note fixed:

* the valid time 1 September 2026 at 00 UTC is excluded from everything, because the
  station network is less than half its usual size at that time;
* a station is on the STABLE network when it reports at more than 90 per cent of the
  valid times of the window for that parameter, which is what keeps the validation
  window comparable with the training window across the Italian feed interruption of
  late August; the full network is carried beside it, never instead of it;
* the gross-error screen of quaver is applied exactly as the `obs_crps` backend
  applies it, that is the hard physical limits of `toss.yaml` and the maximum
  departure from the operational analysis, whose values are copied here from
  `/home/ecm5702/dev/downscaling-tools/eval/_backends/obs_crps/obs_crps_compute.py`
  (read only, never modified);
* the hold-out stations are flagged so that training can drop them.

The three targets are 2 m temperature, dewpoint and 10 m wind speed. Wind speed is
not a predicted field: it is derived at every neighbour from the predicted `10u` and
`10v`, and the same derivation is applied to the interpolated input, so the head and
its controls see the speed the observation reports.

Features, in the order the arrays carry them:

* the neighbourhood, twice: the eleven states (the ten the prediction file carries
  plus the derived wind speed) at each of the twelve nearest output points, once from
  the prediction and once from the interpolated AIFS input, that is 132 numbers each;
* the static features of the station relative to the output grid, 31 numbers;
* the time of day as a sine and a cosine, 2 numbers.

The output of one case is one uncompressed npz file, so that training can stream the
cases rather than hold the whole first cut in memory. These files are regenerable
from the gathered tables, so they go on scratch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PERM = Path("/home/ecm5702/perm/station-head-adapter")
STAGE1 = PERM / "stage1_20260909" / "outputs"
GATHERED = PERM / "gathered"
ANALYSIS = PERM / "analysis_at_stations"
MANIFESTS = PERM / "manifests"
DATASET_ROOT = Path("/home/ecm5702/scratch/eval/station_head_2a/datasets")

# Copied from eval/_backends/obs_crps/obs_crps_compute.py so that the head is scored
# under the same rules as every station scorecard of the epic. That file is read only.
LAPSE_RATE = 0.0065
HARD_LIMITS = {
    "2t": (173.0, 333.0),
    "2d": (150.0, 333.0),
    "10ff": (0.0, 100.0),
    "msl": (80000.0, 110000.0),
}
ANALYSIS_DEPARTURE_MAX = {"2t": 30.0, "10ff": 40.0, "msl": 1000.0}
OROGRAPHY_CORRECTED = {"2t"}

EXCLUDED_VALID = pd.Timestamp("2026-09-01 00:00:00")
STABLE_FRACTION = 0.9
K = 12
# The ten weather states a stage 2a prediction file carries, plus the derived speed.
STATES = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "t_850", "tcw", "z_500"]
STATES_EXT = STATES + ["10ff"]
TARGETS = ["2t", "2d", "10ff"]
TERRAIN_CLASSES = ["flat", "hilly", "mountain"]
REGIONS = ["europe", "n.hem.other", "tropics", "s.hem"]

STATIC_NAMES = (
    ["elevation", "model_height_m", "station_minus_model_height_m", "sdor", "slor",
     "lsm_frac_15km", "coastal", "latitude", "sin_longitude", "cos_longitude",
     "nearest_distance_km"]
    + ["knn_distance_km_k%02d" % j for j in range(K)]
    + ["terrain_" + t for t in TERRAIN_CLASSES]
    + ["region_" + r for r in REGIONS]
    + ["lead_h"]
)
NEIGHBOUR_NAMES = ["%s_k%02d" % (s, j) for s in STATES_EXT for j in range(K)]
FEATURE_NAMES = (["pred_" + n for n in NEIGHBOUR_NAMES]
                 + ["xinterp_" + n for n in NEIGHBOUR_NAMES]
                 + STATIC_NAMES + ["sin_hour", "cos_hour"])
N_NEIGH_BLOCK = len(NEIGHBOUR_NAMES)      # 132, one block
N_NEIGH = 2 * N_NEIGH_BLOCK               # 264, prediction and interpolated input
N_STATIC = len(STATIC_NAMES)              # 31
N_TIME = 2


# ---------------------------------------------------------------- stable network

def stable_network(rebuild: bool = False) -> pd.DataFrame:
    """One row per station and parameter with the report count and the stable flag.

    The count is the number of distinct valid times of the window at which the
    station reported that parameter, with the excluded valid time left out of both
    the count and the number of possible reports. A station is stable when it
    reports at more than 90 per cent of the possible times.
    """
    out = MANIFESTS / "stable_network.parquet"
    if out.exists() and not rebuild:
        return pd.read_parquet(out)
    pm = pd.read_parquet(STAGE1 / "pairing_manifest.parquet")
    pm["valid"] = pd.to_datetime(pm["valid"])
    keep_idx = set(pm.loc[pm["valid"] != EXCLUDED_VALID, "synthetic_index"].astype(int))
    n_possible = len(keep_idx)
    frames = []
    for p in ["2t", "2d", "10ff", "msl", "tp"]:
        f = STAGE1 / "stations_2026" / (p + ".parquet")
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=["synthetic_index", "stnid"])
        d = d[d["synthetic_index"].astype(int).isin(keep_idx)]
        c = d.groupby("stnid").size().rename("n_reports_param").reset_index()
        c["parameter"] = p
        c["n_possible"] = n_possible
        c["stable"] = c["n_reports_param"] > STABLE_FRACTION * n_possible
        frames.append(c)
        print("%s: %d stations, %d stable of %d possible valid times"
              % (p, len(c), int(c["stable"].sum()), n_possible), flush=True)
    res = pd.concat(frames, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out, index=False)
    print("wrote " + str(out), flush=True)
    return res


# ---------------------------------------------------------------- static features

def static_matrix(stations: pd.DataFrame, lead_h: int) -> np.ndarray:
    n = len(stations)
    cols = []
    elev = stations["elevation"].to_numpy(dtype=np.float64)
    # About 2,400 stations have no reported elevation. The model height is the best
    # stand-in and makes the station-minus-model difference zero for them, which is
    # what "no correction is possible" means.
    model_h = stations["model_height_m"].to_numpy(dtype=np.float64)
    elev = np.where(np.isfinite(elev), elev, model_h)
    smm = elev - model_h
    lon = np.radians(stations["longitude"].to_numpy(dtype=np.float64))
    cols += [elev, model_h, smm,
             stations["sdor"].to_numpy(dtype=np.float64),
             stations["slor"].to_numpy(dtype=np.float64),
             stations["lsm_frac_15km"].to_numpy(dtype=np.float64),
             stations["coastal"].to_numpy().astype(np.float64),
             stations["latitude"].to_numpy(dtype=np.float64),
             np.sin(lon), np.cos(lon),
             stations["nearest_distance_km"].to_numpy(dtype=np.float64)]
    knn = np.stack(stations["knn_distances_km"].to_numpy())[:, :K].astype(np.float64)
    cols += [knn[:, j] for j in range(K)]
    tc = stations["terrain_class"].to_numpy()
    cols += [(tc == t).astype(np.float64) for t in TERRAIN_CLASSES]
    rg = stations["region"].to_numpy()
    cols += [(rg == r).astype(np.float64) for r in REGIONS]
    cols += [np.full(n, float(lead_h))]
    m = np.stack(cols, axis=1).astype(np.float32)
    assert m.shape[1] == N_STATIC, (m.shape, N_STATIC)
    return m


# ---------------------------------------------------------------- one case

def _neighbour_block(df_m: list[pd.DataFrame], prefix: str) -> np.ndarray:
    """(n_stations, n_members, 132) for one of the two blocks."""
    per_member = []
    for d in df_m:
        cols = []
        u = np.stack([d["%s_10u_k%02d" % (prefix, j)].to_numpy(dtype=np.float32) for j in range(K)], axis=1)
        v = np.stack([d["%s_10v_k%02d" % (prefix, j)].to_numpy(dtype=np.float32) for j in range(K)], axis=1)
        speed = np.sqrt(u * u + v * v)
        for s in STATES:
            cols.append(np.stack([d["%s_%s_k%02d" % (prefix, s, j)].to_numpy(dtype=np.float32)
                                  for j in range(K)], axis=1))
        cols.append(speed)
        per_member.append(np.concatenate(cols, axis=1))
    return np.stack(per_member, axis=1).astype(np.float32)


def build_case(case_id: str, target: str, stations: pd.DataFrame,
               stable: dict[str, set], analysis: pd.DataFrame,
               gathered_root: Path = GATHERED) -> dict | None:
    """Return the arrays of one case for one target, or None if the case is excluded."""
    if target not in TARGETS:
        raise ValueError("unknown target " + target)
    path = gathered_root / (case_id + ".parquet")
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    valid = pd.Timestamp(df["valid"].iloc[0])
    if valid == EXCLUDED_VALID:
        print("case %s is the excluded valid time, skipped" % case_id, flush=True)
        return None
    synthetic_index = int(df["synthetic_index"].iloc[0])
    lead_h = int(df["lead_h"].iloc[0])

    obs_col = "obs_" + target
    members = sorted(int(m) for m in pd.unique(df["member"]))
    blocks = []
    order = None
    for m in members:
        d = df[df["member"] == m].set_index("stnid")
        if order is None:
            order = d.index.to_numpy()
        d = d.reindex(order)
        blocks.append(d)

    st = stations.set_index("stnid").reindex(order).reset_index()
    keep = st["stnid"].notna().to_numpy()
    if not keep.all():
        raise RuntimeError("gathered case %s carries stations absent from the static table" % case_id)

    y = blocks[0][obs_col].to_numpy(dtype=np.float64)
    good = np.isfinite(y)

    lo, hi = HARD_LIMITS.get(target, (-np.inf, np.inf))
    good &= (y >= lo) & (y <= hi)

    an = analysis[analysis["synthetic_index"] == synthetic_index]
    an_map = dict(zip(an["stnid"].to_numpy(), an["analysis_value"].to_numpy(dtype=np.float64)))
    an_vals = np.array([an_map.get(s, np.nan) for s in order], dtype=np.float64)
    limit = ANALYSIS_DEPARTURE_MAX.get(target)
    if limit is not None:
        has_an = np.isfinite(an_vals)
        good &= (~has_an) | (np.abs(y - an_vals) <= limit)

    feat_pred = _neighbour_block(blocks, "pred")
    feat_int = _neighbour_block(blocks, "xinterp")
    good &= np.isfinite(feat_pred).all(axis=(1, 2)) & np.isfinite(feat_int).all(axis=(1, 2))

    sel = np.where(good)[0]
    if sel.size == 0:
        print("case %s has no usable %s observation" % (case_id, target), flush=True)
        return None

    order = order[sel]
    st = st.iloc[sel].reset_index(drop=True)
    feat_pred = feat_pred[sel]
    feat_int = feat_int[sel]
    y = y[sel].astype(np.float32)
    an_vals = an_vals[sel].astype(np.float32)

    static = static_matrix(st, lead_h)
    hour = float(valid.hour) + float(valid.minute) / 60.0
    tod = np.stack([np.full(len(st), np.sin(2 * np.pi * hour / 24.0)),
                    np.full(len(st), np.cos(2 * np.pi * hour / 24.0))], axis=1).astype(np.float32)

    # The nearest-point control, with quaver's lapse-rate correction for 2 m
    # temperature only. delta = model height minus station height, exactly as in
    # obs_crps_compute.py.
    i_target = STATES_EXT.index(target if target != "10ff" else "10ff")
    col0 = i_target * K  # neighbour k00 of that state inside a block
    ctrl_near = feat_pred[:, :, col0].astype(np.float32).copy()
    ctrl_int_near = feat_int[:, :, col0].astype(np.float32).copy()
    if target in OROGRAPHY_CORRECTED:
        elev = static[:, STATIC_NAMES.index("elevation")].astype(np.float64)
        mh = static[:, STATIC_NAMES.index("model_height_m")].astype(np.float64)
        delta = (mh - elev) * LAPSE_RATE
        ctrl_near = (ctrl_near + delta[:, None]).astype(np.float32)
        ctrl_int_near = (ctrl_int_near + delta[:, None]).astype(np.float32)

    stable_set = stable.get(target, set())
    return {
        "case_id": case_id,
        "stnid": order.astype(str),
        "feat_pred": feat_pred,
        "feat_int": feat_int,
        "static": static,
        "tod": tod,
        "y": y,
        "analysis": an_vals,
        "ctrl_near": ctrl_near,
        "ctrl_int_near": ctrl_int_near,
        "holdout": st["holdout_station"].to_numpy().astype(bool),
        "stable": np.array([s in stable_set for s in order], dtype=bool),
        "terrain": np.array([TERRAIN_CLASSES.index(t) for t in st["terrain_class"]], dtype=np.int8),
        "region": np.array([REGIONS.index(r) for r in st["region"]], dtype=np.int8),
        "members": np.array(members, dtype=np.int32),
        "lead_h": np.int32(lead_h),
        "synthetic_index": np.int32(synthetic_index),
        "valid": str(valid),
        "init": str(pd.Timestamp(blocks[0]["init"].iloc[0])),
    }


def write_case(arrays: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (arrays["case_id"] + ".npz")
    tmp = out.with_name("." + out.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(out)
    return out


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=TARGETS)
    ap.add_argument("--tag", default="firstcut", help="name of the dataset directory")
    ap.add_argument("--out-root", default=str(DATASET_ROOT))
    ap.add_argument("--cases", default=None,
                    help="comma-separated case ids; default is every gathered case")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--stable-only", action="store_true",
                    help="build the stable-network cache and stop; run this once before "
                         "an assembly array so that the tasks do not race to build it")
    args = ap.parse_args()

    if args.stable_only:
        stable_network(rebuild=True)
        return

    if args.cases:
        case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        case_ids = sorted(p.stem for p in GATHERED.glob("*.parquet"))
    case_ids = case_ids[args.shard::args.nshards]
    out_dir = Path(args.out_root) / ("%s_%s" % (args.target, args.tag))
    print("target %s, %d cases, out %s" % (args.target, len(case_ids), out_dir), flush=True)

    stations = pd.read_parquet(STAGE1 / "static_stations.parquet")
    sn = stable_network()
    stable = {p: set(sn.loc[(sn["parameter"] == p) & sn["stable"], "stnid"])
              for p in sn["parameter"].unique()}
    af = ANALYSIS / (args.target + ".parquet")
    analysis = (pd.read_parquet(af, columns=["synthetic_index", "stnid", "analysis_value"])
                if af.exists() else pd.DataFrame(columns=["synthetic_index", "stnid", "analysis_value"]))
    print("analysis rows for %s: %d" % (args.target, len(analysis)), flush=True)

    written = 0
    for cid in case_ids:
        out = out_dir / (cid + ".npz")
        if out.exists() and not args.overwrite:
            continue
        t0 = time.time()
        arr = build_case(cid, args.target, stations, stable, analysis)
        if arr is None:
            continue
        p = write_case(arr, out_dir)
        written += 1
        print("wrote %s stations=%d members=%d in %.1f s"
              % (p, arr["y"].size, arr["feat_pred"].shape[1], time.time() - t0), flush=True)

    meta = out_dir / "features.json"
    if written or not meta.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps({
            "target": args.target,
            "states": STATES_EXT,
            "k": K,
            "n_neighbourhood_block": N_NEIGH_BLOCK,
            "n_neighbourhood": N_NEIGH,
            "n_static": N_STATIC,
            "n_time": N_TIME,
            "feature_names": FEATURE_NAMES,
            "excluded_valid": str(EXCLUDED_VALID),
            "stable_fraction": STABLE_FRACTION,
            "hard_limits": HARD_LIMITS,
            "analysis_departure_max": ANALYSIS_DEPARTURE_MAX,
            "lapse_rate": LAPSE_RATE,
        }, indent=2))
    print("shard %d wrote %d case files" % (args.shard, written), flush=True)


if __name__ == "__main__":
    sys.exit(main())
