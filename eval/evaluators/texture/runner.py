"""Texture evaluator: fine-scale texture statistics on the NATIVE O1280 grid.

Why this exists
---------------
The o320->o1280 diffusion downscaler adds fine-scale structure to its output.
The open question is whether that structure is "grain" -- roughly the right
amount of small-scale energy, but placed point by point instead of organised
into coherent features -- or texture that looks like the truth. Power spectra
cannot tell the two apart (they are blind to phase), and the earlier map-space
probe regridded the field onto a lat-lon box before measuring, which mixes the
regridding kernel into the answer. This evaluator measures texture directly on
the 6,599,680 O1280 points, with NO regridding, so the model output and the
truth receive identical treatment.

Definitions
-----------
For one prediction file, one ensemble member and one weather state, with the
O320 driver ``x``, the model output ``y_pred`` and the truth ``y``::

    x_interp = up @ x                          driver linearly interpolated to O1280
    r_truth  = (y      - x_interp) / stdev[state]
    r_model  = (y_pred - x_interp) / stdev[state]

``stdev`` is the training residual standard deviation of that state, so both
residuals are in the units the network itself works in. The "fine part" of a
residual is what a linear round trip through the O320 grid cannot carry, which
is the lane's own definition of the scales the driver does not resolve::

    rf = r - up @ (down @ r)

Statistics per stratum (a boolean mask over the grid points), computed for the
truth and for the model:

    resid_var        variance of the full residual r (sanity: normalised units)
    fine_var         variance of rf
    zonal_diff_var   variance of r[next] - r[i], with next the cyclic successor of
                     i along its latitude row: grid-scale roughness, no regridding
    fine_lag1_zonal  correlation between rf[i] and rf[next]
    fine_nn_corr     correlation between rf[i] and the mean of rf over the 6
                     nearest neighbours of i (great-circle neighbours, self excluded)
    top5_share       share of the stratum's total rf**2 carried by the 5% of
                     points with the largest rf**2
    kurtosis         excess kurtosis of rf
    n_points         number of points in the stratum

plus the ratios model/truth of the three variances and the differences
model - truth of the two correlations.

White-noise reference and grain index. The fine-part operator (I - up@down) is
a sharp high-pass filter and leaves its own signature on every statistic: on
this grid, Gaussian white noise comes out with fine_lag1_zonal of about -0.66
and fine_nn_corr of about +0.11 (measured 2026-09-02), NOT zero, while a field
that is smooth at the grid scale gives lag-1 near 0 and fine_nn_corr near 0.85.
So the evaluator also pushes Gaussian white noise through the same operator
(fixed seeds, once per run) and reports those values per stratum as the
``noise`` reference, together with a grain index for the two correlations::

    grain_index = (model - truth) / (noise - truth)

which is 0 when the model's fine part is textured like the truth and 1 when it
is indistinguishable from white noise through the same filter.

Strata: ``all``; three terrain classes from the lane's forcings zarr (``ocean``,
``flat_land``, ``mountain``, from the land-sea mask and the standard deviation
of the orography over the 32 nearest neighbours); and the region boxes from the
``regions:`` mapping of the evaluator config block.

Everything is aggregated over the (file, member) samples per (state, stratum):
mean and standard deviation of each statistic. The standard deviation of the
model - truth correlation differences is the null scatter that later arms are
judged against.

Static data (built once per run, cached on disk where expensive):
  * the 32-nearest-neighbour table of the O1280 grid, cached as an .npz;
  * the zonal-successor index (cyclic along each latitude row);
  * the terrain classes and region masks.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)

# z_500 is left out by default: its fine band is numerically delicate (the
# truth carries regrid moire at those scales) and would dominate any pooled
# reading. Ask for it explicitly through weather_states when needed.
DEFAULT_STATES = ["10u", "10v", "2t", "msl", "t_850"]

# [lat_min, lat_max, lon_min, lon_max], longitudes in -180..180.
DEFAULT_REGIONS: dict[str, list[float]] = {
    "europe": [35.0, 60.0, -10.0, 25.0],
    "alps": [43.0, 49.0, 4.0, 16.0],
    "open_north_atlantic": [35.0, 55.0, -45.0, -20.0],
    "west_tropical_atlantic": [10.0, 25.0, -70.0, -45.0],
}

DEFAULT_TERRAIN = {
    "ocean_lsm_max": 0.05,          # ocean: land-sea mask below this
    "land_lsm_min": 0.95,           # land classes: land-sea mask above this
    "flat_roughness_max_m": 30.0,   # flat_land: orography roughness below this
    "mountain_roughness_min_m": 150.0,  # mountain: orography roughness above this
}
TERRAIN_CLASSES = ["ocean", "flat_land", "mountain"]

KNN_K = 32              # neighbours kept in the cache (the point itself excluded)
NN_COUNT = 6            # neighbours averaged for fine_nn_corr
TOP_FRACTION = 0.05     # top5_share
EARTH_RADIUS_KM = 6371.0088
GRAVITY = 9.80665

STAT_NAMES = [
    "resid_var", "fine_var", "zonal_diff_var",
    "fine_lag1_zonal", "fine_nn_corr", "top5_share", "kurtosis",
]
RATIO_STATS = ["resid_var", "fine_var", "zonal_diff_var"]
DELTA_STATS = ["fine_lag1_zonal", "fine_nn_corr"]
NOISE_SEEDS = (20260902, 20260903)   # white-noise reference draws, fixed for reproducibility

_FILE_RE = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")


def _default_paths() -> dict[str, str]:
    inter = os.environ.get("INTER_MAT_DIR", "/home/ecm5702/hpcperm/data/inter_mat")
    resid = os.environ.get(
        "RESIDUAL_STATISTICS_DIR", "/home/ecm5702/hpcperm/data/residuals_statistics"
    )
    return {
        "up_matrix": os.path.join(inter, "interpol_O320_to_O1280_linear.mat.npz"),
        "down_matrix": os.path.join(inter, "interpol_o1280_to_o320_linear.mat.npz"),
        "residual_stats": os.path.join(resid, "o1280_dict_0_72.npy"),
        "forcings_zarr": (
            "/home/mlx/ai-ml/datasets/"
            "downscaling-od-cf-enfh-0001-mars-o1280-2003-2023-12h-v3-forcings.zarr"
        ),
        "knn_cache": "/home/ecm5702/hpcperm/data/static/o1280_knn32.npz",
    }


# ---------------------------------------------------------------------------
# Static grid structures
# ---------------------------------------------------------------------------

def _unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    c = np.cos(lat)
    return np.column_stack([c * np.cos(lon), c * np.sin(lon), np.sin(lat)])


def _build_knn(lat: np.ndarray, lon: np.ndarray, k: int, chunk: int = 400_000):
    """k nearest neighbours of every grid point (self excluded), great-circle km."""
    from scipy.spatial import cKDTree

    xyz = _unit_vectors(lat, lon)
    n = xyz.shape[0]
    t0 = time.time()
    tree = cKDTree(xyz)
    LOG.info("texture: KD-tree on %d points built in %.1fs; querying k=%d", n, time.time() - t0, k)
    idx = np.empty((n, k), dtype=np.int32)
    dist_km = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d, i = tree.query(xyz[s:e], k=k + 1, workers=-1)
        # The query returns the point itself (chord distance 0) among the k+1;
        # drop it wherever it sits. Should it ever be absent (it never is on a
        # grid without coincident points) drop the farthest column instead.
        drop = i == np.arange(s, e)[:, None]
        drop[~drop.any(axis=1), -1] = True
        keep = ~drop
        idx[s:e] = i[keep].reshape(e - s, k)
        chord = d[keep].reshape(e - s, k)
        dist_km[s:e] = EARTH_RADIUS_KM * 2.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))
    LOG.info("texture: kNN query finished in %.1fs", time.time() - t0)
    return idx, dist_km


def _load_or_build_knn(path: str | Path, lat: np.ndarray, lon: np.ndarray, k: int):
    """Return (idx, dist_km, built_now); idx/dist exclude the point itself."""
    p = Path(path)
    if p.exists():
        with np.load(p) as z:
            idx = np.asarray(z["idx"])
            dist = np.asarray(z["dist_km"])
        if idx.shape[0] != lat.size:
            raise RuntimeError(
                f"kNN cache {p} has {idx.shape[0]} rows but the grid has {lat.size} points"
            )
        # A cache written from raw cKDTree output carries the point itself in
        # column 0 (distance exactly 0); strip it so neighbours are neighbours.
        probe = min(1000, idx.shape[0])
        if np.array_equal(idx[:probe, 0], np.arange(probe)) and float(dist[:probe, 0].max()) == 0.0:
            idx = idx[:, 1:]
            dist = dist[:, 1:]
        LOG.info("texture: loaded kNN cache %s (%d neighbours)", p, idx.shape[1])
        return idx, dist, False

    idx, dist = _build_knn(lat, lon, k)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez(
        tmp, idx=idx, dist_km=dist, k=np.int32(k), self_excluded=np.bool_(True),
        note=np.array(
            "O1280 grid, storage order of the forcings zarr / prediction files; "
            "idx[i] = the k nearest neighbours of point i (self excluded) by "
            "great-circle distance, nearest first; dist_km = great-circle km."
        ),
    )
    os.replace(tmp, p)
    LOG.info("texture: wrote kNN cache %s", p)
    return idx, dist, True


def _zonal_successor(lat: np.ndarray, lon: np.ndarray):
    """Index of the cyclic successor of every point along its latitude row.

    Rows are groups of equal latitude (rounded to 1e-4 deg); inside a row the
    points are ordered by longitude in 0..360 and the last one wraps to the
    first. Works on any storage order.
    """
    n = lat.size
    key = np.round(np.asarray(lat, dtype=np.float64), 4)
    lon360 = np.mod(np.asarray(lon, dtype=np.float64), 360.0)
    order = np.lexsort((lon360, -key))          # north to south, then west to east
    ks = key[order]
    brk = np.flatnonzero(np.diff(ks) != 0.0) + 1
    starts = np.concatenate(([0], brk))
    ends = np.concatenate((brk, [n]))
    succ_sorted = np.arange(1, n + 1)
    succ_sorted[ends - 1] = starts               # wrap every row onto itself
    nxt = np.empty(n, dtype=np.int32)
    nxt[order] = order[succ_sorted]
    lens = ends - starts
    info = {"n_rows": int(len(starts)), "row_len_min": int(lens.min()), "row_len_max": int(lens.max())}
    return nxt, info


def _roughness(oro_m: np.ndarray, knn_idx: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    """Standard deviation of the orography (m) over each point's neighbours."""
    n = oro_m.size
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        out[s:e] = np.std(oro_m[knn_idx[s:e]].astype(np.float64), axis=1)
    return out


def _region_mask(lat: np.ndarray, lon180: np.ndarray, box) -> np.ndarray:
    lat_min, lat_max, lon_min, lon_max = (float(v) for v in box)
    m = (lat >= lat_min) & (lat <= lat_max)
    if lon_min <= lon_max:
        m &= (lon180 >= lon_min) & (lon180 <= lon_max)
    else:  # box crossing the dateline
        m &= (lon180 >= lon_min) | (lon180 <= lon_max)
    return m


def _build_static(paths: dict, regions: dict, terrain: dict, nn_count: int) -> dict[str, Any]:
    import zarr

    t0 = time.time()
    zpath = paths["forcings_zarr"]
    z = zarr.open(zpath, mode="r")
    with open(os.path.join(zpath, ".zattrs")) as fh:
        variables = json.load(fh)["variables"]
    lat = np.asarray(z["latitudes"][:], dtype=np.float64)
    lon = np.asarray(z["longitudes"][:], dtype=np.float64)
    block = np.asarray(z["data"][0, :, 0, :], dtype=np.float32)   # (n_vars, n_points), time 0
    lsm = block[variables.index("lsm")]
    oro_m = block[variables.index("z")] / GRAVITY
    del block
    n = lat.size
    LOG.info("texture: forcings read (%d points) in %.1fs", n, time.time() - t0)

    knn_idx, knn_dist, built = _load_or_build_knn(paths["knn_cache"], lat, lon, KNN_K)
    nn_dist_km = {
        "nearest_km_median": float(np.median(knn_dist[:, 0])),
        "sixth_km_median": float(np.median(knn_dist[:, min(nn_count, knn_dist.shape[1]) - 1])),
    }
    del knn_dist
    rough = _roughness(oro_m, knn_idx)
    nn = np.ascontiguousarray(knn_idx[:, :nn_count])
    del knn_idx

    nxt, row_info = _zonal_successor(lat, lon)
    lon360 = np.mod(lon, 360.0)
    lon180 = np.where(lon360 > 180.0, lon360 - 360.0, lon360)

    strata: dict[str, dict[str, Any]] = {}
    masks: dict[str, np.ndarray | None] = {"all": None}
    strata["all"] = {"kind": "all", "n_points": int(n)}
    ocean = lsm < float(terrain["ocean_lsm_max"])
    land = lsm > float(terrain["land_lsm_min"])
    class_masks = {
        "ocean": ocean,
        "flat_land": land & (rough < float(terrain["flat_roughness_max_m"])),
        "mountain": land & (rough > float(terrain["mountain_roughness_min_m"])),
    }
    for name in TERRAIN_CLASSES:
        m = class_masks[name]
        masks[name] = m
        strata[name] = {"kind": "terrain", "n_points": int(m.sum())}
    for name, box in regions.items():
        m = _region_mask(lat, lon180, box)
        masks[name] = m
        strata[name] = {"kind": "region", "box": [float(v) for v in box], "n_points": int(m.sum())}

    for name, m in masks.items():
        if m is not None and int(m.sum()) < 1000:
            LOG.warning("texture: stratum %r has only %d points", name, int(m.sum()))

    LOG.info(
        "texture: static structures ready in %.1fs (rows=%d, knn_built=%s, strata=%s)",
        time.time() - t0, row_info["n_rows"], built,
        {k: v["n_points"] for k, v in strata.items()},
    )
    return {
        "lat": lat, "lon": lon, "nn": nn, "nxt": nxt,
        "masks": masks, "strata": strata, "strata_order": list(masks.keys()),
        "grid": {
            "n_points": int(n), **row_info,
            "knn_cache": str(paths["knn_cache"]), "knn_built_this_run": bool(built),
            **nn_dist_km,
            "roughness_m": {
                "p50": float(np.percentile(rough, 50)),
                "p90": float(np.percentile(rough, 90)),
                "max": float(rough.max()),
            },
        },
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _var(a: np.ndarray) -> float:
    d = a - a.mean()
    return float(np.dot(d, d) / a.size)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    da = a - a.mean()
    db = b - b.mean()
    denom = float(np.sqrt(np.dot(da, da) * np.dot(db, db)))
    return float(np.dot(da, db) / denom) if denom > 0.0 else float("nan")


def _kurtosis(a: np.ndarray) -> float:
    d = a - a.mean()
    d2 = d * d
    m2 = float(d2.mean())
    m4 = float(np.dot(d2, d2) / a.size)
    return m4 / (m2 * m2) - 3.0 if m2 > 0.0 else float("nan")


def _top_share(a: np.ndarray, frac: float) -> float:
    sq = a * a
    n = sq.size
    k = max(1, int(np.ceil(frac * n)))
    total = float(sq.sum())
    if total <= 0.0 or k >= n:
        return float("nan")
    top = float(np.partition(sq, n - k)[n - k:].sum())
    return top / total


def _field_stats(r, rf, rf_next, rf_nn, dz, sel, frac: float) -> dict[str, float]:
    take = (lambda v: v) if sel is None else (lambda v: np.take(v, sel))
    rf_s = take(rf)
    return {
        "resid_var": _var(take(r)),
        "fine_var": _var(rf_s),
        "zonal_diff_var": _var(take(dz)),
        "fine_lag1_zonal": _corr(rf_s, take(rf_next)),
        "fine_nn_corr": _corr(rf_s, take(rf_nn)),
        "top5_share": _top_share(rf_s, frac),
        "kurtosis": _kurtosis(rf_s),
        "n_points": int(rf_s.size),
    }


def _texture_arrays(r: np.ndarray, up, down, nxt: np.ndarray, nn: np.ndarray):
    rf = r - up @ (down @ r)
    dz = r[nxt] - r
    rf_next = rf[nxt]
    rf_nn = rf[nn].mean(axis=1)
    return rf, dz, rf_next, rf_nn


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if (np.isfinite(num) and np.isfinite(den) and den != 0.0) else float("nan")


def _grain_index(model: float, truth: float, noise) -> float:
    """(model - truth) / (noise - truth): 0 = textured like the truth, 1 = white noise."""
    if noise is None:
        return float("nan")
    den = float(noise) - truth
    if not (np.isfinite(model) and np.isfinite(truth) and np.isfinite(den)) or abs(den) < 1e-6:
        return float("nan")
    return float((model - truth) / den)


def _noise_reference(n_points: int, up, down, nxt, nn, sels: dict, strata_order: list[str],
                     frac: float, seeds=NOISE_SEEDS) -> dict[str, dict[str, dict]]:
    """Statistics of Gaussian white noise pushed through the same fine-part operator,
    per stratum: {stratum: {stat: {mean, sd, n}}} over the seeds."""
    per_stratum: dict[str, list[dict]] = {s: [] for s in strata_order}
    for seed in seeds:
        w = np.random.default_rng(int(seed)).standard_normal(n_points)
        arrs = _texture_arrays(w, up, down, nxt, nn)
        for stratum in strata_order:
            per_stratum[stratum].append(_field_stats(w, *arrs, sels[stratum], frac))
        del arrs
    return {
        stratum: {stat: _mean_sd([d[stat] for d in dicts]) for stat in STAT_NAMES}
        for stratum, dicts in per_stratum.items()
    }


def _clean(obj):
    """Replace NaN/inf by None recursively so the JSON is strict."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def _mean_sd(vals) -> dict[str, Any]:
    v = np.asarray([np.nan if x is None else x for x in vals], dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": None, "sd": None, "n": 0}
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else None,
        "n": int(v.size),
    }


def _aggregate(samples: list[dict], states: list[str], strata_order: list[str],
               noise_ref: dict) -> list[dict]:
    rows: list[dict] = []
    for state in states:
        for stratum in strata_order:
            sub = [s for s in samples if s["state"] == state and s["stratum"] == stratum]
            if not sub:
                continue
            row: dict[str, Any] = {
                "state": state, "stratum": stratum, "n_samples": len(sub),
                "truth": {}, "model": {}, "ratio": {}, "delta": {},
                "noise": dict(noise_ref.get(stratum, {})), "grain_index": {},
            }
            for side in ("truth", "model"):
                for stat in STAT_NAMES + ["n_points"]:
                    row[side][stat] = _mean_sd([s[side][stat] for s in sub])
            for stat in RATIO_STATS:
                row["ratio"][stat] = _mean_sd([s["ratio"][stat] for s in sub])
            for stat in DELTA_STATS:
                row["delta"][stat] = _mean_sd([s["delta"][stat] for s in sub])
                row["grain_index"][stat] = _mean_sd([s["grain_index"][stat] for s in sub])
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Reading prediction files
# ---------------------------------------------------------------------------

def _read_member(var, member_index: int) -> np.ndarray:
    """One member of a (sample, ensemble_member, grid_point, weather_state) variable
    as a (points, states) array, whatever the on-disk dimension order."""
    dims = var.dimensions
    index = []
    for d in dims:
        if d == "sample":
            index.append(0)
        elif d == "ensemble_member":
            index.append(member_index)
        else:
            index.append(slice(None))
    arr = np.asarray(var[tuple(index)])
    remaining = [d for d in dims if d not in ("sample", "ensemble_member")]
    if arr.ndim == 2 and remaining and remaining[0] == "weather_state":
        arr = arr.T
    return arr


def _lonlat_match(lat_a, lon_a, lat_b, lon_b, tol_deg: float = 1e-3) -> bool:
    if lat_a.shape != lat_b.shape:
        return False
    dlat = float(np.abs(np.asarray(lat_a, dtype=np.float64) - np.asarray(lat_b, dtype=np.float64)).max())
    dlon = np.asarray(lon_a, dtype=np.float64) - np.asarray(lon_b, dtype=np.float64)
    dlon = float(np.abs(np.mod(dlon + 180.0, 360.0) - 180.0).max())
    return dlat < tol_deg and dlon < tol_deg


def _as_list(value, cast=str) -> list | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [cast(tok.strip()) for tok in value.split(",") if tok.strip()]
    return [cast(v) for v in value]


def _select_members(ds, members: list[int] | None, max_members: int | None):
    """[(label, index)] of the members to process."""
    if "ensemble_member" in ds.dimensions:
        n_mem = len(ds.dimensions["ensemble_member"])
        if "ensemble_member" in ds.variables:
            labels = [int(v) for v in np.asarray(ds.variables["ensemble_member"][:]).reshape(-1)]
        else:
            labels = list(range(1, n_mem + 1))
    else:
        labels = [1]
    chosen = list(enumerate(labels))
    if members:
        wanted = [int(m) for m in members]
        by_label = {lab: i for i, lab in chosen}
        missing = [m for m in wanted if m not in by_label]
        if missing:
            raise RuntimeError(f"members {missing} not present in file (has {labels})")
        chosen = [(by_label[m], m) for m in wanted]
    if max_members:
        chosen = chosen[: int(max_members)]
    return [(lab, i) for i, lab in chosen]


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _fmt(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}"


def _write_summary(path: Path, payload: dict) -> None:
    lines = [
        f"# texture -- {payload['run_label']}",
        "",
        f"Predictions: `{payload['predictions_dir']}`  ",
        f"Files: {payload['n_files']}, samples per (state, stratum): up to "
        f"{payload['n_samples_per_cell']}, states: {', '.join(payload['states'])}  ",
        "Columns: T = truth, M = model. lag1 = fine_lag1_zonal (correlation of the fine "
        "part with its zonal neighbour); nn_corr = fine_nn_corr (correlation with the mean "
        "of the 6 nearest neighbours); fine_var and zonal_diff_var as ratios M/T; "
        "top5 = share of the fine energy in the top 5% of points; kurt = excess kurtosis. "
        "The +- values are standard deviations over the (file, member) samples.",
        "",
        "| state | stratum | n | lag1 T | lag1 M | nn_corr T | nn_corr M | fine_var M/T "
        "| zonal_diff_var M/T | top5 T | top5 M | kurt T | kurt M |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        t, m, ra = row["truth"], row["model"], row["ratio"]
        lines.append(
            "| {state} | {stratum} | {n} | {l1t} | {l1m} | {nnt} | {nnm} | {fv} | {zv} "
            "| {t5t} | {t5m} | {kt} | {km} |".format(
                state=row["state"], stratum=row["stratum"], n=row["n_samples"],
                l1t=_fmt(t["fine_lag1_zonal"]["mean"]), l1m=_fmt(m["fine_lag1_zonal"]["mean"]),
                nnt=_fmt(t["fine_nn_corr"]["mean"]), nnm=_fmt(m["fine_nn_corr"]["mean"]),
                fv=_fmt(ra["fine_var"]["mean"]), zv=_fmt(ra["zonal_diff_var"]["mean"]),
                t5t=_fmt(t["top5_share"]["mean"]), t5m=_fmt(m["top5_share"]["mean"]),
                kt=_fmt(t["kurtosis"]["mean"], 2), km=_fmt(m["kurtosis"]["mean"], 2),
            )
        )
    lines += [
        "",
        "White-noise reference: Gaussian white noise pushed through the same fine-part operator "
        f"(seeds {payload.get('noise_seeds')}), per stratum. This is where pure grain sits; the "
        "operator is a sharp high-pass, so its lag-1 correlation is strongly negative, not zero.",
        "",
        "| stratum | lag1 N | nn_corr N | top5 N | kurt N | fine_var/var N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stratum, ref in payload.get("noise_reference", {}).items():
        fv = ref["fine_var"]["mean"]
        rv = ref["resid_var"]["mean"]
        lines.append(
            f"| {stratum} | {_fmt(ref['fine_lag1_zonal']['mean'])} | {_fmt(ref['fine_nn_corr']['mean'])} "
            f"| {_fmt(ref['top5_share']['mean'])} | {_fmt(ref['kurtosis']['mean'], 2)} "
            f"| {_fmt(None if (fv is None or not rv) else fv / rv)} |"
        )
    lines += [
        "",
        "Model - truth differences of the correlations (mean +- sd over samples; the sd is the "
        "null scatter later arms are judged against) and the grain index "
        "(model - truth) / (noise - truth): 0 = textured like the truth, 1 = white noise.",
        "",
        "| state | stratum | d lag1 | d nn_corr | grain (lag1) | grain (nn_corr) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        d = row["delta"]
        g = row["grain_index"]
        lines.append(
            f"| {row['state']} | {row['stratum']} | "
            f"{_fmt(d['fine_lag1_zonal']['mean'])} +- {_fmt(d['fine_lag1_zonal']['sd'])} | "
            f"{_fmt(d['fine_nn_corr']['mean'])} +- {_fmt(d['fine_nn_corr']['sd'])} | "
            f"{_fmt(g['fine_lag1_zonal']['mean'], 2)} +- {_fmt(g['fine_lag1_zonal']['sd'], 2)} | "
            f"{_fmt(g['fine_nn_corr']['mean'], 2)} +- {_fmt(g['fine_nn_corr']['sd'], 2)} |"
        )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    run_label: str = "",
    **kwargs,
) -> Path:
    import netCDF4
    import scipy.sparse as sps

    from eval.evaluators.spectra.proxy_runner import valid_prediction_files

    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "texture"
    output_dir.mkdir(parents=True, exist_ok=True)

    states = _as_list(eval_config.get("weather_states")) or list(DEFAULT_STATES)
    steps = _as_list(eval_config.get("steps"), int)
    dates = _as_list(eval_config.get("dates"), str)
    members = _as_list(eval_config.get("members"), int)
    max_members = eval_config.get("max_members")
    regions = eval_config.get("regions")
    if regions is None:
        regions = dict(DEFAULT_REGIONS)
    for name, box in regions.items():
        if len(box) != 4:
            raise ValueError(f"texture.regions[{name!r}] must be [lat_min, lat_max, lon_min, lon_max]")
    terrain = {**DEFAULT_TERRAIN, **(eval_config.get("terrain") or {})}
    paths = {**_default_paths(), **(eval_config.get("paths") or {})}
    nn_count = int(eval_config.get("nn_count", NN_COUNT))
    frac = float(eval_config.get("top_fraction", TOP_FRACTION))

    files = valid_prediction_files(predictions_dir, steps=steps)
    if dates:
        wanted = set(dates)

        def _file_date(f: Path) -> str | None:
            m = _FILE_RE.search(f.name)
            return m.group(1) if m else None

        files = [f for f in files if _file_date(f) in wanted]
    if not files:
        raise RuntimeError(f"No prediction files under {predictions_dir} (steps={steps}, dates={dates})")

    LOG.info(
        "texture: %d file(s), states=%s steps=%s dates=%s members=%s max_members=%s regions=%s",
        len(files), states, steps, dates, members, max_members, list(regions),
    )

    t0 = time.time()
    up = sps.load_npz(paths["up_matrix"]).tocsr()
    down = sps.load_npz(paths["down_matrix"]).tocsr()
    stdev_all = np.load(paths["residual_stats"], allow_pickle=True).item()["stdev"]
    missing_sd = [s for s in states if s not in stdev_all]
    if missing_sd:
        raise RuntimeError(f"residual stdev missing for {missing_sd} in {paths['residual_stats']}")
    stdev = {s: float(stdev_all[s]) for s in states}
    LOG.info("texture: matrices up=%s down=%s loaded in %.1fs", up.shape, down.shape, time.time() - t0)

    static = _build_static(paths, regions, terrain, nn_count)
    n_points = static["grid"]["n_points"]
    if up.shape[0] != n_points or down.shape[1] != n_points:
        raise RuntimeError(
            f"grid size mismatch: forcings {n_points} points, up {up.shape}, down {down.shape}"
        )
    masks = static["masks"]
    strata_order = static["strata_order"]
    sels = {name: (None if m is None else np.flatnonzero(m)) for name, m in masks.items()}

    t_noise = time.time()
    noise_ref = _noise_reference(
        n_points, up, down, static["nxt"], static["nn"], sels, strata_order, frac,
    )
    LOG.info(
        "texture: white-noise reference in %.1fs (all: lag1=%.3f nn_corr=%.3f)",
        time.time() - t_noise,
        noise_ref["all"]["fine_lag1_zonal"]["mean"], noise_ref["all"]["fine_nn_corr"]["mean"],
    )
    noise_mean = {s: {k: noise_ref[s][k]["mean"] for k in DELTA_STATS} for s in strata_order}

    samples: list[dict[str, Any]] = []
    for file_path in files:
        m = _FILE_RE.search(file_path.name)
        date, step = (m.group(1), int(m.group(2))) if m else ("", -1)
        t_file = time.time()
        with netCDF4.Dataset(file_path) as ds:
            ds.set_auto_mask(False)
            for required in ("x", "y", "y_pred"):
                if required not in ds.variables:
                    raise RuntimeError(f"{file_path} is missing {required!r}")
            ws = [str(v) for v in np.asarray(ds.variables["weather_state"][:]).reshape(-1)]
            idx_of = {s: i for i, s in enumerate(ws)}
            absent = [s for s in states if s not in idx_of]
            if absent:
                LOG.warning("texture: %s lacks states %s; skipping them", file_path.name, absent)
            lat_f = np.asarray(ds.variables["lat_hres"][:]).reshape(-1)
            lon_f = np.asarray(ds.variables["lon_hres"][:]).reshape(-1)
            if not _lonlat_match(lat_f, lon_f, static["lat"], static["lon"]):
                raise RuntimeError(
                    f"{file_path.name}: lat/lon_hres do not match the forcings grid "
                    f"({lat_f.size} vs {n_points} points); the evaluator relies on identical point order"
                )
            lead = ds.getncattr("lead_step_hours") if "lead_step_hours" in ds.ncattrs() else step

            for member_label, member_index in _select_members(ds, members, max_members):
                t_mem = time.time()
                yp = np.asarray(_read_member(ds.variables["y_pred"], member_index), dtype=np.float64)
                yt = np.asarray(_read_member(ds.variables["y"], member_index))
                xx = np.asarray(_read_member(ds.variables["x"], member_index))
                t_read = time.time() - t_mem
                for state in states:
                    si = idx_of.get(state)
                    if si is None:
                        continue
                    x_interp = up @ np.asarray(xx[:, si], dtype=np.float64)
                    sd = stdev[state]
                    r_t = (np.asarray(yt[:, si], dtype=np.float64) - x_interp) / sd
                    r_m = (yp[:, si] - x_interp) / sd
                    finite = np.isfinite(r_t) & np.isfinite(r_m)
                    all_finite = bool(finite.all())
                    if not all_finite:
                        LOG.warning(
                            "texture: %s member %s %s has %d non-finite points; excluded",
                            file_path.name, member_label, state, int((~finite).sum()),
                        )
                        r_t = np.where(finite, r_t, 0.0)
                        r_m = np.where(finite, r_m, 0.0)
                    arrs_t = _texture_arrays(r_t, up, down, static["nxt"], static["nn"])
                    arrs_m = _texture_arrays(r_m, up, down, static["nxt"], static["nn"])
                    for stratum in strata_order:
                        sel = sels[stratum]
                        if not all_finite:
                            base = np.ones(n_points, dtype=bool) if masks[stratum] is None else masks[stratum]
                            sel = np.flatnonzero(base & finite)
                        st_t = _field_stats(r_t, *arrs_t, sel, frac)
                        st_m = _field_stats(r_m, *arrs_m, sel, frac)
                        samples.append({
                            "file": file_path.name, "date": date, "step": int(lead),
                            "member": int(member_label), "state": state, "stratum": stratum,
                            "truth": st_t, "model": st_m,
                            "ratio": {k: _safe_ratio(st_m[k], st_t[k]) for k in RATIO_STATS},
                            "delta": {k: st_m[k] - st_t[k] for k in DELTA_STATS},
                            "grain_index": {
                                k: _grain_index(st_m[k], st_t[k], noise_mean[stratum][k])
                                for k in DELTA_STATS
                            },
                        })
                    del arrs_t, arrs_m
                LOG.info(
                    "texture: %s member %s done (read %.1fs, total %.1fs)",
                    file_path.name, member_label, t_read, time.time() - t_mem,
                )
        LOG.info("texture: %s finished in %.1fs", file_path.name, time.time() - t_file)

    aggregate = _aggregate(samples, states, strata_order, noise_ref)
    n_cell = max((r["n_samples"] for r in aggregate), default=0)
    payload = {
        "run_label": run_label or predictions_dir.name,
        "predictions_dir": str(predictions_dir),
        "n_files": len(files),
        "files": [f.name for f in files],
        "n_samples_per_cell": n_cell,
        "states": [s for s in states if any(r["state"] == s for r in aggregate)],
        "strata_order": strata_order,
        "strata": static["strata"],
        "grid": static["grid"],
        "config": {
            "weather_states": states, "steps": steps, "dates": dates,
            "members": members, "max_members": max_members,
            "regions": regions, "terrain": terrain, "paths": paths,
            "knn_k": KNN_K, "nn_count": nn_count, "top_fraction": frac,
            "stdev": stdev,
        },
        "statistics": STAT_NAMES,
        "ratio_statistics": RATIO_STATS,
        "delta_statistics": DELTA_STATS,
        "grain_statistics": DELTA_STATS,
        "noise_seeds": [int(s) for s in NOISE_SEEDS],
        "noise_reference": noise_ref,
        "aggregate": aggregate,
        "samples": samples,
        "elapsed_s": time.time() - t0,
    }
    payload = _clean(payload)
    (output_dir / "texture.json").write_text(json.dumps(payload, indent=1) + "\n")
    _write_summary(output_dir / "texture_summary.md", payload)
    LOG.info(
        "texture: wrote %s (%d samples, %d aggregate rows) in %.1fs",
        output_dir / "texture.json", len(samples), len(aggregate), time.time() - t0,
    )
    return output_dir
