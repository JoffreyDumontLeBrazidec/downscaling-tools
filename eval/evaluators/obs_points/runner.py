"""obs_points run(): one pass over the prediction files, scores at the stations, raw tables only (no figures)."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.discovery.predictions import find_predictions
from eval.evaluators.obs_points.core.scores import _wmean, ensemble_scores, rank_counts, seeps
from eval.evaluators.obs_points.core.stations import (
    StationMatcher, density_weights, height_masks, load_cache, load_seeps_clim, region_masks, resolve_cache, xyz,
)
from eval.evaluators.obs_points.core.summary import paired, summarise, write_summary_md
from eval.shared.provenance import (
    ENSEMBLE_RULE, SINGLE_FIELD, ensemble_construction, is_single_input, members_differ, provenance,
)

LOG = logging.getLogger(__name__)

DEFAULT_VARS = ("2t", "10ff", "msl", "tp")
DEFAULT_REGIONS = {
    "all": None,
    "europe": [35.0, 75.0, -12.5, 42.5],
    "40-70N": [40.0, 70.01, -180.0, 180.0],
    "25-40N": [25.0, 40.0, -180.0, 180.0],
    "10-25N": [10.0, 25.0, -180.0, 180.0],
}
# Subsets of "all" by |O1280 orography - station height|, in metres, [lo, hi).
DEFAULT_HEIGHT_CLASSES = {"all_dz<100m": [0.0, 100.0], "all_dz>=300m": [300.0, 1.0e9]}
DEFAULT_THRESHOLDS = (5.0, 10.0, 20.0)  # mm per 12 h
DEFAULT_RADII = (0.0, 18.0, 36.0)  # km, tp only; 0 = the nearest grid point
LAPSE_RATE = 0.0065  # K/m, quaver
MM = 1000.0

# Kind of each row as it is written into the tables. The input and the truth are ensembles when the file carries
# their members (x_interp_members / y_members, or a member axis on x_interp / y whose members differ, as the
# ENS-member files of 4 Nov 2026 do); otherwise single fields, whose "fair CRPS" is the MAE of one field and is
# not comparable with an ensemble's (audit 5 Nov 2026, D4; rule 34(e)). HRES (STVL) is always a single field.
ROW_KIND = {"model": "ensemble", "input": "single field", "truth": "single field", "hres": "single field"}
ENSEMBLE_KIND = {"input": "ensemble (input members)", "truth": "ensemble (truth members)"}


def rank_seed(base: dict, row: str) -> int:
    """Seed of the random tie-breaking in the rank histogram, fixed per (var, variant, radius, region, lead,
    date, row), so a rerun gives the same counts and two rows never share one random stream."""
    import zlib

    key = "|".join(str(base[k]) for k in ("var", "variant", "radius_km", "region", "lead", "date")) + "|" + row
    return zlib.crc32(key.encode())


def file_provenance(ds, n_model: int) -> dict:
    """What the prediction file says about its rows: the model ensemble's construction (file attributes
    ``member_mode`` / ``competitor_row`` of the member kit, ``source_bundle`` of eval.cli predict, or whether the
    per-member inputs differ) and which of the input and truth rows come as ensembles. ``ds`` is an open
    netCDF4.Dataset."""
    attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
    n = ds["lat_hres"].size
    ens, differ = {}, {}
    for row, names in (("input", ("x_interp_members", "x_interp")), ("truth", ("y_members", "y"))):
        differ[row] = None  # None: the file carries one field only (no member axis, or members fewer than M)
        for name in names:
            if name in ds.variables:
                a = np.asarray(ds[name][0], np.float64).reshape(-1, n, len(ds.dimensions["weather_state"]))
                if a.shape[0] == n_model and n_model > 1:
                    differ[row] = members_differ(a)
                break
        ens[row] = bool(differ[row])
    bundles = None
    if "source_bundle" in ds.variables:
        bundles = [str(b) for b in np.asarray(ds["source_bundle"][:]).reshape(-1)]
    return {"attrs": attrs, "ensemble_rows": ens,
            "model": ensemble_construction(attrs, n_model, input_members_differ=differ["input"],
                                           source_bundles=bundles)}


def read_fields(path, variables, *, ensembles: bool = True, native_input: bool = True):
    """lat, lon (-180..180), {var: {"truth": (n,) or (M, n), "input": (n,) or (M, n), "model": (M, n)}} in
    observation units: 2t K, 10ff m/s (from 10u/10v), msl Pa, tp mm (file metres x 1000, clipped at 0), and the
    file's provenance (``file_provenance``). The input and the truth come as ensembles when the file carries
    their members (``x_interp_members`` / ``y_members``, or a member axis on ``x_interp`` / ``y`` with M members
    that differ); with ``ensembles=False``, or otherwise, member 0 (the control)."""
    import netCDF4

    out = {}
    with netCDF4.Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        ws = [str(s) for s in ds["weather_state"][:]]
        lat = np.asarray(ds["lat_hres"][:], np.float64)
        lon = np.asarray(ds["lon_hres"][:], np.float64)
        n = lat.size
        prov = file_provenance(ds, int(ds["y_pred"].shape[1]))
        if not ensembles:
            prov["ensemble_rows"] = {"input": False, "truth": False}
        # The input is scored on ITS OWN grid (owner, 9 Nov 2026, standing rule 41): the native low-resolution
        # field ``x`` on lat_lres/lon_lres, nearest native point to the station, as quaver does. Interpolating
        # it to the high-resolution grid first smooths the members, narrows their spread and flatters the model
        # (6 h tp: +5 % against the interpolated input, 0 % against quaver's native input, note s19.7).
        # ``x_interp`` is used only when the file has no native input, and the provenance says so.
        native = native_input and "x" in ds.variables and "lat_lres" in ds.variables
        in_n = int(ds["lat_lres"].size) if native else n
        src = {"model": "y_pred",
               "input": "x" if native else ("x_interp_members" if "x_interp_members" in ds.variables else "x_interp"),
               "truth": "y_members" if "y_members" in ds.variables else "y"}
        if native:
            a = np.asarray(ds["x"][0], np.float64).reshape(-1, in_n, len(ws))
            m = int(ds["y_pred"].shape[1])
            prov["ensemble_rows"]["input"] = bool(ensembles and a.shape[0] == m and m > 1 and members_differ(a))
            prov["input_grid"] = {"source": "x (native input grid)", "points": in_n,
                                  "lat": np.asarray(ds["lat_lres"][:], np.float64),
                                  "lon": np.asarray(ds["lon_lres"][:], np.float64)}
        else:
            prov["input_grid"] = {"source": f"{src['input']} (input interpolated to the output grid; no native x "
                                            "in the file)", "points": n, "lat": None, "lon": None}

        def get(row, ch):
            nn = in_n if row == "input" else n
            a = np.asarray(ds[src[row]][0, ..., ws.index(ch)], np.float64).reshape(-1, nn)
            return a if row == "model" or prov["ensemble_rows"].get(row) else a[0]

        def triple(ch):
            return {"truth": get("truth", ch), "input": get("input", ch), "model": get("model", ch)}

        for v in variables:
            if v == "10ff":
                if "10u" in ws and "10v" in ws:
                    u, w = triple("10u"), triple("10v")
                    out[v] = {k: np.hypot(u[k], w[k]) for k in u}
            elif v in ws:
                out[v] = triple(v)
                if v == "tp":
                    out[v] = {k: np.clip(a, 0.0, None) * MM for k, a in out[v].items()}
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    g = prov["input_grid"]
    if g["lon"] is not None:
        g["lon"] = np.where(g["lon"] > 180.0, g["lon"] - 360.0, g["lon"])
    return lat, lon, out, prov


def input_height_column(lat, lon) -> str:
    """Cache column of the model height at the station for the native input grid: ``h_o1280`` when the input
    grid's point spacing is near O1280's (about 9 km), ``h_o320`` near O320's (about 35 km)."""
    from scipy.spatial import cKDTree

    k = min(2000, lat.size)
    sel = np.linspace(0, lat.size - 1, k).astype(int)
    d, _ = cKDTree(xyz(lat, lon)).query(xyz(lat[sel], lon[sel]), k=2)
    km = float(np.median(d[:, 1])) * 6371.0
    return "h_o1280" if km < 20.0 else "h_o320"


def row_kinds(prov: dict) -> dict:
    """Row name -> kind written into the tables, for one file's provenance."""
    kinds = dict(ROW_KIND)
    for row, is_ens in prov.get("ensemble_rows", {}).items():
        if is_ens:
            kinds[row] = ENSEMBLE_KIND[row]
    return kinds


def _hres_at(hres, date, lead, o):
    """HRES value and HRES model height at the sample's stations, or (None, None) when STVL had nothing."""
    if hres is None:
        return None, None
    h = hres[(hres.date == date) & (hres.lead.astype(int) == int(lead))]
    if h.empty:
        return None, None
    h = h.drop_duplicates(subset=["stnid"]).set_index("stnid")
    return (o.stnid.map(h.hres).to_numpy(np.float64),
            o.stnid.map(h.hres_elevation).to_numpy(np.float64) if "hres_elevation" in h else None)


def hres_tp_in_metres(hr, ov) -> bool:
    """The A0 audit, decided once per run on pooled values: HRES tp from STVL looks like metres when its maximum
    is below 1 and its mean is below 1 % of the mean of the wet observations (> 1 mm). False when there is no
    wet observation or no HRES value (nothing to decide on)."""
    hr, ov = np.asarray(hr, np.float64), np.asarray(ov, np.float64)
    wet = ov[np.isfinite(ov) & (ov > 1.0)]
    if not np.isfinite(hr).any() or not wet.size:
        return False
    return bool(np.nanmax(hr) < 1.0 and np.nanmean(hr) < 0.01 * wet.mean())


def tp_window_check(fields, obs_v, matcher, *, date, lead, cfg, log=None, input_matcher=None):
    """The A0 accumulation-window audit, at the nearest grid point: correlation of the truth, the input and the
    ensemble mean with the station tp of the 12 h window ending at valid time (corr_valid) and of the window
    before (corr_prev, observations valid 12 h earlier), on the stations that report both windows (joined on
    station id, so both correlations use the same stations and the same forecast values). The file's tp must
    match the window ending at valid time, so corr_valid should be the larger; prev_better flags the dates
    where it is not. Skipped with a log line when a window has no observations, fewer than min_stations
    stations report both, or a series is constant."""
    import pandas as pd

    log = [] if log is None else log
    valid = pd.Timestamp(date) + pd.Timedelta(hours=int(lead))
    prev = valid - pd.Timedelta(hours=12)
    o = obs_v[obs_v.valid == valid.strftime("%Y%m%d%H")]
    op = obs_v[obs_v.valid == prev.strftime("%Y%m%d%H")]
    if o.empty or op.empty:
        log.append(f"window audit skipped for {date} +{lead} h: no station tp at "
                   f"{(valid if o.empty else prev):%Y%m%d%H}")
        return []
    j = o[["stnid", "lat", "lon", "obs"]].drop_duplicates(subset=["stnid"]).merge(
        op[["stnid", "obs"]].drop_duplicates(subset=["stnid"]), on="stnid", suffixes=("", "_prev"))
    idx, keep = matcher.match(j.lat.to_numpy(), j.lon.to_numpy(), cfg["max_km"])
    j, idx = j[keep], idx[keep]
    if len(j) < cfg["min_stations"]:
        log.append(f"window audit skipped for {date} +{lead} h: {len(j)} stations report both windows")
        return []
    ov, opv = j.obs.to_numpy(np.float64), j.obs_prev.to_numpy(np.float64)
    idx_in = idx if input_matcher is None else input_matcher.match(j.lat.to_numpy(), j.lon.to_numpy(), np.inf)[0]
    out = []
    for row, f in (("truth", fields["truth"]), ("input", fields["input"]), ("model", fields["model"])):
        x = (f.mean(0) if f.ndim == 2 else f)[idx_in if row == "input" else idx]
        if x.std() == 0 or ov.std() == 0 or opv.std() == 0:
            log.append(f"window audit skipped for {date} +{lead} h {row}: a constant series")
            continue
        cv, cp = float(np.corrcoef(x, ov)[0, 1]), float(np.corrcoef(x, opv)[0, 1])
        out.append({"lead": int(lead), "date": str(date), "row": row, "corr_valid": cv, "corr_prev": cp,
                    "prev_better": bool(cp > cv), "n": int(ov.size)})
    return out


def score_valid(var, fields, obs_v, hres_v, matcher, *, date, lead, cfg, seeps_clim=None, hres_pool=None,
                kinds=None, input_matcher=None, input_h="h_o320"):
    """Rows of one variable, one date and one lead. The station sample is the intersection over rows
    (model, input, truth and, where STVL has it, HRES), so every row is scored on the same stations.
    ``kinds`` (row -> kind, ``row_kinds``) labels each row; the input and truth rows are ensembles (M, n)
    when the file carried their members, else single fields (n,).
    With ``seeps_clim`` (24 h tp) the sample is restricted to stations with a SEEPS climatology whose dry-day
    probability is in [0.1, 0.85) (quaver's seeps_filter), and every row gets a ``seeps`` column: SEEPS of
    each member at each station, averaged over members and stations."""
    import pandas as pd

    kinds = kinds or ROW_KIND
    valid = (pd.Timestamp(date) + pd.Timedelta(hours=int(lead))).strftime("%Y%m%d%H")
    o = obs_v[obs_v.valid == valid]
    clim = None
    if seeps_clim is not None:
        c = seeps_clim[seeps_clim.valid == valid].drop_duplicates(subset=["stnid"]).set_index("stnid")
        p1 = o.stnid.map(c.p1)
        o = o[(p1 >= 0.1) & (p1 < 0.85)]
    if o.empty:
        return [], [], 0
    idx, keep = matcher.match(o.lat.to_numpy(), o.lon.to_numpy(), cfg["max_km"])
    o, idx = o[keep].reset_index(drop=True), idx[keep]
    # the input's own nearest native point (rule 41); same stations as every other row
    idx_in = idx if input_matcher is None else input_matcher.match(o.lat.to_numpy(), o.lon.to_numpy(), np.inf)[0]
    if seeps_clim is not None:
        cc = c.loc[o.stnid]
        clim = (cc.t1.to_numpy(), cc.t2.to_numpy(), cc[[f"s{k}" for k in range(9)]].to_numpy())
    ov = o.obs.to_numpy(np.float64)
    hr, hel = _hres_at(hres_v, date, lead, o)
    if var.startswith("tp") and hr is not None and hres_pool is not None:
        hres_pool.setdefault(var, []).append((hr, ov))  # the metres audit is decided once per run
    is_tp = var == "tp"
    thresholds = cfg["thresholds"] if is_tp else ()
    radii = cfg["radii"] if is_tp else (0.0,)
    elev = o.elevation.to_numpy(np.float64)
    dz_abs = np.abs(o.h_o1280.to_numpy(np.float64) - elev)
    rows, ranks = [], []
    for r in radii:
        ball = None if r == 0 else matcher.discs(idx, r)
        ball_in = ball if input_matcher is None or r == 0 else input_matcher.discs(idx_in, r)

        def at(a, ix=idx, bl=ball):
            if bl is None:
                return a[..., ix]
            if a.ndim == 1:
                return np.array([a[b].mean() for b in bl])
            return np.stack([a[:, b].mean(1) for b in bl], 1)

        vals = {"model": at(fields["model"]), "input": at(fields["input"], idx_in, ball_in),
                "truth": at(fields["truth"])}
        if hr is not None and r == 0:
            vals["hres"] = hr
        variants = [("raw", vals)]
        if var == "2t" and r == 0 and cfg["lapse_rate"]:
            dz = {"model": o.h_o1280.to_numpy() - elev, "truth": o.h_o1280.to_numpy() - elev,
                  "input": o[input_h].to_numpy() - elev}
            if "hres" in vals:
                dz["hres"] = (hel if hel is not None else np.full(elev.size, np.nan)) - elev
            variants.append(("lapse", {k: x + cfg["lapse_rate"] * np.asarray(dz[k], np.float64) for k, x in vals.items()}))
        for tag, vv in variants:
            ok = np.isfinite(ov)
            for x in vv.values():
                ok &= np.isfinite(x).all(0) if x.ndim == 2 else np.isfinite(x)
            masks = region_masks(o.lat.to_numpy(), o.lon.to_numpy(), cfg["regions"])
            masks.update(height_masks(dz_abs, cfg["height_classes"]))
            for reg, m in masks.items():
                m = m & ok
                if m.sum() < cfg["min_stations"]:
                    continue
                w = density_weights(o.lat.to_numpy()[m], o.lon.to_numpy()[m]) if cfg["weighting"] == "station_density" else None
                base = dict(var=var, variant=tag, radius_km=float(r), region=reg, lead=int(lead), date=str(date))
                for k, x in vv.items():
                    ens = x[:, m] if x.ndim == 2 else x[m][None]
                    extra = {}
                    if clim is not None:
                        extra["seeps"] = _wmean(seeps(ens, ov[m], clim[0][m], clim[1][m], clim[2][m]).mean(0), w)
                    rows.append({**base, "row": k, "kind": kinds[k],
                                 **ensemble_scores(ens, ov[m], thresholds, weights=w), **extra})
                    if ens.shape[0] > 1:
                        counts = rank_counts(ens, ov[m], seed=rank_seed(base, k))
                        ranks.append({**base, "row": k, "members": int(ens.shape[0]), "n": int(m.sum()),
                                      **{f"r{i}": int(c) for i, c in enumerate(counts)}})
    return rows, ranks, int(ov.size)


def resolve_config(eval_config: dict) -> dict:
    regions = eval_config.get("regions")
    weighting = str(eval_config.get("weighting", "none") or "none")
    if weighting not in ("none", "station_density"):
        raise ValueError(f"obs_points: weighting must be 'none' or 'station_density', not {weighting!r}")
    return {
        "vars": [str(v) for v in eval_config.get("vars", DEFAULT_VARS)],
        "regions": dict(DEFAULT_REGIONS if regions is None else regions),
        "height_classes": dict(eval_config.get("height_classes", DEFAULT_HEIGHT_CLASSES) or {}),
        "thresholds": tuple(float(t) for t in eval_config.get("thresholds", DEFAULT_THRESHOLDS)),
        "radii": tuple(float(r) for r in eval_config.get("radii", DEFAULT_RADII)),
        "max_km": float(eval_config.get("max_km", 15.0)),
        "lapse_rate": float(eval_config.get("lapse_rate", LAPSE_RATE)),
        "min_stations": int(eval_config.get("min_stations", 30)),
        "weighting": weighting,
        "competitor_ensembles": bool(eval_config.get("competitor_ensembles", True)),
        # rule 41: score the input on its native grid (x, lat_lres/lon_lres); False only to reproduce old tables
        "native_input": bool(eval_config.get("native_input", True)),
        "seeps": bool(eval_config.get("seeps", True)),
        "dates": [str(d) for d in eval_config.get("dates", [])],
        "steps": [int(s) for s in eval_config.get("steps", [])],
    }


def run(predictions_dir, lane_config: dict, eval_config: dict, *, output_dir=None, overwrite: bool = False,
        checkpoint=None, run_label: str = "", **kwargs) -> Path:
    import pandas as pd

    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "obs_points"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"obs_points output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = resolve_config(eval_config)
    cache_dir = resolve_cache(eval_config.get("cache"))
    obs, hres, cache_meta = load_cache(cache_dir, cfg["vars"] + (["tp24"] if cfg["seeps"] else []))
    seeps_clim = load_seeps_clim(cache_dir) if cfg["seeps"] and "tp24" in obs and "tp" in cfg["vars"] else None
    if cfg["seeps"] and seeps_clim is None:
        LOG.info("obs_points: no 24 h tp or SEEPS climatology in the cache; SEEPS skipped")
    missing_vars = [v for v in cfg["vars"] if v not in obs]
    if missing_vars:
        LOG.warning("obs_points: no observations in %s for %s", cache_dir, missing_vars)
    preds = [p for p in find_predictions(predictions_dir)
             if (not cfg["dates"] or p.date in cfg["dates"]) and (not cfg["steps"] or p.step in cfg["steps"])]
    if not preds:
        raise FileNotFoundError(f"No prediction files to read in {predictions_dir}")

    rows, ranks, log, files, window, hres_pool = [], [], [], [], [], {}
    members = set()
    stations = defaultdict(list)
    matcher = input_matcher = None
    input_h, input_sources = "h_o320", set()
    constructions, kinds_seen = set(), {}
    prev = {}  # tp of the previous lead of the same date, for the 24 h windows of SEEPS
    for p in preds:
        lat, lon, fields, prov = read_fields(p.path, [v for v in cfg["vars"] if v in obs],
                                             ensembles=cfg["competitor_ensembles"], native_input=cfg["native_input"])
        g = prov["input_grid"]
        input_sources.add(g["source"])
        if g["lat"] is None:
            input_matcher, input_h = None, "h_o320"
        elif input_matcher is None or not input_matcher.same_grid(g["lat"], g["lon"]):
            input_matcher = StationMatcher(g["lat"], g["lon"])
            input_h = input_height_column(g["lat"], g["lon"])
        kinds = row_kinds(prov)
        constructions.add(prov["model"])
        for row, kind in kinds.items():
            kinds_seen.setdefault(row, set()).add(kind)
        if matcher is None or not matcher.same_grid(lat, lon):
            matcher = StationMatcher(lat, lon)
        for v, f in fields.items():
            members.add(int(f["model"].shape[0]))
            r, k, nst = score_valid(v, f, obs[v], hres.get(v), matcher, date=p.date, lead=p.step, cfg=cfg,
                                    hres_pool=hres_pool, kinds=kinds, input_matcher=input_matcher, input_h=input_h)
            if v == "tp":
                window += tp_window_check(f, obs[v], matcher, date=p.date, lead=p.step, cfg=cfg, log=log,
                                          input_matcher=input_matcher)
            rows += r
            ranks += k
            stations[f"{v} +{p.step:03d}"].append(nst)
            if nst == 0:
                log.append(f"WARN no observations within {cfg['max_km']:g} km for {v} {p.date} +{p.step} h")
        if seeps_clim is not None and "tp" in fields:
            before = prev.get((p.date, p.step - 12))
            last = next(iter(prev), None)
            if last is not None and last[0] == p.date and p.step - last[1] != 12:
                log.append(f"WARN {p.date}: leads +{last[1]} and +{p.step} h are not 12 h apart; 24 h SEEPS windows "
                           "need consecutive 12 h files")
            if before is None and p.step >= 24:
                log.append(f"WARN {p.date}: no +{p.step - 12} h file, so no 24 h SEEPS window ending at +{p.step} h")
            if before is not None:
                f24 = {k: fields["tp"][k] + before[k] for k in fields["tp"]}
                r, k, nst = score_valid("tp24", f24, obs["tp24"], None, matcher, date=p.date, lead=p.step,
                                        cfg=cfg, seeps_clim=seeps_clim, kinds=kinds, input_matcher=input_matcher,
                                        input_h=input_h)
                rows += r
                ranks += k
                stations[f"tp24 +{p.step:03d}"].append(nst)
            prev = {(p.date, p.step): fields["tp"]}
        absent = [v for v in cfg["vars"] if v in obs and v not in fields]
        if absent:
            log.append(f"WARN {p.path.name}: no channel for {absent}")
        files.append(str(p.path))
        LOG.info("obs_points: scored %s +%03d h", p.date, p.step)
    if not rows:
        raise RuntimeError("obs_points: no station sample reached min_stations; nothing scored")
    for src_name in sorted(input_sources):
        if "interpolated" in src_name:
            log.append(f"WARN input row = {src_name}: rule 41 says the input is scored on its own grid; this "
                       "file set carries no native x, so the input row is smoothed and its scores are not quotable "
                       "against the model")
    if len(constructions) > 1:
        log.append(f"WARN the files disagree on how the model ensemble was built: {sorted(constructions)}")
    construction = sorted(constructions)[0]
    if is_single_input(construction):
        log.append(f"WARN model row = {construction}. {ENSEMBLE_RULE}")
    for row, ks in kinds_seen.items():
        if len(ks) > 1:
            log.append(f"WARN row {row} is an ensemble in some files and a single field in others: {sorted(ks)}")
    row_kind = {row: sorted(ks)[0] for row, ks in kinds_seen.items()}
    competitor = {row: (ENSEMBLE_KIND[row] if row_kind[row].startswith("ensemble") else SINGLE_FIELD)
                  for row in ("input", "truth") if row in row_kind}
    if "hres" in set(r["row"] for r in rows):
        competitor["hres"] = SINGLE_FIELD
    for row in ("input", "truth"):
        if competitor.get(row, "").startswith("single field"):
            log.append(f"WARN the {row} row is a single field (the files carry no {row} members): its fair CRPS is "
                       "the MAE of one field; compare it with the model's ensemble mean only (rule 34(e))")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "scores_per_date.csv", index=False)
    for v, pairs in hres_pool.items():
        hr, ov = np.concatenate([a for a, _ in pairs]), np.concatenate([b for _, b in pairs])
        if hres_tp_in_metres(hr, ov):
            log.append(f"WARN HRES {v} looks like metres over the run (max {np.nanmax(hr):.3g}, mean {np.nanmean(hr):.3g}; "
                       f"mean of the observations above 1 mm {ov[ov > 1].mean():.3g} mm)")
    if window:
        w = pd.DataFrame(window)
        w.to_csv(output_dir / "tp_window_check.csv", index=False)
        ws = w.groupby(["lead", "row"]).agg(corr_valid=("corr_valid", "mean"), corr_prev=("corr_prev", "mean"),
                                            dates=("corr_valid", "count"), dates_prev_better=("prev_better", "sum"),
                                            mean_n=("n", "mean")).reset_index()
        ws.to_csv(output_dir / "tp_window_summary.csv", index=False)
        for r in ws[ws.corr_prev > ws.corr_valid].itertuples():
            log.append(f"WARN tp window: at +{r.lead} h the {r.row} correlates better with the 12 h window before "
                       f"valid time ({r.corr_prev:.3f}) than with the window ending at it ({r.corr_valid:.3f})")
    pd.DataFrame(ranks).to_csv(output_dir / "rank_hist.csv", index=False)
    summ = summarise(df)
    summ.to_csv(output_dir / "summary.csv", index=False)
    pr = paired(df)
    pr.to_csv(output_dir / "paired.csv", index=False)
    meta = {
        "evaluator": "obs_points", "predictions_dir": str(predictions_dir),
        "run_label": str(run_label or eval_config.get("run_label") or predictions_dir.name),
        "cache_dir": str(cache_dir), "cache_meta": cache_meta,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "members": sorted(members), "dates": sorted({p.date for p in preds}), "steps": sorted({p.step for p in preds}),
        "files": files, "stations_per_file": {k: v for k, v in sorted(stations.items())},
        "rows": {k: row_kind.get(k, ROW_KIND.get(k, "")) for k in sorted(set(df.row))},
        "input_source": sorted(input_sources), "input_height_column": input_h,
        "provenance": provenance(
            verifying_field=f"station observations (SYNOP / STVL cache {cache_dir.name}; nearest output-grid point "
                            f"within {cfg['max_km']:g} km for model and truth, nearest point of the input's own grid "
                            f"for the input ({'; '.join(sorted(input_sources))}); 2t lapse {cfg['lapse_rate']:g} K/m "
                            "on the 'lapse' variant)",
            against_observations=True,
            ensemble=construction, competitor_rows=competitor,
            weight_form=("unweighted mean over stations" if cfg["weighting"] == "none"
                         else "quaver station-density weights (weighted mean over stations)"),
            spread_form="spread = mean member SD (ddof 1); spread_rms = root of the mean member variance; "
                        "ssr_corrected = spread_rms / rmse_mean * sqrt((M+1)/M); SE = over dates",
            domain_bounds={**cfg["regions"], **{k: f"|h_o1280 - h_station| in {v} m"
                                                for k, v in cfg["height_classes"].items()}},
            dates=sorted({p.date for p in preds}),
            leads=sorted({p.step for p in preds}), members=sorted(members),
        ),
        "warnings": log,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    (output_dir / "obs_points_log.txt").write_text("\n".join(log) + "\n")
    write_summary_md(summ, pr, meta, output_dir / "summary.md")
    for line in log:
        LOG.warning("obs_points: %s", line)
    return output_dir
