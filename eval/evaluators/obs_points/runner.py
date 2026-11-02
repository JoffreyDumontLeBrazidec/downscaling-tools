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
    StationMatcher, density_weights, height_masks, load_cache, load_seeps_clim, region_masks, resolve_cache,
)
from eval.evaluators.obs_points.core.summary import paired, summarise, write_summary_md

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

ROW_KIND = {"model": "ensemble", "input": "single field", "truth": "single field", "hres": "single field"}


def read_fields(path, variables):
    """lat, lon (-180..180), {var: {"truth": (n,), "input": (n,), "model": (M, n)}} in observation units:
    2t K, 10ff m/s (from 10u/10v), msl Pa, tp mm (file metres x 1000, clipped at 0). Truth and input take
    member 0 when they carry a member axis."""
    import netCDF4

    out = {}
    with netCDF4.Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        ws = [str(s) for s in ds["weather_state"][:]]
        lat = np.asarray(ds["lat_hres"][:], np.float64)
        lon = np.asarray(ds["lon_hres"][:], np.float64)
        n = lat.size

        def get(name, ch):
            a = np.asarray(ds[name][0, ..., ws.index(ch)], np.float64).reshape(-1, n)
            return a if name == "y_pred" else a[0]

        def triple(ch):
            return {"truth": get("y", ch), "input": get("x_interp", ch), "model": get("y_pred", ch)}

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
    return lat, lon, out


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


def score_valid(var, fields, obs_v, hres_v, matcher, *, date, lead, cfg, seeps_clim=None):
    """Rows of one variable, one date and one lead. The station sample is the intersection over rows
    (model, input, truth and, where STVL has it, HRES), so every row is scored on the same stations.
    With ``seeps_clim`` (24 h tp) the sample is restricted to stations with a SEEPS climatology whose dry-day
    probability is in [0.1, 0.85) (quaver's seeps_filter), and every row gets a ``seeps`` column: SEEPS of
    each member at each station, averaged over members and stations."""
    import pandas as pd

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
    if seeps_clim is not None:
        cc = c.loc[o.stnid]
        clim = (cc.t1.to_numpy(), cc.t2.to_numpy(), cc[[f"s{k}" for k in range(9)]].to_numpy())
    ov = o.obs.to_numpy(np.float64)
    hr, hel = _hres_at(hres_v, date, lead, o)
    is_tp = var == "tp"
    thresholds = cfg["thresholds"] if is_tp else ()
    radii = cfg["radii"] if is_tp else (0.0,)
    elev = o.elevation.to_numpy(np.float64)
    dz_abs = np.abs(o.h_o1280.to_numpy(np.float64) - elev)
    rows, ranks = [], []
    for r in radii:
        ball = None if r == 0 else matcher.discs(idx, r)

        def at(a):
            if ball is None:
                return a[..., idx]
            if a.ndim == 1:
                return np.array([a[b].mean() for b in ball])
            return np.stack([a[:, b].mean(1) for b in ball], 1)

        vals = {k: at(fields[k]) for k in ("model", "input", "truth")}
        if hr is not None and r == 0:
            vals["hres"] = hr
        variants = [("raw", vals)]
        if var == "2t" and r == 0 and cfg["lapse_rate"]:
            dz = {"model": o.h_o1280.to_numpy() - elev, "truth": o.h_o1280.to_numpy() - elev,
                  "input": o.h_o320.to_numpy() - elev}
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
                    rows.append({**base, "row": k, "kind": ROW_KIND[k],
                                 **ensemble_scores(ens, ov[m], thresholds, weights=w), **extra})
                    if ens.shape[0] > 1:
                        counts = rank_counts(ens, ov[m])
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

    rows, ranks, log, files = [], [], [], []
    members = set()
    stations = defaultdict(list)
    matcher = None
    prev = {}  # tp of the previous lead of the same date, for the 24 h windows of SEEPS
    for p in preds:
        lat, lon, fields = read_fields(p.path, [v for v in cfg["vars"] if v in obs])
        if matcher is None or not matcher.same_grid(lat, lon):
            matcher = StationMatcher(lat, lon)
        for v, f in fields.items():
            members.add(int(f["model"].shape[0]))
            r, k, nst = score_valid(v, f, obs[v], hres.get(v), matcher, date=p.date, lead=p.step, cfg=cfg)
            rows += r
            ranks += k
            stations[f"{v} +{p.step:03d}"].append(nst)
            if nst == 0:
                log.append(f"WARN no observations within {cfg['max_km']:g} km for {v} {p.date} +{p.step} h")
        if seeps_clim is not None and "tp" in fields:
            before = prev.get((p.date, p.step - 12))
            if before is not None:
                f24 = {k: fields["tp"][k] + before[k] for k in fields["tp"]}
                r, k, nst = score_valid("tp24", f24, obs["tp24"], None, matcher, date=p.date, lead=p.step,
                                        cfg=cfg, seeps_clim=seeps_clim)
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

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "scores_per_date.csv", index=False)
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
        "rows": {k: ROW_KIND[k] for k in sorted(set(df.row))},
        "warnings": log,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    (output_dir / "obs_points_log.txt").write_text("\n".join(log) + "\n")
    write_summary_md(summ, pr, meta, output_dir / "summary.md")
    for line in log:
        LOG.warning("obs_points: %s", line)
    return output_dir
