"""tc_obs run(): per storm case, the model members, truth and input (from tc_structure's cases.csv) and the
operational HRES and ENS tracks, against the IBTrACS best track at valid time. Tables only, no figures."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from eval.evaluators.obs_points.core.scores import fair_crps
from eval.evaluators.obs_points.core.stations import resolve_cache
from eval.evaluators.tc_obs.core.reference import HRES_MEMBER, best_track_at, load_cache, read_noaa_csv

LOG = logging.getLogger(__name__)

R_EARTH_KM = 6371.0
ROW_KIND = {"model": "ensemble", "ens": "ensemble", "truth": "single field", "input": "single field",
            "hres": "single field"}
SCORES = ("pmin_crps", "pmin_err", "pmin_mae", "pos_err_member_km", "pos_err_mean_km", "pos_spread_km")
VERDICT_NOTE = "These are {n} storm cases on {d} dates: cases, not a verdict."


def gc_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (np.asarray(lat1, float), np.asarray(lon1, float),
                                          np.asarray(lat2, float), np.asarray(lon2, float)))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def mean_centre(lat, lon):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    x, y, z = (np.cos(la) * np.cos(lo)).mean(), (np.cos(la) * np.sin(lo)).mean(), np.sin(la).mean()
    return float(np.degrees(np.arctan2(z, np.hypot(x, y)))), float(np.degrees(np.arctan2(y, x)))


def _scalar_scores(x, ref, name) -> dict:
    """fair CRPS (absolute error for one value) and mean error of the finite values of x against ref."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not x.size or not np.isfinite(ref):
        return {}
    return {f"{name}_crps": float(fair_crps(x[:, None], np.array([ref]))[0]), f"{name}_err": float(x.mean() - ref),
            f"{name}_members": int(x.size)}


def case_scores(pmin, lat, lon, bt, wind=None, rmw=None) -> dict:
    """Scores of one row (an ensemble of M centres, or one centre) against the best track of one case.
    pmin_crps is the fair CRPS of the members' central pressure (the absolute error for one value);
    pos_err_member_km the mean distance of the members' centres from the best track; pos_err_mean_km the
    distance of the mean centre; pos_spread_km the root-mean-square distance of the members from their
    mean centre (0 for one value)."""
    pmin, lat, lon = (np.asarray(a, np.float64) for a in (pmin, lat, lon))
    m = pmin.size
    clat, clon = mean_centre(lat, lon)
    d = gc_km(lat, lon, bt["lat"], bt["lon"])
    out = {"n_found": int(m), "pmin_mean": float(pmin.mean()),
           "pmin_crps": float(fair_crps(pmin[:, None], np.array([bt["pmin_hpa"]]))[0]),
           "pmin_err": float(pmin.mean() - bt["pmin_hpa"]), "pmin_mae": float(np.abs(pmin - bt["pmin_hpa"]).mean()),
           "pos_err_member_km": float(d.mean()), "pos_err_mean_km": float(gc_km(clat, clon, bt["lat"], bt["lon"])),
           "pos_spread_km": float(np.sqrt((gc_km(lat, lon, clat, clon) ** 2).mean())) if m > 1 else 0.0}
    if wind is not None:
        wind = np.asarray(wind, np.float64)
        out["wind_mean"] = float(np.nanmean(wind)) if np.isfinite(wind).any() else np.nan
        out.update(_scalar_scores(wind, bt.get("wind_ms", np.nan), "wind"))
    if rmw is not None:
        rmw = np.asarray(rmw, np.float64)
        out["rmw_mean"] = float(np.nanmean(rmw)) if np.isfinite(rmw).any() else np.nan
        out.update(_scalar_scores(rmw, bt.get("rmw_km", np.nan), "rmw"))
    return out


def _single(rows, log, tag):
    """One value from tc_structure's per-member copies of a single field (truth, input)."""
    rows = rows[(rows.found == 1) & np.isfinite(rows.pmin_hpa) & np.isfinite(rows.centre_lat)]
    if rows.empty:
        return None
    if rows.pmin_hpa.max() - rows.pmin_hpa.min() > 0.01:
        log.append(f"WARN {tag}: the copies differ ({rows.pmin_hpa.min():.2f}-{rows.pmin_hpa.max():.2f} hPa); "
                   "the first is used")
    return rows.iloc[:1]


def score_cases(cases, bt, trackers, *, only_storm_ok=True, max_gap_h=6.0, wind_col="maxwind300_ms"):
    """(case rows, member rows, log) for every storm case of a tc_structure cases.csv frame."""
    import pandas as pd

    log, out, members = [], [], []
    c = cases.copy()
    c["date"] = c["date"].astype(str)
    keys = ["event", "date", "lead"]
    ok_cases = c[c.storm_ok == 1][keys].drop_duplicates() if only_storm_ok else c[keys].drop_duplicates()
    for _, k in ok_cases.sort_values(keys).iterrows():
        event, date, lead = str(k.event), str(k.date), int(k.lead)
        valid = pd.Timestamp(date) + pd.Timedelta(hours=lead)
        b = best_track_at(bt, event, valid.to_pydatetime(), max_gap_h=max_gap_h)
        if b is not None and b.get("noaa_incomplete"):
            log.append(f"WARN {event} {date} +{lead} h: the NOAA best track has no pressure or position near "
                       f"{valid:%Y%m%d%H}; the anemoi zarr fix is used")
        tag = f"{event} {date} +{lead} h"
        if b is None:
            log.append(f"WARN {tag}: no best-track fix within {max_gap_h:g} h of {valid:%Y%m%d%H}; case skipped")
            continue
        base = {"event": event, "date": date, "lead": lead, "valid": valid.strftime("%Y%m%d%H"),
                "bt_lat": b["lat"], "bt_lon": b["lon"], "bt_pmin_hpa": b["pmin_hpa"], "bt_wind_ms": b["wind_ms"],
                "bt_interpolated": bool(b["interpolated"]), "bt_wind_src": b.get("wind_src", ""),
                "bt_rmw_km": b.get("rmw_km", np.nan), "bt_source": b.get("source")}
        sub = c[(c.event == event) & (c.date == date) & (c.lead == lead)]
        rows = {}
        mod = sub[(sub.field == "model") & (sub.found == 1) & np.isfinite(sub.pmin_hpa) & np.isfinite(sub.centre_lat)]
        if len(mod):
            rows["model"] = (mod.pmin_hpa, mod.centre_lat, mod.centre_lon,
                             mod[wind_col] if wind_col in mod else None, mod.member, int((sub.field == "model").sum()),
                             mod["rmw_km"] if "rmw_km" in mod else None)
        for f in ("truth", "input"):
            s = _single(sub[sub.field == f], log, f"{tag} {f}")
            if s is not None:
                rows[f] = (s.pmin_hpa, s.centre_lat, s.centre_lon, s[wind_col] if wind_col in s else None, None, 1,
                           s["rmw_km"] if "rmw_km" in s else None)
        for stream, row in (("oper", "hres"), ("enfo", "ens")):
            tr = trackers.get((stream, date))
            if tr is None:
                log.append(f"WARN {tag}: no {stream} tracker file for {date}")
                continue
            storm = tr[tr.name.astype(str).str.upper() == event.upper()]
            if row == "hres":
                # HRES is member 52; a message without a member number decodes as -1: take the single member then
                present = set(storm.member)
                keep = {HRES_MEMBER} if HRES_MEMBER in present else (present if len(present) == 1 else set())
                sel = storm.member.isin(keep)
            else:
                sel = (storm.member >= 1) & (storm.member <= 51)
            t = storm[sel & (storm.step_h == lead)].dropna(subset=["lat", "lon", "msl_hpa"])
            if t.empty:
                log.append(f"WARN {tag}: no {row.upper()} row in the {stream} tracker for {event.upper()} at +{lead} h")
                continue
            # members_total: HRES 1; ENS = the members 1-51 present in this date's file (not only those that
            # tracked this storm), so found_frac counts a member that lost the storm as not found
            n_tot = (int(storm[sel].member.nunique()) if row == "hres" else
                     int(tr[(tr.member >= 1) & (tr.member <= 51)].member.nunique()))
            rows[row] = (t.msl_hpa, t.lat, t.lon, t.wind_ms, t.member, n_tot, None)
        for row, (p, la, lo, w, mem, n_tot, rmw) in rows.items():
            sc = case_scores(p, la, lo, b, w, rmw)
            out.append({**base, "row": row, "kind": ROW_KIND[row], "members_total": n_tot,
                        "found_frac": sc["n_found"] / n_tot if n_tot else np.nan, **sc})
            if mem is not None:
                d = gc_km(la, lo, b["lat"], b["lon"])
                for mm, pp, dd in zip(np.asarray(mem), np.asarray(p, float), np.asarray(d)):
                    members.append({**base, "row": row, "member": int(mm), "pmin_hpa": pp,
                                    "pmin_err": pp - b["pmin_hpa"], "pos_err_km": dd})
    return pd.DataFrame(out), pd.DataFrame(members), log


def summarise(df):
    """Mean and SE over cases per (lead, row), and pooled over leads (lead = 'all')."""
    import pandas as pd

    cols = [c for c in SCORES + ("wind_crps", "wind_err", "wind_mean", "rmw_crps", "rmw_err", "rmw_mean", "pmin_mean",
                                 "n_found", "found_frac") if c in df.columns]

    def agg(g):
        s = g[cols].mean().join(g[cols].sem().add_suffix("_se")).join(g[cols].count().add_suffix("_cases"))
        return s.join(g.size().rename("cases")).join(g.date.nunique().rename("dates"))

    by = agg(df.groupby(["lead", "row", "kind"])).reset_index()
    pooled = agg(df.groupby(["row", "kind"])).reset_index()
    pooled.insert(0, "lead", "all")
    return pd.concat([by.astype({"lead": str}), pooled], ignore_index=True)


def paired(df, a="model", others=("truth", "input", "hres", "ens"),
           metrics=("pmin_crps", "pos_err_member_km", "wind_crps", "rmw_crps")):
    """Per case a - b, then mean, SE over cases, wins of a (lower is better), by lead and pooled."""
    import pandas as pd

    metrics = [m for m in metrics if m in df.columns and df[m].notna().any()]
    w = df.pivot_table(index=["event", "date", "lead"], columns="row", values=metrics)
    out = []
    for b in others:
        for met in metrics:
            if (met, a) not in w.columns or (met, b) not in w.columns:
                continue
            d = (w[(met, a)] - w[(met, b)]).dropna()
            if d.empty:
                continue
            for lead, dd in list(d.groupby(level="lead")) + [("all", d)]:
                out.append({"lead": str(lead), "pair": f"{a}-{b}", "metric": met, "mean": dd.mean(),
                            "se": dd.sem() if dd.size > 1 else np.nan, "cases": int(dd.size),
                            "dates": int(dd.index.get_level_values("date").nunique()), "wins": int((dd < 0).sum())})
    return pd.DataFrame(out)


def _first_file(spec) -> Path:
    """A file path, or the first existing of a list or mapping of candidates (one per cluster)."""
    cands = list(spec.values()) if isinstance(spec, dict) else ([spec] if isinstance(spec, (str, Path)) else list(spec))
    for c in cands:
        if Path(str(c)).expanduser().is_file():
            return Path(str(c)).expanduser()
    raise FileNotFoundError(f"tc_obs: no NOAA IBTrACS CSV found among {cands}")


def run(predictions_dir, lane_config: dict, eval_config: dict, *, output_dir=None, overwrite: bool = False,
        checkpoint=None, run_label: str = "", **kwargs) -> Path:
    import pandas as pd

    output_dir = Path(output_dir) if output_dir else Path(predictions_dir) / "evaluators" / "tc_obs"
    output_dir.mkdir(parents=True, exist_ok=True)
    label = str(run_label or eval_config.get("run_label") or Path(predictions_dir).name)
    cases_path = eval_config.get("cases")
    cases_path = Path(str(cases_path).format(run_label=label)) if cases_path else \
        output_dir.parent / "tc_structure" / "cases.csv"
    if not cases_path.exists():
        raise FileNotFoundError(f"tc_obs: no tc_structure cases.csv at {cases_path}; run tc_structure first "
                                "or set tc_obs.cases in the lane")
    cache_dir = resolve_cache(eval_config.get("cache"))
    bt, trackers, cache_meta = load_cache(cache_dir)
    cases = pd.read_csv(cases_path, dtype={"date": str})
    noaa_path = None
    if eval_config.get("noaa_csv"):
        # the NOAA best track (with winds and RMW) read directly; it replaces any NOAA rows of the cache
        noaa_path = _first_file(eval_config["noaa_csv"])
        events = sorted(set(cases.event.astype(str)))
        # ISO_TIME window around the cases (the IBTrACS season is not the calendar year everywhere)
        t0 = pd.Timestamp(min(cases.date)) - pd.Timedelta(days=1)
        t1 = pd.Timestamp(max(cases.date)) + pd.Timedelta(hours=int(cases.lead.max()) + 24)
        noaa = read_noaa_csv(noaa_path, events, window=(t0, t1))
        bt = pd.concat([bt[bt.source != "noaa_ibtracs_csv"], noaa], ignore_index=True)
    df, mem, log = score_cases(cases, bt, trackers, only_storm_ok=bool(eval_config.get("only_storm_ok", True)),
                               max_gap_h=float(eval_config.get("max_gap_h", 6.0)),
                               wind_col=str(eval_config.get("wind_col", "maxwind300_ms")))
    if df.empty:
        raise RuntimeError(f"tc_obs: no storm case could be scored ({len(log)} log lines: {log[:3]})")
    df.to_csv(output_dir / "cases_obs.csv", index=False)
    mem.to_csv(output_dir / "members_obs.csv", index=False)
    summ = summarise(df)
    summ.to_csv(output_dir / "summary.csv", index=False)
    pr = paired(df)
    pr.to_csv(output_dir / "paired.csv", index=False)
    n_cases = int(df[["event", "date", "lead"]].drop_duplicates().shape[0])
    n_dates = int(df.date.nunique())
    meta = {"evaluator": "tc_obs", "run_label": label, "cases_csv": str(cases_path), "cache_dir": str(cache_dir),
            "noaa_csv": str(noaa_path) if noaa_path else None,
            "best_track_source": sorted(set(bt.source)) if "source" in bt else None,
            "cache_meta": cache_meta, "n_cases": n_cases, "n_dates": n_dates,
            "events": sorted(set(df.event)), "note": VERDICT_NOTE.format(n=n_cases, d=n_dates), "log": log}
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    write_summary_md(summ, pr, meta, output_dir / "summary.md")
    return output_dir


def write_summary_md(summ, pr, meta, path):
    lead_order = [x for x in sorted({s for s in summ.lead if s != "all"}, key=int)] + ["all"]
    lines = [f"# tc_obs: {meta['run_label']} storms against the IBTrACS best track", "",
             f"**{meta['note']}** Cases from the same date share the weather, so the SE over cases understates "
             "the uncertainty.",
             "`model` and `ens` (operational ENS tracker, 51 members) are ensembles: fair CRPS of their central "
             "pressures, mean member position error, spread of the centres. `truth`, `input` and `hres` "
             "(operational HRES tracker) are single fields: their CRPS is the absolute error and their spread zero.",
             "Model, truth and input centres and pressures are tc_structure's on the O1280 grid; the trackers' are "
             "the operational tracker's on the operational grids, so part of any pressure difference is resolution.",
             f"Events: {', '.join(meta['events'])}.", ""]
    for key, name in (("pmin_crps", "Central pressure: fair CRPS (hPa)"), ("pmin_err", "Central pressure: mean error (hPa)"),
                      ("pos_err_member_km", "Centre position: mean member error (km)"),
                      ("pos_spread_km", "Centre position: ensemble spread (km)"),
                      ("wind_crps", "Maximum 10 m wind: fair CRPS (m/s) against the best-track 10-minute wind"),
                      ("wind_err", "Maximum 10 m wind: mean error (m/s)"),
                      ("rmw_crps", "Radius of maximum wind: fair CRPS (km) against USA_RMW (tc_structure's RMW "
                                   "uses 10 km bins on O1280, while the best-track RMW of intense storms is 9-20 km, "
                                   "so RMW errors partly reflect resolution)"),
                      ("rmw_err", "Radius of maximum wind: mean error (km)")):
        if key not in summ or summ[key].isna().all():
            continue
        lines += [f"## {name}", "", "| row | " + " | ".join(("+" + l + " h") if l != "all" else "all leads"
                                                            for l in lead_order) + " |",
                  "|---|" + "---|" * len(lead_order)]
        for row in ("model", "ens", "truth", "input", "hres"):
            q = summ[summ.row == row].set_index("lead")
            if q.empty or q[key].isna().all() or (key == "pos_spread_km" and row not in ("model", "ens")):
                continue
            cells = [f"{q.loc[l, key]:.1f} ({q.loc[l, key + '_se']:.1f}) n{int(q.loc[l, key + '_cases'])}"
                     if l in q.index and np.isfinite(q.loc[l, key]) else "" for l in lead_order]
            tag = row if ROW_KIND[row] == "ensemble" else f"{row} (single field)"
            lines.append(f"| {tag} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Members that found the storm (fraction of members; the ensemble scores above use only these)", "",
              "| row | " + " | ".join(("+" + l + " h") if l != "all" else "all leads" for l in lead_order) + " |",
              "|---|" + "---|" * len(lead_order)]
    for row in ("model", "ens"):
        q = summ[summ.row == row].set_index("lead")
        if len(q) and "found_frac" in q:
            lines.append(f"| {row} | " + " | ".join(f"{q.loc[l, 'found_frac']:.2f}" if l in q.index else ""
                                                   for l in lead_order) + " |")
    lines.append("")
    lines += ["## Paired, per case (model minus reference; wins = cases where the model is better)", "",
              "| pair | metric | " + " | ".join(("+" + l + " h") if l != "all" else "all leads" for l in lead_order) + " |",
              "|---|---|" + "---|" * len(lead_order)]
    for (pair, met), q in pr.groupby(["pair", "metric"], sort=False):
        q = q.set_index("lead")
        lines.append(f"| {pair} | {met} | " + " | ".join(
            f"{q.loc[l, 'mean']:+.1f} ({q.loc[l, 'se']:.1f}) w{int(q.loc[l, 'wins'])}/{int(q.loc[l, 'cases'])}"
            if l in q.index else "" for l in lead_order) + " |")
    if meta["log"]:
        lines += ["", "## Log", ""] + [f"- {x}" for x in meta["log"]]
    Path(path).write_text("\n".join(lines) + "\n")
