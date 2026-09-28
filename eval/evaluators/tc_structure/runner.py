"""tc_structure evaluator: tropical-cyclone structure per storm, valid time, member and field.

For every prediction file (one initial date, one lead, ten members) and every event
(storm box from ``eval/config/events/*.yaml`` through ``eval._backends.tc.events``)
whose dates match the file, the evaluator measures the storm in three fields on the
same native grid points: the model ``y_pred``, the truth ``y`` and the interpolated
coarse input ``x_interp``. The measurements are defined in ``core.py``.

The truth at member index k is an ENFO member stored there by convention; it is NOT
the truth that model member k should reproduce. The three fields are therefore
treated as three ensembles, compared through means and spreads over members and
valid times, never member against member. For the same reason the first guess that
keeps every centre search on the right storm is taken from the truth ENSEMBLE MEAN
msl (its minimum inside the event box, refined like any centre), and the centre
displacement of every member of every field is its distance to that first guess.

The first guess is followed as a track along the leads of each initial date. At
the first lead with a storm it is the deepest low of the truth ensemble-mean msl in
the event box; at every later lead it is the truth ensemble-mean low within
``track_km_per_24h`` (900 km per 24 h) of the previous lead's first guess. A
(date, lead) is kept as a storm case only if that first guess is a closed low
inside the box: its minimum lies at least ``storm_edge_km`` (50 km) from the box
edge, not on the edge of the track search disc, and its central pressure is at
most ``storm_max_pmin_hpa`` (1005 hPa). Once the track has been lost (the storm left
the box or filled), the later leads of that date are not storm cases; this stops a
second storm (for example the remnant of Idalia in the Franklin box) from being
taken for the first. Cases that are not storm cases are recorded with storm_ok = 0.

Outputs in the results directory:
  cases.csv       one row per (event, date, lead, field, member)
  profiles.npz    the binned tangential-wind profile of every measured case
  summary.json    means over members and valid times per (event, field, lead band)
                  with date-clustered bootstrap standard errors and 95 % intervals,
                  the differences input-truth, model-truth, model-input, and the
                  wind-pressure relation per field
  run_meta.json   parameters, files, git commit
"""
from __future__ import annotations

import csv
import json
import logging
import math
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from eval._backends.tc.data_types import BoundingBox
from eval._backends.tc.events import EVENTS
from eval._backends.tc.grid import normalize_lon, point_mask
from eval._backends.tc.loading_predictions import select_prediction_files_for_event
from eval.discovery.predictions import PREDICTION_RE, find_predictions
from eval.shared.date_bootstrap import (
    DEFAULT_N_BOOT, DEFAULT_SEED, boot_mean, boot_mean_diff, boot_slope,
)

from .core import StructureParams, find_centre, gc_distance_km, measure_structure

LOG = logging.getLogger(__name__)

FIELDS = (("model", "y_pred"), ("truth", "y"), ("input", "x_interp"))
SCORES = ("pmin_hpa", "displacement_km", "rmw_km", "vmax_tan_ms", "maxwind300_ms",
          "r34_km", "r50_km", "zeta50_s", "zeta100_s", "zeta200_s", "asym_rmw")
DEFAULT_BANDS = {
    "24": [24], "48": [48], "72": [72], "96": [96], "120": [120],
    "24-48": [24, 48], "96-120": [96, 120], "all": None,
}
CASE_COLS = ["arm", "event", "date", "lead", "field", "member", "storm_ok", "found", "reason",
             "box_min_hpa", "pmin_hpa", "centre_lat", "centre_lon", "fg_lat", "fg_lon",
             "fg_pmin_hpa", "displacement_km", "rmw_km", "vmax_tan_ms", "vmax_tan_bin_ms",
             "maxwind300_ms", "r34_km", "r34_censored", "r50_km", "r50_censored",
             "zeta50_s", "zeta100_s", "zeta200_s", "asym_rmw", "ring_points", "n_valid_bins",
             "file"]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _discover(predictions_dir: Path):
    files = [(p.path, int(p.date), int(p.step)) for p in find_predictions(predictions_dir)]
    if not files:
        # arm layout <dir>/date_YYYYMMDD/predictions/predictions_*.nc
        for p in sorted(predictions_dir.glob("date_*/predictions/predictions_*.nc")):
            m = PREDICTION_RE.match(p.name)
            if m:
                files.append((p, int(m.group(1)), int(m.group(2))))
    return files


def _select_events(eval_config: dict, files):
    names = eval_config.get("events")
    if names:
        events = [EVENTS[n] for n in names]
    else:
        events = [e for e in EVENTS.values() if e.scoring_eligible]
    return [e for e in events if select_prediction_files_for_event(files, e)]


def _bbox_tuple(ev):
    b = ev.bbox
    w = float(normalize_lon(np.asarray([b.west]))[0])
    e = float(normalize_lon(np.asarray([b.east]))[0])
    return (float(b.south), float(b.north), w, e)


def _read_region(path: Path, events, margin_deg: float):
    """Read 10u, 10v, msl of the three fields on the rows covering every event box
    plus a margin. Returns dict with lat, lon (-180..180), and per field an array
    (member, point, 3) ordered (10u, 10v, msl[hPa])."""
    import netCDF4 as nc

    with nc.Dataset(path) as d:
        ws = [str(x) for x in d["weather_state"][:]]
        iu, iv, ip = ws.index("10u"), ws.index("10v"), ws.index("msl")
        lat = np.asarray(d["lat_hres"][:], dtype=np.float64)
        lon = normalize_lon(np.asarray(d["lon_hres"][:], dtype=np.float64))
        lo_lat = min(e.bbox.south for e in events) - margin_deg
        hi_lat = max(e.bbox.north for e in events) + margin_deg
        rows = np.flatnonzero((lat >= lo_lat) & (lat <= hi_lat))
        if rows.size == 0:
            return None
        i0, i1 = int(rows.min()), int(rows.max()) + 1
        lat_r, lon_r = lat[i0:i1], lon[i0:i1]
        coslat = max(math.cos(math.radians(max(abs(lo_lat), abs(hi_lat)))), 0.1)
        keep = np.zeros(lat_r.size, dtype=bool)
        for e in events:
            s, n, w, ea = _bbox_tuple(e)
            dl = margin_deg / coslat
            inlat = (lat_r >= s - margin_deg) & (lat_r <= n + margin_deg)
            w2 = w - dl
            width = ((ea - w) % 360.0) + 2.0 * dl
            lon_ok = ((lon_r - w2) % 360.0) <= min(width, 360.0)
            keep |= inlat & lon_ok
        sel = np.flatnonzero(keep)
        s0, s1 = min(iu, iv, ip), max(iu, iv, ip) + 1
        out = {"lat": lat_r[sel], "lon": lon_r[sel], "n_file_points": int(lat.size),
               "attrs": {k: str(d.getncattr(k)) for k in d.ncattrs()}}
        for fname, var in FIELDS:
            blk = np.asarray(d[var][0, :, i0:i1, s0:s1], dtype=np.float32)[:, sel, :]
            arr = np.stack([blk[..., iu - s0], blk[..., iv - s0], blk[..., ip - s0]], axis=-1)
            del blk
            arr = arr.astype(np.float64)
            med = float(np.median(arr[0, :, 2]))
            if med > 5000.0:          # Pa -> hPa
                arr[..., 2] /= 100.0
            out[fname] = arr
    return out


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.8g}"
    return str(v)


def measure_file(path: Path, ymd: int, step: int, events, params: StructureParams, *,
                 arm: str, storm_edge_km: float, storm_max_pmin_hpa: float,
                 track_km_per_24h: float = 900.0, track=None, members=None):
    """All case rows (and profiles) of one prediction file.

    ``track`` maps an event name to the first-guess state left by the previous lead
    of the same initial date: None (no storm yet), "lost" (the storm was followed and
    then lost) or (lat, lon, step). Returns (rows, profiles, new track).
    """
    track = dict(track or {})
    margin = params.rmax_km / 111.0 + 0.5
    data = _read_region(path, events, margin)
    rows, profiles = [], []
    if data is None:
        return rows, profiles, track
    lat, lon = data["lat"], data["lon"]
    nmem = data["truth"].shape[0]
    ks = list(range(nmem)) if members is None else list(members)
    for ev in events:
        bbox = _bbox_tuple(ev)
        ev_mask = point_mask(lon, lat, BoundingBox(north=bbox[1], south=bbox[0],
                                                   east=bbox[3], west=bbox[2]))
        if not ev_mask.any():
            LOG.warning("tc_structure: %s has no points in the %s box; skipped", path.name, ev.name)
            continue
        truth_mean = data["truth"][:, :, 2].mean(axis=0)
        fg_params = replace(params, box_edge_km=storm_edge_km)
        state = track.get(ev.name)
        if state == "lost":
            fg = {"found": False, "reason": "storm lost at an earlier lead of this date",
                  "lat": np.nan, "lon": np.nan, "pmin_hpa": np.nan}
        elif state is None:
            # first lead with a storm: the deepest closed low of the truth ensemble mean
            fg = find_centre(lat, lon, truth_mean, ev_mask, bbox=bbox, params=fg_params)
        else:
            # follow the first-guess track: the truth ensemble-mean low within the
            # distance a storm can travel since the previous lead
            plat, plon, pstep = state
            radius = track_km_per_24h * max(int(step) - int(pstep), 1) / 24.0
            fg = find_centre(lat, lon, truth_mean, ev_mask, bbox=bbox, first_guess=(plat, plon),
                             params=replace(fg_params, search_km=radius))
            if not fg["found"] and fg["reason"]:
                fg["reason"] = "first-guess track: " + fg["reason"]
        storm_ok = bool(fg["found"] and fg["pmin_hpa"] <= storm_max_pmin_hpa)
        fg_reason = "" if storm_ok else (fg["reason"] or
                                         f"truth ensemble-mean Pmin {fg['pmin_hpa']:.1f} hPa "
                                         f"> {storm_max_pmin_hpa} hPa")
        if storm_ok:
            track[ev.name] = (fg["lat"], fg["lon"], int(step))
        elif state is not None:
            track[ev.name] = "lost"
        for fname, _ in FIELDS:
            arr = data[fname]
            for k in ks:
                msl = arr[k, :, 2]
                row = {c: "" for c in CASE_COLS}
                row.update(arm=arm, event=ev.name, date=f"{ymd:08d}", lead=int(step), field=fname,
                           member=int(k), storm_ok=int(storm_ok), found=0,
                           box_min_hpa=float(msl[ev_mask].min()),
                           fg_lat=fg["lat"], fg_lon=fg["lon"], fg_pmin_hpa=fg["pmin_hpa"],
                           file=str(path))
                if not storm_ok:
                    row["reason"] = "no storm case: " + fg_reason
                    rows.append(row)
                    continue
                c = find_centre(lat, lon, msl, ev_mask, bbox=bbox,
                                first_guess=(fg["lat"], fg["lon"]), params=params)
                row.update(pmin_hpa=c["pmin_hpa"], centre_lat=c["lat"], centre_lon=c["lon"])
                if not c["found"]:
                    row["reason"] = c["reason"]
                    rows.append(row)
                    continue
                row["displacement_km"] = float(gc_distance_km(fg["lat"], fg["lon"],
                                                              [c["lat"]], [c["lon"]])[0])
                sc, vt = measure_structure(lat, lon, arr[k, :, 0], arr[k, :, 1],
                                           c["lat"], c["lon"], params)
                row.update(found=1, **{kk: sc[kk] for kk in sc if kk in CASE_COLS})
                rows.append(row)
                profiles.append((len(rows) - 1, vt))
    return rows, profiles, track


def _num(rows, key):
    return np.array([float(r[key]) if r[key] not in ("", None) else np.nan for r in rows],
                    dtype=np.float64)


def aggregate(rows: list[dict], bands: dict, n_boot: int, seed: int,
              pressure_ref_hpa: float = 1010.0) -> dict:
    """Means per (event, lead band, field) with date-clustered errors."""
    out = {"n_boot": n_boot, "seed": seed, "bands": {k: v for k, v in bands.items()},
           "events": {}}
    events = sorted({r["event"] for r in rows})
    for ev in events:
        ev_rows = [r for r in rows if r["event"] == ev]
        ev_out = {}
        for band, leads in bands.items():
            br = [r for r in ev_rows if leads is None or int(r["lead"]) in leads]
            if not br:
                continue
            b_out = {"fields": {}, "diff": {}, "wind_pressure": {}, "counts": {}}
            per_field = {}
            for fname, _ in FIELDS:
                fr = [r for r in br if r["field"] == fname]
                found = [r for r in fr if int(r["found"]) == 1]
                per_field[fname] = found
                dates = [r["date"] for r in found]
                b_out["counts"][fname] = {
                    "n_rows": len(fr),
                    "n_storm_cases": sum(int(r["storm_ok"]) for r in fr),
                    "n_found": len(found),
                    "n_not_found": sum(1 for r in fr if int(r["storm_ok"]) and not int(r["found"])),
                    "n_r34_missing": int(np.isnan(_num(found, "r34_km")).sum()) if found else 0,
                    "n_r50_missing": int(np.isnan(_num(found, "r50_km")).sum()) if found else 0,
                    "n_dates": len(set(dates)),
                    "n_members": len({r["member"] for r in found}),
                    "n_valid_times": len({(r["date"], r["lead"]) for r in found}),
                }
                b_out["fields"][fname] = {s: boot_mean(dates, _num(found, s), n_boot, seed)
                                          for s in SCORES} if found else {}
                if len(found) >= 3:
                    x = pressure_ref_hpa - _num(found, "pmin_hpa")
                    y = _num(found, "vmax_tan_ms")
                    b_out["wind_pressure"][fname] = boot_slope(dates, x, y, n_boot, seed)
            for a, b in (("input", "truth"), ("model", "truth"), ("model", "input")):
                A, B = per_field.get(a, []), per_field.get(b, [])
                if A and B:
                    b_out["diff"][f"{a}_minus_{b}"] = {
                        s: boot_mean_diff([r["date"] for r in A], _num(A, s),
                                          [r["date"] for r in B], _num(B, s), n_boot, seed)
                        for s in SCORES}
            ev_out[band] = b_out
        out["events"][ev] = ev_out
    return out


def run(predictions_dir, lane_config: dict, eval_config: dict, *, output_dir=None,
        overwrite: bool = False, run_label: str = "", **kwargs) -> Path:
    t0 = time.time()
    predictions_dir = Path(predictions_dir).expanduser()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "tc_structure"
    output_dir.mkdir(parents=True, exist_ok=True)

    params = StructureParams(**(eval_config.get("params") or {}))
    storm_edge_km = float(eval_config.get("storm_edge_km", 50.0))
    storm_max_pmin = float(eval_config.get("storm_max_pmin_hpa", 1005.0))
    track_speed = float(eval_config.get("track_km_per_24h", 900.0))
    n_boot = int(eval_config.get("n_boot", DEFAULT_N_BOOT))
    seed = int(eval_config.get("seed", DEFAULT_SEED))
    bands = eval_config.get("lead_bands") or DEFAULT_BANDS
    arm = run_label or eval_config.get("run_label") or predictions_dir.name

    files = _discover(predictions_dir)
    steps = eval_config.get("steps")
    dates = eval_config.get("dates")
    if steps:
        files = [f for f in files if f[2] in {int(s) for s in steps}]
    if dates:
        files = [f for f in files if f"{f[1]:08d}" in {str(d) for d in dates}]
    if not files:
        raise FileNotFoundError(f"tc_structure: no prediction files under {predictions_dir}")
    events = _select_events(eval_config, files)
    if not events:
        raise RuntimeError("tc_structure: no event matches the dates of the prediction files")
    LOG.info("tc_structure: %d files, events %s, git %s", len(files), [e.name for e in events],
             _git_commit())

    rows, profs = [], []
    tracks: dict[int, dict] = {}
    # leads in increasing order within each initial date, so the first-guess track
    # of a date is followed lead after lead
    for path, ymd, step in sorted(files, key=lambda f: (f[1], f[2])):
        evs = [e for e in events if select_prediction_files_for_event([(path, ymd, step)], e)]
        if not evs:
            continue
        tf = time.time()
        r, p, tracks[ymd] = measure_file(
            Path(path), ymd, step, evs, params, arm=arm, storm_edge_km=storm_edge_km,
            storm_max_pmin_hpa=storm_max_pmin, track_km_per_24h=track_speed,
            track=tracks.get(ymd))
        base = len(rows)
        rows += r
        profs += [(base + i, vt) for i, vt in p]
        LOG.info("tc_structure: %s done in %.1fs (%d rows)", Path(path).name, time.time() - tf, len(r))

    with open(output_dir / "cases.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CASE_COLS)
        for r in rows:
            w.writerow([_fmt(r[c]) for c in CASE_COLS])
    if profs:
        np.savez(output_dir / "profiles.npz", row=np.array([i for i, _ in profs]),
                 vt=np.stack([vt for _, vt in profs]),
                 r_centres=(np.arange(int(round(params.rmax_km / params.dr_km))) + 0.5) * params.dr_km)

    summary = aggregate(rows, bands, n_boot, seed, params.pressure_ref_hpa)
    summary["arm"] = arm
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    meta = {
        "evaluator": "tc_structure", "git_commit": _git_commit(), "arm": arm,
        "predictions_dir": str(predictions_dir), "n_files": len(files),
        "files": [str(f[0]) for f in files], "events": [e.name for e in events],
        "params": asdict(params), "storm_edge_km": storm_edge_km,
        "storm_max_pmin_hpa": storm_max_pmin, "track_km_per_24h": track_speed,
        "n_boot": n_boot, "seed": seed,
        "seconds": time.time() - t0,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    LOG.info("tc_structure: wrote %s (%d case rows) in %.1fs", output_dir, len(rows), time.time() - t0)
    return output_dir
