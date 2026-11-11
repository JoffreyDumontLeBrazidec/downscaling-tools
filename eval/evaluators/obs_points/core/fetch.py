"""Station observation cache for obs_points, from STVL (the database quaver reads). Atos only, reads only.

Run as ``python3 -m eval.cli obs fetch`` under ``module load python3 vtb ecmwf-toolbox`` (the module's
python3, not the eval venv: vtb lives there), with TMPDIR set to a folder you own. mars is used for the
model orography. Never writes to FDB.

    python3 -m eval.cli obs fetch --dates 20230101,20230111 --leads 12,24,36,48,60,72 \
        --vars 2t,10ff,msl,tp --tp-periods 12,24 --box 9,71,-81,41 --out <cache dir>

Writes into <cache dir>:
  obs_<var>.csv   valid (YYYYMMDDHH), lat, lon, stnid, elevation, obs, h_o1280, h_o320
                  2t [K], 10ff [m/s], msl [Pa] at valid = date + lead; tp [mm per 12 h] for windows ENDING at
                  date + 0..max lead every 12 h (the extra early window serves the accumulation-window check);
                  tp24 [mm per 24 h] (``--tp-periods`` includes 24) for windows ending at date + 24..max lead
                  every 12 h, for SEEPS; tp<N> with N under 12 (``--tp-periods 6`` gives tp6, the 4 km lane's
                  6 h tp) [mm per N h] for windows ending at every lead of at least N h;
                  h_o1280 / h_o320 = height [m] of the nearest O1280 / O320 grid point (2t height correction);
                  h_<name> (``--add-height <name>=<source>``) the same from another orography, e.g. h_o2560
  hres_<var>.csv  operational HRES at the stations from STVL's model table (date, lead, stnid, hres, hres_elevation),
                  a reference row; skipped per (date, lead) when STVL holds nothing (msl and tp are absent for 2023)
  orography_O1280.grib, orography_O320.grib (reused when present), fetch_log.txt
  seeps_clim_tp24.csv  (``--seeps``) the station climatology quaver's SEEPS uses (VINO ``rodw``, climate period
                  1980-2009, category equitable_categories, synop; it exists for 24 h only): valid, stnid, lat, lon,
                  p1 (probability of a dry day), t1 (dry/light boundary, 0.2 mm), t2 (light/heavy boundary, mm) and
                  s0..s8, the 3 x 3 SEEPS error matrix, index = forecast category x 3 + observed category
  cache_meta.json cache_version, created (UTC), host, dates, leads, vars, box, filters, the STVL request
                  parameters, rows per file and stations per valid time; a later fetch into the same directory
                  adds its files and merges its entries into the manifest
Filters: quaver toss hard limits (2t 173-333 K, 10ff 0-100 m/s, msl 800-1100 hPa, tp 0-300 mm), one report per
station and time, and quaver's analysis-departure check (toss_with_analysis): the operational analysis
(od oper an, O1280) at valid time is read at the nearest grid point and stored as column ``an`` of obs_<var>.csv
(2t, 10ff, msl); rows with |obs - an| above 30 K, 40 m/s or 1000 Pa are dropped at fetch and again at load
(``stations.ANL_DEP_MAX``). quaver does not height-correct the analysis 2t before this check; neither do we.
``--add-analysis`` adds the ``an`` column to an existing cache in place (caches before 9 Nov 2026 lack it:
on 1-12 Nov 2024 they kept 68-89 msl reports near 91,000 Pa and a 63-97 m/s wind station per lead).
``--add-height <name>=<source>`` (repeatable) adds column ``h_<name>``, the height [m] of the nearest point of
that orography: a GRIB file of z (geopotential, any grid) or an anemoi zarr dataset with a ``z`` variable
(row 0, its own latitudes/longitudes), both divided by g. With ``--dates`` it goes into the files of that
fetch; without, into every obs_<var>.csv already in --out, in place (e.g. h_o2560 for the 4 km lane, whose
model rows obs_points corrects with ``model_height_column: h_o2560``).
"""
import argparse
import datetime as dt
import json
import socket
import subprocess
from pathlib import Path

import numpy as np

from eval.evaluators.obs_points.core.stations import ANL_DEP_MAX, LIMITS, analysis_departure_ok, xyz

CACHE_VERSION = 1
G = 9.80665
HRES_MODEL = {"class": "od", "stream": "oper", "expver": "0001", "type": "fc"}
ATOS_PREFIXES = ("aa", "ab", "ac", "ad", "ag")


def require_atos(what: str = "obs fetch") -> None:
    """Stop with a clear message unless this is an Atos node with vtb importable."""
    host = socket.gethostname()
    if not (host[:2] in ATOS_PREFIXES and host[2:3].isdigit()):
        raise SystemExit(f"{what} runs on Atos only (STVL through vtb, and mars); this host is {host!r}. "
                         "Fetch there and copy the cache directory to the other cluster.")
    try:
        import vtb.media  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"{what}: cannot import vtb ({exc}). Run it as `module load python3 vtb ecmwf-toolbox; "
                         "export TMPDIR=<a folder you own>; python3 -m eval.cli obs ...` (the module python3, "
                         "not the eval venv).") from exc


def orography(out, grid, log):
    from scipy.spatial import cKDTree

    path = out / f"orography_{grid}.grib"
    if not path.exists():
        req = ("retrieve,\n  class=od, expver=0001, stream=enfo, type=pf, number=1,\n  date=2024-01-01, time=0000, step=0,\n"
               f"  levtype=sfc, param=z, grid={grid}, database=off,\n  target=\"{path}\"\n")
        subprocess.run(["mars"], input=req, text=True, check=True)
        log.append(f"orography {grid}: retrieved {path}")
    import earthkit.data as ekd
    f = ekd.from_source("file", str(path)).to_fieldlist()[0]
    lat, lon = f.geography.latitudes(), f.geography.longitudes()
    lon = ((np.asarray(lon) + 180.0) % 360.0) - 180.0
    return cKDTree(xyz(np.asarray(lat), lon)), f.to_numpy().ravel() / G


def height_source(src):
    """(cKDTree, height [m]) of an extra orography: a GRIB file of z (geopotential, any grid) or an anemoi zarr
    dataset with a ``z`` variable (row 0 of ``data``, its own latitudes/longitudes); both divided by g."""
    from scipy.spatial import cKDTree

    src = Path(src)
    if not src.exists():
        raise SystemExit(f"--add-height: no orography source at {src}")
    if src.is_dir():  # anemoi zarr dataset: data (dates, variables, ensemble, values), attrs["variables"]
        import zarr

        root = zarr.open(str(src), mode="r")
        names = list(root.attrs["variables"])
        if "z" not in names:
            raise ValueError(f"{src}: no variable z among {names}")
        z = np.asarray(root["data"][0, names.index("z")], np.float64).reshape(-1, root["latitudes"].shape[0])[0]
        lat, lon = np.asarray(root["latitudes"][:], np.float64), np.asarray(root["longitudes"][:], np.float64)
    else:
        import eccodes  # directly, not earthkit: its field API differs between the module and the venv versions

        with open(src, "rb") as f:
            while (gid := eccodes.codes_grib_new_from_file(f)) is not None:
                try:
                    if eccodes.codes_get(gid, "shortName") == "z":
                        lat, lon, z = (np.asarray(eccodes.codes_get_double_array(gid, k), np.float64)
                                       for k in ("latitudes", "longitudes", "values"))
                        break
                finally:
                    eccodes.codes_release(gid)
            else:
                raise ValueError(f"{src}: no z (geopotential) field")
    lon = ((lon + 180.0) % 360.0) - 180.0
    return cKDTree(xyz(lat, lon)), z / G


def parse_heights(specs):
    """{name: source} from ``--add-height`` values ``<name>=<source>`` (name: lower-case letters and digits)."""
    out = {}
    for s in specs or ():
        name, sep, src = s.partition("=")
        name = name.strip().lower()
        if not sep or not name.isalnum() or not src.strip():
            raise SystemExit(f"--add-height expects <name>=<source>, e.g. o2560=/path/orography_O2560.grib; got {s!r}")
        out[name] = src.strip()
    return out


def add_height_columns(obs, sources):
    """Column h_<name> per orography in ``sources`` (name -> (tree, height)): the height of the nearest point."""
    p = xyz(obs.lat.to_numpy(), obs.lon.to_numpy())
    for name, (tree, h) in sources.items():
        obs[f"h_{name}"] = h[tree.query(p)[1]]
    return obs


def height_meta(name, src, h, log):
    log.append(f"height h_{name}: {src}, {h.size} points, {np.nanmin(h):.1f} to {np.nanmax(h):.1f} m")
    print(log[-1], flush=True)
    return {"source": str(Path(src).resolve()), "points": int(h.size), "min_m": round(float(np.nanmin(h)), 1),
            "max_m": round(float(np.nanmax(h)), 1), "nearest point": True,
            "added": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def add_height(out, heights):
    """Add the columns h_<name> of ``heights`` (name -> source) to every obs_<var>.csv of an existing cache, in place."""
    import pandas as pd

    out = Path(out); log = []
    files = sorted(out.glob("obs_*.csv"))
    if not files:
        raise SystemExit(f"obs fetch --add-height: no obs_<var>.csv in {out}")
    sources, info = {}, {}
    for name, src in heights.items():
        sources[name] = height_source(src)
        info[name] = height_meta(name, src, sources[name][1], log)
    for p in files:
        o = add_height_columns(pd.read_csv(p, dtype={"valid": str, "stnid": str}), sources)
        tmp = p.with_suffix(".csv.tmp")
        o.to_csv(tmp, index=False, float_format="%.4f")
        tmp.replace(p)  # a crash mid-write leaves the old file whole
        log.append(f"height {', '.join('h_' + n for n in sources)}: {p.name}, {len(o)} rows")
    for name in info:
        info[name]["files"] = [p.name for p in files]
    with open(out / "fetch_log.txt", "a") as f:
        f.write("\n".join(log) + "\n")
    mp = out / "cache_meta.json"
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    meta["heights"] = {**meta.get("heights", {}), **info}
    mp.write_text(json.dumps(meta, indent=2))
    return info


def analysis_fields(out, valid, log, grid="O1280"):
    """The operational analysis (od oper an) at ``valid``: {var: values} for 2t, 10ff, msl on ``grid``,
    retrieved once per valid time into <out>/analysis/ (what quaver's toss_with_analysis reads)."""
    d = out / "analysis"; d.mkdir(exist_ok=True)
    path = d / f"an_{valid:%Y%m%d%H}_{grid}.grib"
    if not path.exists():
        req = ("retrieve,\n  class=od, expver=0001, stream=oper, type=an, levtype=sfc,\n"
               f"  date={valid:%Y%m%d}, time={valid:%H}00, step=0,\n  param=2t/10u/10v/msl, grid={grid},\n"
               f"  target=\"{path}\"\n")
        subprocess.run(["mars"], input=req, text=True, check=True)
        log.append(f"analysis {valid:%Y%m%d%H}: retrieved {path}")
    import earthkit.data as ekd
    fl = ekd.from_source("file", str(path)).to_fieldlist()
    by = {f.metadata("shortName"): f.to_numpy().ravel() for f in fl}
    missing = {"2t", "10u", "10v", "msl"} - set(by)
    if missing:
        raise ValueError(f"analysis {path}: missing {sorted(missing)}")
    return {"2t": by["2t"], "10ff": np.hypot(by["10u"], by["10v"]), "msl": by["msl"]}


def add_analysis_column(obs, var, out, tree, log):
    """Add column ``an`` (analysis at the nearest O1280 point, valid time of the row) to an obs frame of a
    variable with an analysis-departure limit; other variables are returned unchanged."""
    if var not in ANL_DEP_MAX or not len(obs):
        return obs
    obs = obs.copy()
    obs["an"] = np.nan
    idx = tree.query(xyz(obs.lat.to_numpy(), obs.lon.to_numpy()))[1]
    for v in sorted(obs["valid"].astype(str).unique()):
        rows = (obs["valid"].astype(str) == v).to_numpy()
        an = analysis_fields(out, dt.datetime.strptime(v, "%Y%m%d%H"), log)[var]
        obs.loc[rows, "an"] = an[idx[rows]]
    return obs


def drop_gross(obs, var, log, tag):
    keep = analysis_departure_ok(obs, var)
    n = int((~keep).sum())
    if n:
        log.append(f"analysis check {var} {tag}: dropped {n} of {len(obs)} reports "
                   f"(|obs - an| > {ANL_DEP_MAX[var]})"); print(log[-1], flush=True)
    return obs[keep]


def add_analysis(out, variables):
    """Add the ``an`` column to the obs files of an existing cache and drop the gross errors, in place."""
    import pandas as pd

    out = Path(out); log, dropped = [], {}
    tree = orography(out, "O1280", log)[0]
    for var in variables:
        p = out / f"obs_{var}.csv"
        if var not in ANL_DEP_MAX or not p.exists():
            continue
        o = pd.read_csv(p, dtype={"valid": str, "stnid": str})
        n0 = len(o)
        o = drop_gross(add_analysis_column(o.drop(columns=["an"], errors="ignore"), var, out, tree, log),
                       var, log, "existing cache")
        dropped[var] = n0 - len(o)
        o.to_csv(p, index=False, float_format="%.4f")
    with open(out / "fetch_log.txt", "a") as f:
        f.write("\n".join(log) + "\n")
    mp = out / "cache_meta.json"
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    meta["analysis_check"] = {"source": "od oper an O1280 at valid time, nearest point", "limits": ANL_DEP_MAX,
                              "added": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                              "dropped_existing": {**meta.get("analysis_check", {}).get("dropped_existing", {}),
                                                   **dropped}}
    mp.write_text(json.dumps(meta, indent=2))
    return dropped


def _stvl_param(var):
    """(STVL parameter, accumulation hours or None) of a cache variable: tp is 12 h, tp24 is 24 h."""
    if var == "tp":
        return "tp", 12
    if var.startswith("tp") and var[2:].isdigit():
        return "tp", int(var[2:])
    return var, None


def obs_at(var, valid, box, *, limits=True, dedup=True):
    """Station observations of ``var`` valid at ``valid`` inside ``box``; quaver's hard limits and one report
    per station unless ``limits`` / ``dedup`` are False (precip_read applies its own obs_max when it loads)."""
    import pandas as pd
    import vtb.media as vm

    param, period = _stvl_param(var)
    kw = dict(table="observation", parameter=param, reference_datetimes=[valid.strftime("%Y-%m-%dT%H:%M:%S")],
              forecast_lengths=[dt.timedelta(hours=0)])
    if period:
        kw["period"] = dt.timedelta(hours=period)
    fs = vm.stvl_retrieve(**kw)
    if len(fs) == 0:
        return pd.DataFrame()
    df = fs[0].to_dataframe().rename(columns={"value_0": "obs", "latitude": "lat", "longitude": "lon"})
    df = df.dropna(subset=["obs"])
    df["lon"] = ((df["lon"] + 180.0) % 360.0) - 180.0
    if limits:
        lo, hi = LIMITS.get(var, LIMITS.get(param))
        df = df[(df.obs >= lo) & (df.obs <= hi)]
    df = df[(df.lat >= box[0]) & (df.lat <= box[1]) & (df.lon >= box[2]) & (df.lon <= box[3])]
    if dedup:
        df = df.drop_duplicates(subset=["stnid"])
    df.insert(0, "valid", valid.strftime("%Y%m%d%H"))
    return df[["valid", "lat", "lon", "stnid", "elevation", "obs"]]


def hres_at(var, date, lead):
    import vtb.media as vm

    param, period = _stvl_param(var)
    ref = dt.datetime.strptime(date, "%Y%m%d")
    kw = dict(table="model", parameter=param, reference_datetimes=[ref.strftime("%Y-%m-%dT%H:%M:%S")],
              forecast_lengths=[dt.timedelta(hours=lead)], model=dict(HRES_MODEL))
    if period:
        kw["period"] = dt.timedelta(hours=period)
    try:
        fs = vm.stvl_retrieve(**kw)
    except Exception as e:  # noqa: BLE001 - STVL raises on unknown combinations; logged, not fatal
        return None, repr(e)
    if len(fs) == 0:
        return None, "empty"
    df = fs[0].to_dataframe().rename(columns={"value_0": "hres", "elevation": "hres_elevation"})
    df = df.drop_duplicates(subset=["stnid"])
    df.insert(0, "lead", lead); df.insert(0, "date", date)
    return df[["date", "lead", "stnid", "hres", "hres_elevation"]], "ok"


SEEPS_CLIM = {"climate_period": "1980-2009", "category": "equitable_categories"}
SEEPS_MODEL = {"class": "od", "stream": "oper", "expver": "0001", "type": "ec", "anoffset": 0}


def seeps_clim_at(valid, box):
    """The 13 SEEPS climatology values per station for the 24 h window ending at ``valid`` (VINO through STVL):
    number 0 = p1, 1 = t1, 3 = t2, 4..12 = the 3 x 3 matrix (quaver ScoresLibAux.seepsField)."""
    import pandas as pd
    import vtb.media as vm

    fs = vm.stvl_retrieve(table="climatology", parameter="tp", period=dt.timedelta(hours=24),
                          climatology=dict(SEEPS_CLIM), model=dict(SEEPS_MODEL), sources=["synop"],
                          reference_datetimes=[valid.strftime("%Y-%m-%dT%H:%M:%S")],
                          forecast_lengths=[dt.timedelta(hours=0)])
    if len(fs) < 13:
        return pd.DataFrame()
    numbers = [int(x) for x in fs.header["number"]]
    if numbers[:13] != list(range(13)):
        raise ValueError(f"SEEPS climatology fields out of order: numbers {numbers}")

    def field(k, name):  # one climatology field by station id (never by row position)
        f = fs[k].to_dataframe().drop_duplicates(subset=["stnid"]).set_index("stnid")
        return f["value_0"].rename(name)

    base = fs[0].to_dataframe().drop_duplicates(subset=["stnid"]).rename(
        columns={"latitude": "lat", "longitude": "lon", "value_0": "p1"}).set_index("stnid")[["lat", "lon", "p1"]]
    out = base.join([field(1, "t1"), field(3, "t2")] + [field(4 + k, f"s{k}") for k in range(9)], how="inner")
    out = out.reset_index()
    out["lon"] = ((out["lon"] + 180.0) % 360.0) - 180.0
    out = out[(out.lat >= box[0]) & (out.lat <= box[1]) & (out.lon >= box[2]) & (out.lon <= box[3])]
    out = out.drop_duplicates(subset=["stnid"])
    out.insert(0, "valid", valid.strftime("%Y%m%d%H"))
    return out


def valid_hours(var, leads):
    """Hours after the 00 UTC date at which the cache holds observations of ``var``: the leads; for 12 h tp the
    windows ending every 12 h from 0, for 24 h from 24; for a period under 12 h (6 h tp of the 4 km lane) the
    windows ending at each lead of at least one period."""
    _, period = _stvl_param(var)
    if period is None:
        return sorted(set(leads))
    if period < 12:
        return sorted(h for h in set(leads) if h >= period)
    lo = 0 if period == 12 else period
    return list(range(lo, max(leads) + 1, 12))


def fetch_cache(dates, leads, variables, box, out, *, hres=True, seeps=False, an_check=True, heights=None):
    import pandas as pd

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    log, rows, per_valid = [], {}, {}
    trees = {g: orography(out, g, log) for g in ("O1280", "O320")} if variables else {}
    extra = {n: height_source(s) for n, s in (heights or {}).items()} if variables else {}
    height_info = {n: height_meta(n, heights[n], extra[n][1], log) for n in extra}
    for var in variables:
        valids = sorted({dt.datetime.strptime(d, "%Y%m%d") + dt.timedelta(hours=h)
                         for d in dates for h in valid_hours(var, leads)})
        frames, per_valid[var] = [], {}
        for v in valids:
            df = obs_at(var, v, box)
            per_valid[var][v.strftime("%Y%m%d%H")] = int(len(df))
            log.append(f"obs {var} {v:%Y%m%d%H}: {len(df)} stations"); print(log[-1], flush=True)
            if len(df):
                frames.append(df)
        if not frames:
            log.append(f"obs {var}: nothing"); continue
        obs = pd.concat(frames, ignore_index=True)
        p = xyz(obs.lat.to_numpy(), obs.lon.to_numpy())
        for g, (tree, h) in trees.items():
            obs[f"h_{g.lower()}"] = h[tree.query(p)[1]]
        obs = add_height_columns(obs, extra)
        for n in extra:
            height_info[n].setdefault("files", []).append(f"obs_{var}.csv")
        if an_check:
            obs = drop_gross(add_analysis_column(obs, var, out, trees["O1280"][0], log), var, log, "fetch")
        obs.to_csv(out / f"obs_{var}.csv", index=False, float_format="%.4f")
        rows[f"obs_{var}.csv"] = int(len(obs))
        if not hres or var == "tp24":
            continue
        frames = []
        for d in dates:
            for lead in leads:
                df, status = hres_at(var, d, lead)
                log.append(f"hres {var} {d} +{lead}: {status} {0 if df is None else len(df)}")
                if df is not None:
                    frames.append(df)
        if frames:
            h = pd.concat(frames, ignore_index=True)
            h.to_csv(out / f"hres_{var}.csv", index=False, float_format="%.4f")
            rows[f"hres_{var}.csv"] = int(len(h))
    if seeps:
        frames, per_valid["seeps_clim_tp24"] = [], {}
        for v in sorted({dt.datetime.strptime(d, "%Y%m%d") + dt.timedelta(hours=h)
                         for d in dates for h in valid_hours("tp24", leads)}):
            df = seeps_clim_at(v, box)
            per_valid["seeps_clim_tp24"][v.strftime("%Y%m%d%H")] = int(len(df))
            log.append(f"seeps climatology {v:%Y%m%d%H}: {len(df)} stations"); print(log[-1], flush=True)
            if len(df):
                frames.append(df)
        if frames:
            c = pd.concat(frames, ignore_index=True)
            c.to_csv(out / "seeps_clim_tp24.csv", index=False, float_format="%.5f")
            rows["seeps_clim_tp24.csv"] = int(len(c))
    with open(out / "fetch_log.txt", "a") as f:
        f.write("\n".join(log) + "\n")
    meta = {
        "cache_version": CACHE_VERSION, "kind": "station observations (obs_points)",
        "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "host": socket.gethostname(),
        "dates": list(dates), "leads": list(leads), "vars": list(variables), "box": list(box),
        "limits": {v: LIMITS[v] for v in variables if v in LIMITS},
        "analysis_check": None if not an_check else {"source": "od oper an O1280 at valid time, nearest point",
                                                     "limits": {v: ANL_DEP_MAX[v] for v in variables if v in ANL_DEP_MAX}},
        "stvl": None if not variables else {
            "observation": {"table": "observation", "forecast_lengths_h": [0], "reference_datetimes": "valid time",
                            "period_h": {v: _stvl_param(v)[1] for v in variables if _stvl_param(v)[1]}},
            "hres": None if not hres else {"table": "model", "model": HRES_MODEL}},
        "orography": "mars class=od stream=enfo type=pf number=1 date=2024-01-01 step=0 param=z, grids O1280 and O320",
        "rows": rows, "stations_per_valid": per_valid, "source": "STVL via vtb", "heights": height_info,
    }
    if seeps:
        meta["seeps_climatology"] = {"table": "climatology", "climat": "rodw", **SEEPS_CLIM, "model": SEEPS_MODEL,
                                     "sources": ["synop"], "period_h": 24}
    path = out / "cache_meta.json"
    if path.exists():  # a second fetch into the same cache: keep what is there, add or replace the new entries
        old = json.loads(path.read_text())
        this = {k: meta[k] for k in ("created", "host", "dates", "leads", "vars")} | {"seeps": seeps}
        for k in ("rows", "stations_per_valid", "limits", "heights"):
            meta[k] = {**old.get(k, {}), **meta[k]}
        meta["vars"] = sorted(set(old.get("vars", [])) | set(meta["vars"]))
        meta["history"] = old.get("history", [{k: old.get(k) for k in ("created", "host", "dates", "leads", "vars")}])
        meta["history"].append(this)
        for k in ("stvl", "seeps_climatology", "orography", "analysis_check"):
            if k not in meta or meta[k] is None:
                meta[k] = old.get(k)
        meta["dates"] = sorted(set(old.get("dates", [])) | set(meta["dates"]))
        meta["leads"] = sorted(set(old.get("leads", [])) | set(meta["leads"]))
    path.write_text(json.dumps(meta, indent=2))
    return out


def add_args(ap):
    ap.add_argument("--dates", help="comma-separated YYYYMMDD (00 UTC runs); required unless --add-analysis or --add-height alone")
    ap.add_argument("--leads", default="12,24,36,48,60,72")
    ap.add_argument("--vars", default="2t,10ff,msl,tp")
    ap.add_argument("--tp-periods", default="12", help="tp accumulations in hours: 12 gives tp, 24 adds tp24 (SEEPS)")
    ap.add_argument("--box", default="9,71,-81,41", help="lat0,lat1,lon0,lon1 (lon -180..180)")
    ap.add_argument("--no-hres", action="store_true", help="skip HRES at the stations")
    ap.add_argument("--seeps", action="store_true",
                    help="also fetch the SEEPS station climatology (24 h windows ending at date + 24..max lead)")
    ap.add_argument("--no-an-check", action="store_true",
                    help="skip quaver's analysis-departure check (no od oper an retrieval, no 'an' column)")
    ap.add_argument("--add-analysis", action="store_true",
                    help="only add the 'an' column to the obs files already in --out and drop the gross errors")
    ap.add_argument("--add-height", action="append", default=[], metavar="NAME=SOURCE",
                    help="add column h_NAME, the height [m] of the nearest point of an orography (GRIB of z on any "
                         "grid, or an anemoi zarr with z), e.g. o2560=orography_O2560.grib; repeatable. With --dates "
                         "into the files fetched; without, into every obs_<var>.csv already in --out, in place")
    ap.add_argument("--out", required=True, help="cache directory (created)")


def variables_from_args(a):
    out = []
    for v in filter(None, a.vars.split(",")):
        if v == "tp":
            out += ["tp" if p == 12 else f"tp{p}" for p in (int(x) for x in a.tp_periods.split(","))]
        else:
            out.append(v)
    return out


def run_from_args(a):
    heights = parse_heights(getattr(a, "add_height", None))
    if heights and not a.dates and not a.add_analysis:
        return add_height(a.out, heights)  # reads local files only: runs on any cluster
    require_atos()
    if a.add_analysis:
        done = add_analysis(a.out, variables_from_args(a))
        return (done, add_height(a.out, heights)) if heights else done
    if not a.dates:
        raise SystemExit("obs fetch: --dates is required")
    return fetch_cache(a.dates.split(","), [int(x) for x in a.leads.split(",")], variables_from_args(a),
                       [float(x) for x in a.box.split(",")], a.out, hres=not a.no_hres, seeps=a.seeps,
                       an_check=not a.no_an_check, heights=heights)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_args(ap)
    run_from_args(ap.parse_args(argv))


if __name__ == "__main__":
    main()
