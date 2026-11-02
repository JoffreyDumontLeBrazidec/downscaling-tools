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
                  every 12 h, for SEEPS; h_o1280 / h_o320 = height [m] of the nearest O1280 / O320 grid point
                  (2t height correction)
  hres_<var>.csv  operational HRES at the stations from STVL's model table (date, lead, stnid, hres, hres_elevation),
                  a reference row; skipped per (date, lead) when STVL holds nothing (msl and tp are absent for 2023)
  orography_O1280.grib, orography_O320.grib (reused when present), fetch_log.txt
  cache_meta.json cache_version, created (UTC), host, dates, leads, vars, box, filters, the STVL request
                  parameters, rows per file and stations per valid time
Filters: quaver toss hard limits (2t 173-333 K, 10ff 0-100 m/s, msl 800-1100 hPa, tp 0-300 mm), one report per
station and time. No analysis gross-error check.
"""
import argparse
import datetime as dt
import json
import socket
import subprocess
from pathlib import Path

import numpy as np

from eval.evaluators.obs_points.core.stations import LIMITS, xyz

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


def _stvl_param(var):
    """(STVL parameter, accumulation hours or None) of a cache variable: tp is 12 h, tp24 is 24 h."""
    if var == "tp":
        return "tp", 12
    if var.startswith("tp") and var[2:].isdigit():
        return "tp", int(var[2:])
    return var, None


def obs_at(var, valid, box):
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
    lo, hi = LIMITS.get(var, LIMITS.get(param))
    df = df[(df.obs >= lo) & (df.obs <= hi)]
    df = df[(df.lat >= box[0]) & (df.lat <= box[1]) & (df.lon >= box[2]) & (df.lon <= box[3])]
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


def valid_hours(var, leads):
    """Hours after the 00 UTC date at which the cache holds observations of ``var``."""
    _, period = _stvl_param(var)
    if period is None:
        return sorted(set(leads))
    lo = 0 if period == 12 else period
    return list(range(lo, max(leads) + 1, 12))


def fetch_cache(dates, leads, variables, box, out, *, hres=True):
    import pandas as pd

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    log, rows, per_valid = [], {}, {}
    trees = {g: orography(out, g, log) for g in ("O1280", "O320")}
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
    (out / "fetch_log.txt").write_text("\n".join(log) + "\n")
    meta = {
        "cache_version": CACHE_VERSION, "kind": "station observations (obs_points)",
        "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "host": socket.gethostname(),
        "dates": list(dates), "leads": list(leads), "vars": list(variables), "box": list(box),
        "limits": {v: LIMITS[v] for v in variables if v in LIMITS},
        "stvl": {"observation": {"table": "observation", "forecast_lengths_h": [0], "reference_datetimes": "valid time",
                                 "period_h": {v: _stvl_param(v)[1] for v in variables if _stvl_param(v)[1]}},
                 "hres": None if not hres else {"table": "model", "model": HRES_MODEL}},
        "orography": "mars class=od stream=enfo type=pf number=1 date=2024-01-01 step=0 param=z, grids O1280 and O320",
        "rows": rows, "stations_per_valid": per_valid, "source": "STVL via vtb",
    }
    (out / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    return out


def add_args(ap):
    ap.add_argument("--dates", required=True, help="comma-separated YYYYMMDD (00 UTC runs)")
    ap.add_argument("--leads", default="12,24,36,48,60,72")
    ap.add_argument("--vars", default="2t,10ff,msl,tp")
    ap.add_argument("--tp-periods", default="12", help="tp accumulations in hours: 12 gives tp, 24 adds tp24 (SEEPS)")
    ap.add_argument("--box", default="9,71,-81,41", help="lat0,lat1,lon0,lon1 (lon -180..180)")
    ap.add_argument("--no-hres", action="store_true", help="skip HRES at the stations")
    ap.add_argument("--out", required=True, help="cache directory (created)")


def variables_from_args(a):
    out = []
    for v in a.vars.split(","):
        if v == "tp":
            out += ["tp" if p == 12 else f"tp{p}" for p in (int(x) for x in a.tp_periods.split(","))]
        else:
            out.append(v)
    return out


def run_from_args(a):
    require_atos()
    return fetch_cache(a.dates.split(","), [int(x) for x in a.leads.split(",")], variables_from_args(a),
                       [float(x) for x in a.box.split(",")], a.out, hres=not a.no_hres)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_args(ap)
    run_from_args(ap.parse_args(argv))


if __name__ == "__main__":
    main()
