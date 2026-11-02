"""Station observation cache for obs_points, from STVL (the database quaver reads). Atos only, reads only.

Run under ``module load python3 vtb ecmwf-toolbox`` with Metview on PATH (the obs_crps recipe), plus earthkit.data
and mars for the orography. Never writes to FDB.

    python -m eval.evaluators.obs_points.core.fetch --dates 20230101,20230111 --leads 12,24,36,48,60,72 \
        --vars 2t,10ff,msl,tp --box 9,71,-81,41 --out <cache dir>

Writes into <cache dir>:
  obs_<var>.csv   valid (YYYYMMDDHH), lat, lon, stnid, elevation, obs, h_o1280, h_o320
                  2t [K], 10ff [m/s], msl [Pa] at valid = date + lead; tp [mm per --tp-period h] for windows ENDING at
                  date + 0..max lead every 12 h (the extra early window serves the accumulation-window check);
                  h_o1280 / h_o320 = height [m] of the nearest O1280 / O320 grid point (2t height correction)
  hres_<var>.csv  operational HRES at the stations from STVL's model table (date, lead, stnid, hres, hres_elevation),
                  a reference row; skipped per (date, lead) when STVL holds nothing (msl and tp are absent for 2023)
  cache_meta.json dates, leads, vars, box, filters, counts
Filters: quaver toss hard limits (2t 173-333 K, 10ff 0-100 m/s, msl 800-1100 hPa, tp 0-300 mm), one report per
station and time. No analysis gross-error check.
"""
import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

LIMITS = {"2t": (173.0, 333.0), "10ff": (0.0, 100.0), "msl": (80000.0, 110000.0), "tp": (0.0, 300.0)}
G = 9.80665
def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def orography(out, grid, log):
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


def obs_at(param, valid, box, tp_period=12):
    import vtb.media as vm
    kw = dict(table="observation", parameter=param, reference_datetimes=[valid.strftime("%Y-%m-%dT%H:%M:%S")],
              forecast_lengths=[dt.timedelta(hours=0)])
    if param == "tp":
        kw["period"] = dt.timedelta(hours=tp_period)
    fs = vm.stvl_retrieve(**kw)
    if len(fs) == 0:
        return pd.DataFrame()
    df = fs[0].to_dataframe().rename(columns={"value_0": "obs", "latitude": "lat", "longitude": "lon"})
    df = df.dropna(subset=["obs"])
    df["lon"] = ((df["lon"] + 180.0) % 360.0) - 180.0
    lo, hi = LIMITS[param]
    df = df[(df.obs >= lo) & (df.obs <= hi)]
    df = df[(df.lat >= box[0]) & (df.lat <= box[1]) & (df.lon >= box[2]) & (df.lon <= box[3])]
    df = df.drop_duplicates(subset=["stnid"])
    df.insert(0, "valid", valid.strftime("%Y%m%d%H"))
    return df[["valid", "lat", "lon", "stnid", "elevation", "obs"]]


def hres_at(param, date, lead, tp_period=12):
    import vtb.media as vm
    ref = dt.datetime.strptime(date, "%Y%m%d")
    kw = dict(table="model", parameter=param, reference_datetimes=[ref.strftime("%Y-%m-%dT%H:%M:%S")],
              forecast_lengths=[dt.timedelta(hours=lead)],
              model={"class": "od", "stream": "oper", "expver": "0001", "type": "fc"})
    if param == "tp":
        kw["period"] = dt.timedelta(hours=tp_period)
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True, help="comma-separated YYYYMMDD (00 UTC runs)")
    ap.add_argument("--leads", default="12,24,36,48,60,72")
    ap.add_argument("--vars", default="2t,10ff,msl,tp")
    ap.add_argument("--box", default="9,71,-81,41", help="lat0,lat1,lon0,lon1 (lon -180..180)")
    ap.add_argument("--tp-period", type=int, default=12)
    ap.add_argument("--no-hres", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dates = a.dates.split(",")
    leads = [int(x) for x in a.leads.split(",")]
    box = [float(x) for x in a.box.split(",")]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log, counts = [], {}
    trees = {g: orography(out, g, log) for g in ("O1280", "O320")}
    for param in a.vars.split(","):
        hours = range(0, max(leads) + 1, 12) if param == "tp" else leads
        valids = sorted({dt.datetime.strptime(d, "%Y%m%d") + dt.timedelta(hours=h) for d in dates for h in hours})
        frames = []
        for v in valids:
            df = obs_at(param, v, box, a.tp_period)
            log.append(f"obs {param} {v:%Y%m%d%H}: {len(df)} stations"); print(log[-1], flush=True)
            if len(df):
                frames.append(df)
        if not frames:
            log.append(f"obs {param}: nothing"); continue
        obs = pd.concat(frames, ignore_index=True)
        p = xyz(obs.lat.to_numpy(), obs.lon.to_numpy())
        for g, (tree, h) in trees.items():
            obs[f"h_{g.lower()}"] = h[tree.query(p)[1]]
        obs.to_csv(out / f"obs_{param}.csv", index=False, float_format="%.4f")
        counts[param] = int(len(obs))
        if a.no_hres:
            continue
        hres = []
        for d in dates:
            for lead in leads:
                df, status = hres_at(param, d, lead, a.tp_period)
                log.append(f"hres {param} {d} +{lead}: {status} {0 if df is None else len(df)}")
                if df is not None:
                    hres.append(df)
        if hres:
            pd.concat(hres, ignore_index=True).to_csv(out / f"hres_{param}.csv", index=False, float_format="%.4f")
    (out / "fetch_log.txt").write_text("\n".join(log) + "\n")
    (out / "cache_meta.json").write_text(json.dumps({"dates": dates, "leads": leads, "vars": a.vars.split(","),
                                                     "box": box, "tp_period_h": a.tp_period, "limits": LIMITS,
                                                     "rows": counts, "source": "STVL via vtb"}, indent=2))


if __name__ == "__main__":
    main()
