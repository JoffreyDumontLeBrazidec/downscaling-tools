"""A0 station observations for the 30 seed-locked dates of 2023, E2 box. Atos login node, CPU, reads only.

Run under the obs_crps environment (`module load python3 vtb ecmwf-toolbox`, plus Metview on PATH; copy the recipe
from eval/tools/obs_crps if retrieval fails with FileNotFoundError). Never writes to FDB.

Writes into <out>:
  obs_<param>.csv     valid, lat, lon, stnid, elevation, obs, h_o1280, h_o320
                      2t [K], 10ff [m/s], msl [Pa] at valid = date + 12..72 h; tp [mm per 12 h] at valid = date + 0..72 h
                      (the 12 h window ENDS at valid; date + 0 h gives the previous window of +12 h, for the window audit)
                      h_o1280 / h_o320 = model orography height [m] of the nearest O1280 / O320 grid point
  hres_<param>.csv    operational HRES (class od, stream oper, expver 0001) at the stations from STVL's model table,
                      same reference date and lead as our rows: date, lead, stnid, hres, hres_elevation
                      (skipped with a log line when STVL holds nothing)
  orography_O1280.grib, orography_O320.grib, fetch_log.txt

Filters (quaver toss hard limits): 2t [173, 333] K, 10ff [0, 100] m/s, msl [80000, 110000] Pa, tp [0, 300] mm.
No analysis gross-error check (stated in the read). Box 9-71 N, 81 W-41 E (the scorer drops stations > 15 km from an
E2 point).
Usage: python3 fetch_obs_atos.py <out dir>
"""
import datetime as dt
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATES = ["20230101", "20230111", "20230122", "20230201", "20230212", "20230226", "20230304", "20230314", "20230325",
         "20230404", "20230415", "20230425", "20230506", "20230516", "20230527", "20230606", "20230617", "20230627",
         "20230708", "20230718", "20230729", "20230808", "20230822", "20230829", "20230909", "20230919", "20230930",
         "20231010", "20231021", "20231031"]
LEADS = (12, 24, 36, 48, 60, 72)
LIMITS = {"2t": (173.0, 333.0), "10ff": (0.0, 100.0), "msl": (80000.0, 110000.0), "tp": (0.0, 300.0)}
BOX = (9.0, 71.0, -81.0, 41.0)
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


def obs_at(param, valid):
    import vtb.media as vm
    kw = dict(table="observation", parameter=param, reference_datetimes=[valid.strftime("%Y-%m-%dT%H:%M:%S")],
              forecast_lengths=[dt.timedelta(hours=0)])
    if param == "tp":
        kw["period"] = dt.timedelta(hours=12)
    fs = vm.stvl_retrieve(**kw)
    if len(fs) == 0:
        return pd.DataFrame()
    df = fs[0].to_dataframe().rename(columns={"value_0": "obs", "latitude": "lat", "longitude": "lon"})
    df = df.dropna(subset=["obs"])
    df["lon"] = ((df["lon"] + 180.0) % 360.0) - 180.0
    lo, hi = LIMITS[param]
    df = df[(df.obs >= lo) & (df.obs <= hi)]
    df = df[(df.lat >= BOX[0]) & (df.lat <= BOX[1]) & (df.lon >= BOX[2]) & (df.lon <= BOX[3])]
    df = df.drop_duplicates(subset=["stnid"])
    df.insert(0, "valid", valid.strftime("%Y%m%d%H"))
    return df[["valid", "lat", "lon", "stnid", "elevation", "obs"]]


def hres_at(param, date, lead):
    import vtb.media as vm
    ref = dt.datetime.strptime(date, "%Y%m%d")
    kw = dict(table="model", parameter=param, reference_datetimes=[ref.strftime("%Y-%m-%dT%H:%M:%S")],
              forecast_lengths=[dt.timedelta(hours=lead)],
              model={"class": "od", "stream": "oper", "expver": "0001", "type": "fc"})
    if param == "tp":
        kw["period"] = dt.timedelta(hours=12)
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


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    log = []
    trees = {g: orography(out, g, log) for g in ("O1280", "O320")}
    for param in ("2t", "10ff", "msl", "tp"):
        hours = range(0, 73, 12) if param == "tp" else LEADS
        valids = sorted({dt.datetime.strptime(d, "%Y%m%d") + dt.timedelta(hours=h) for d in DATES for h in hours})
        frames = []
        for v in valids:
            df = obs_at(param, v)
            log.append(f"obs {param} {v:%Y%m%d%H}: {len(df)} stations"); print(log[-1], flush=True)
            if len(df):
                frames.append(df)
        obs = pd.concat(frames, ignore_index=True)
        p = xyz(obs.lat.to_numpy(), obs.lon.to_numpy())
        for g, (tree, h) in trees.items():
            obs[f"h_{g.lower()}"] = h[tree.query(p)[1]]
        obs.to_csv(out / f"obs_{param}.csv", index=False, float_format="%.4f")
        hres = []
        for d in DATES:
            for lead in LEADS:
                df, status = hres_at(param, d, lead)
                log.append(f"hres {param} {d} +{lead}: {status} {0 if df is None else len(df)}")
                if df is not None:
                    hres.append(df)
        if hres:
            pd.concat(hres, ignore_index=True).to_csv(out / f"hres_{param}.csv", index=False, float_format="%.4f")
    (out / "fetch_log.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
