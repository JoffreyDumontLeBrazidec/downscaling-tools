"""Station observations for precip_read: fetch from STVL (the database quaver reads) and score against them.

Fetch (Atos only, needs ``module load python3 vtb ecmwf-toolbox``)::

    python -m eval.evaluators.precip_read.core.obs fetch --dates 20230101,20230111 --leads 24,72 \\
        --period 12 --box 10,70,-80,40 --out tp_obs.csv

writes one CSV with columns valid (YYYYMMDDHH), lat, lon, stnid, elevation, obs (mm per period). The
accumulation of a station value ends at its valid time. ``station_rows`` scores one prediction file
against it: each station takes the nearest grid point (as quaver does), stations farther than
``max_km`` from any point are dropped, and values outside [0, obs_max] are treated as bad reports.
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from eval.evaluators.precip_read.core.metrics import fair_crps

R_EARTH_KM = 6371.0


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def load_obs(path, obs_max: float = 300.0):
    import pandas as pd

    obs = pd.read_csv(path, dtype={"valid": str})
    n_bad = int(((obs.obs < 0) | (obs.obs > obs_max)).sum())
    return obs[(obs.obs >= 0) & (obs.obs <= obs_max)], n_bad


class StationMatcher:
    """Nearest grid point of each station, on one fixed grid."""

    def __init__(self, lat, lon):
        from scipy.spatial import cKDTree

        self.lat, self.lon = np.asarray(lat), np.asarray(lon)
        self.tree = cKDTree(_xyz(self.lat, self.lon))

    def match(self, slat, slon, max_km: float):
        dist, idx = self.tree.query(_xyz(np.asarray(slat), np.asarray(slon)))
        return idx, dist * R_EARTH_KM <= max_km


def point_metrics(f, ref, ens=None) -> dict:
    """Per-date metrics of one forecast at the stations: bias, RMSE (ensemble mean), RMSE over the
    reference SD, Pearson correlation, fair CRPS (absolute error for a single field), spread."""
    f, ref = np.asarray(f, np.float64), np.asarray(ref, np.float64)
    d = f - ref
    rmse = float(np.sqrt((d ** 2).mean()))
    sd = ref.std()
    out = {"n": int(ref.size), "ref_mean": float(ref.mean()), "f_mean": float(f.mean()), "bias": float(d.mean()),
           "rmse": rmse, "nrmse": rmse / sd if sd > 0 else np.nan,
           "corr": float(np.corrcoef(f, ref)[0, 1]) if f.std() > 0 and sd > 0 else np.nan,
           "crps": float(np.abs(d).mean()), "spread": 0.0, "member_rmse": np.nan}
    if ens is not None:
        ens = np.asarray(ens, np.float64)
        out["crps"] = float(fair_crps(ens, ref).mean())
        out["spread"] = float(ens.std(0, ddof=1).mean()) if ens.shape[0] > 1 else 0.0
        out["member_rmse"] = float(np.sqrt(((ens - ref[None]) ** 2).mean()))
    return out


def station_rows(obs, matcher, *, date, step, truth, inp, ens, regions, max_km=15.0, min_stations=30):
    """Score rows (ref = obs) of one file: model ensemble, interpolated input and the truth field."""
    import pandas as pd

    valid = (pd.Timestamp(date) + pd.Timedelta(hours=int(step))).strftime("%Y%m%d%H")
    o = obs[obs.valid == valid]
    if o.empty:
        return [], 0
    idx, keep = matcher.match(o.lat.to_numpy(), o.lon.to_numpy(), max_km)
    o, idx = o[keep], idx[keep]
    ov, la, lo = o.obs.to_numpy(np.float64), o.lat.to_numpy(), o.lon.to_numpy()
    rows = []
    for name, b in regions.items():
        m = np.ones(ov.size, bool) if b is None else (la >= b[0]) & (la < b[1]) & (lo >= b[2]) & (lo < b[3])
        if m.sum() < min_stations:
            continue
        base = {"ref": "obs", "region": name, "lead": int(step), "date": date}
        rows.append({**base, "forecast": "model", **point_metrics(ens.mean(0)[idx][m], ov[m], ens[:, idx][:, m])})
        rows.append({**base, "forecast": "input", **point_metrics(inp[idx][m], ov[m])})
        rows.append({**base, "forecast": "truth", **point_metrics(truth[idx][m], ov[m])})
    return rows, int(ov.size)


def fetch(dates, leads, period_h, box, out):
    """STVL station tp ending at date + lead, inside box [lat0, lat1, lon0, lon1]."""
    import pandas as pd
    import vtb.media as vm

    frames = []
    for d in dates:
        t0 = dt.datetime.strptime(d, "%Y%m%d")
        for h in leads:
            v = t0 + dt.timedelta(hours=int(h))
            fs = vm.stvl_retrieve(table="observation", parameter="tp", period=dt.timedelta(hours=period_h),
                                  reference_datetimes=[v.strftime("%Y-%m-%dT%H:%M:%S")],
                                  forecast_lengths=[dt.timedelta(hours=0)])
            if len(fs) == 0:
                print("no observations", v, flush=True)
                continue
            df = fs[0].to_dataframe().rename(columns={"value_0": "obs", "latitude": "lat", "longitude": "lon"})
            df = df.dropna(subset=["obs"])
            df["lon"] = ((df["lon"] + 180.0) % 360.0) - 180.0
            df = df[(df.lat >= box[0]) & (df.lat <= box[1]) & (df.lon >= box[2]) & (df.lon <= box[3])]
            df.insert(0, "valid", v.strftime("%Y%m%d%H"))
            frames.append(df[["valid", "lat", "lon", "stnid", "elevation", "obs"]])
            print(v, len(df), "stations, max", df.obs.max(), flush=True)
    pd.concat(frames).to_csv(out, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="retrieve station tp from STVL (Atos, vtb)")
    f.add_argument("--dates", required=True, help="comma-separated YYYYMMDD (00 UTC inits)")
    f.add_argument("--leads", required=True, help="comma-separated lead hours; obs valid at date + lead")
    f.add_argument("--period", type=int, default=12, help="accumulation hours")
    f.add_argument("--box", default="10,70,-80,40", help="lat0,lat1,lon0,lon1")
    f.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fetch(a.dates.split(","), [int(x) for x in a.leads.split(",")], a.period, [float(x) for x in a.box.split(",")], a.out)


if __name__ == "__main__":
    main()
