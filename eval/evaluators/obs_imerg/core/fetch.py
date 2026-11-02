"""IMERG Final V07B 12 h precipitation on the E2 points, cut from the O1280 6-hourly store for obs_imerg.
Atos (or ac6-100) only, reads only.

    python3 -m eval.cli obs fetch-imerg --dates 20230101,20230111 --leads 12,24,36,48,60,72 --out imerg_e2_tp12.nc

The store (``IMERG_ZARR``) is on our O1280 grid and point order; its value at t is the sum over (t - 6 h, t],
in metres. A 12 h window ending at valid time v is the sum of the steps v - 6 h and v. The E2 cut is every
O1280 point with 10 <= lat <= 70 and -80 <= lon <= 40, in the global order: 814,810 points, the same points
in the same order as the E2-v4 target and the prediction files (obs_imerg checks the coordinates).

Writes one netCDF file: tp (valid, point) in mm per window (float32, NaN where IMERG has none), lat, lon,
valid (YYYYMMDDHH strings), attributes with the store, the box, the window and the creation time.
"""
from __future__ import annotations

import datetime as dt
import socket
from pathlib import Path

import numpy as np

IMERG_ZARR = "/ec/ai/project/ai-ml/datasets/aifs-imerg-final-v07b-nasa-o1280-1998-2025-6h-v1-observations.zarr"
E2_BOX = (10.0, 70.0, -80.0, 40.0)
STEP_H = 6


def e2_index(lat, lon, box=E2_BOX):
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return np.flatnonzero((lat >= box[0]) & (lat <= box[1]) & (lon >= box[2]) & (lon <= box[3]))


def valid_times(dates, leads):
    return sorted({dt.datetime.strptime(d, "%Y%m%d") + dt.timedelta(hours=int(h)) for d in dates for h in leads})


def fetch_imerg(dates, leads, out, *, accum_h=12, zarr_path=IMERG_ZARR, box=E2_BOX):
    import netCDF4
    import zarr

    g = zarr.open(str(zarr_path), mode="r")
    times = np.asarray(g["dates"][:]).astype("datetime64[s]").astype(dt.datetime)
    pos = {t: i for i, t in enumerate(times)}
    lat, lon = np.asarray(g["latitudes"][:]), np.asarray(g["longitudes"][:])
    idx = e2_index(lat, lon, box)
    valids = valid_times(dates, leads)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    n_steps = accum_h // STEP_H
    cache = {}

    def step(t):
        if t not in cache:
            if t not in pos:
                raise KeyError(f"IMERG store has no step {t:%Y-%m-%d %H} UTC")
            cache[t] = np.asarray(g["data"][pos[t], 0, 0, :], np.float32)[idx]
        return cache[t]

    with netCDF4.Dataset(str(out), "w") as ds:
        ds.createDimension("valid", len(valids))
        ds.createDimension("point", idx.size)
        ds.createVariable("lat", "f8", ("point",))[:] = lat[idx]
        ds.createVariable("lon", "f8", ("point",))[:] = np.where(lon[idx] > 180, lon[idx] - 360, lon[idx])
        ds.createVariable("index_o1280", "i4", ("point",))[:] = idx
        v = ds.createVariable("valid", str, ("valid",))
        tp = ds.createVariable("tp", "f4", ("valid", "point"), zlib=True, complevel=4, chunksizes=(1, idx.size))
        tp.units = f"mm per {accum_h} h, ending at valid"
        for k, t in enumerate(valids):
            acc = sum(step(t - dt.timedelta(hours=STEP_H * j)) for j in range(n_steps)) * 1000.0
            v[k] = t.strftime("%Y%m%d%H")
            tp[k, :] = acc
            print(f"IMERG {t:%Y%m%d%H}: {np.isfinite(acc).mean():.4f} finite, max {np.nanmax(acc):.1f} mm", flush=True)
            for old in [s for s in cache if s <= t - dt.timedelta(hours=STEP_H * n_steps)]:
                cache.pop(old)
        ds.source = f"{zarr_path} (IMERG Final V07B on O1280, 6 h sums in metres)"
        ds.box = list(box)
        ds.accum_hours = accum_h
        ds.dates = ",".join(dates)
        ds.leads = ",".join(str(x) for x in leads)
        ds.created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + " on " + socket.gethostname()
    return out


def add_args(ap):
    ap.add_argument("--dates", required=True, help="comma-separated YYYYMMDD (00 UTC runs)")
    ap.add_argument("--leads", default="12,24,36,48,60,72")
    ap.add_argument("--accum", type=int, default=12, help="window in hours ending at valid time (multiple of 6)")
    ap.add_argument("--box", default=",".join(f"{x:g}" for x in E2_BOX), help="lat0,lat1,lon0,lon1, inclusive")
    ap.add_argument("--zarr", default=IMERG_ZARR, help="IMERG O1280 6-hourly store")
    ap.add_argument("--out", required=True, help="output netCDF file")


def run_from_args(a):
    if not Path(a.zarr).exists():
        raise SystemExit(f"obs fetch-imerg: no IMERG store at {a.zarr} (it is on Atos; run there or on ac6-100)")
    if a.accum % STEP_H:
        raise SystemExit(f"--accum must be a multiple of {STEP_H} h")
    return fetch_imerg(a.dates.split(","), [int(x) for x in a.leads.split(",")], a.out, accum_h=a.accum,
                       zarr_path=a.zarr, box=tuple(float(x) for x in a.box.split(",")))
