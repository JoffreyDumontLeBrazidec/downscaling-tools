"""Reference tracks for tc_obs: the IBTrACS best track and the ECMWF operational tracker (HRES and ENS).

``eval.cli obs fetch-tc`` (Atos) writes a small cache directory that can be copied to any cluster:

  best_track.csv      event, storm_id, time (YYYYMMDDHH), lat, lon (-180..180), pmin_hpa, wind_ms, source
                      from the anemoi IBTrACS zarr (6-hourly, central pressure only, no wind; storms by numeric
                      id) and, when a NOAA IBTrACS CSV path is given, from that file (adds wind_ms, a 10-minute
                      sustained wind in m/s: USA_WIND (1-minute) x 0.93, or WMO_WIND where USA_WIND is missing;
                      wind_src says which).
  tracker_<stream>_<YYYYMMDD>.csv
                      storm, name, member (-1 when the message has no member number), step_h, lat, lon, msl_hpa, wind_ms decoded from MARS
                      (class=od, expver=0001, type=tf, obstype=32, stream=oper|enfo, time=00). In both files the
                      HRES is member 52; the enfo file also carries the 50 perturbed members (1-50) and the
                      control (51). The BUFR files are kept next to them.
  tc_meta.json        storms, dates, sources, cache_version, created.

The NOAA CSV is never downloaded by this code; it is read only when a local path is configured.
"""
from __future__ import annotations

import datetime as dt
import json
import socket
import subprocess
from pathlib import Path

import numpy as np

CACHE_VERSION = 1
KT = 0.514444
USA_1MIN_TO_10MIN = 0.93
NMILE_KM = 1.852
IBTRACS_ZARR = "/ec/ai/project/ai-ml/datasets/dop-file-1980-2025-v1-ibtracs.zarr"
# Column order of the anemoi IBTrACS zarr: __date (days since 1970), __time (s), __latitude, __longitude (0..360),
# storm (numeric id), date_time, obsvalue_pmsl_0 (Pa), then derived columns.
ZARR_COLS = {"date": 0, "time": 1, "lat": 2, "lon": 3, "storm": 4, "pmsl": 6}
HRES_MEMBER = 52


def _wrap(lon):
    lon = np.asarray(lon, np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def read_ibtracs_zarr(path, storms: dict):
    """Rows of the anemoi IBTrACS zarr for ``storms`` = {event name: numeric id}."""
    import pandas as pd
    import zarr

    d = np.asarray(zarr.open(str(path), mode="r")["data"][:], np.float64)
    out = []
    epoch = dt.datetime(1970, 1, 1)
    for event, sid in storms.items():
        r = d[d[:, ZARR_COLS["storm"]] == float(sid)]
        for row in r:
            t = epoch + dt.timedelta(days=float(row[ZARR_COLS["date"]]), seconds=float(row[ZARR_COLS["time"]]))
            out.append({"event": event, "storm_id": str(int(sid)), "time": t.strftime("%Y%m%d%H"),
                        "lat": row[ZARR_COLS["lat"]], "lon": float(_wrap(row[ZARR_COLS["lon"]])),
                        "pmin_hpa": row[ZARR_COLS["pmsl"]] / 100.0, "wind_ms": np.nan, "wind_src": "",
                        "source": "ibtracs_zarr"})
    return pd.DataFrame(out)


def read_noaa_csv(path, events, season: int | None = None, window=None):
    """Best-track rows of a NOAA NCEI IBTrACS v04 CSV (``ibtracs.<basin|last3years|ALL>.list.v04r01.csv``) for
    the storms named like ``events`` (case-insensitive), in ``season`` and/or with ISO_TIME inside ``window``
    (start, end), which is the safer selection: the IBTrACS SEASON is not the calendar year for southern-hemisphere
    storms.

    The file has a row of column names and then a row of units (skipped); a missing value is a single space;
    TRACK_TYPE 'main' rows only (not the provisional or spur tracks); LON is -180..180; times are kept to the
    minute (3-hourly interpolated rows and off-synoptic landfall rows included). Units: winds in knots, RMW in
    nautical miles, pressures in hPa. Output: wind_ms = USA_WIND x 0.93 (1-minute to 10-minute sustained at
    sea, WMO guidance) x 0.514444, or WMO_WIND x 0.514444 where USA_WIND is missing (``wind_src`` says which);
    pmin_hpa = USA_PRES, or WMO_PRES where missing; rmw_km = USA_RMW x 1.852. A name that matches more than
    one storm (SID) in the season is an error."""
    import pandas as pd

    x = pd.read_csv(path, skiprows=[1], low_memory=False, keep_default_na=False, dtype=str)
    x["NAME"] = x["NAME"].astype(str).str.strip().str.lower()
    x = x[x.NAME.isin([e.lower() for e in events])]
    if season is not None:
        x = x[pd.to_numeric(x["SEASON"], errors="coerce") == season]
    if window is not None:
        t = pd.to_datetime(x["ISO_TIME"].str.strip())
        x = x[(t >= pd.Timestamp(window[0])) & (t <= pd.Timestamp(window[1]))]
    if "TRACK_TYPE" in x:
        x = x[x["TRACK_TYPE"].str.strip() == "main"]
    sids = x.groupby("NAME")["SID"].nunique()
    if (sids > 1).any():
        raise ValueError(f"IBTrACS names matching several storms (season {season}, window {window}): "
                         f"{x[x.NAME.isin(sids[sids > 1].index)].groupby('NAME').SID.unique().to_dict()}")

    def num(col):
        return pd.to_numeric(x[col].str.strip(), errors="coerce") if col in x else pd.Series(np.nan, index=x.index)

    usa, wmo = num("USA_WIND"), num("WMO_WIND")
    # WMO_WIND reported by HURDAT (NHC) is also a 1-minute wind: the same 0.93 applies to it
    hurdat = x["WMO_AGENCY"].str.strip().str.lower().str.startswith("hurdat") if "WMO_AGENCY" in x else \
        pd.Series(False, index=x.index)
    wmo10 = wmo.where(~hurdat, wmo * USA_1MIN_TO_10MIN)
    wind = (usa * USA_1MIN_TO_10MIN).where(usa.notna(), wmo10) * KT
    wind_src = np.where(usa.notna(), "usa_wind_1min_x0.93",
                        np.where(wmo.notna() & hurdat, "wmo_wind_hurdat_1min_x0.93",
                                 np.where(wmo.notna(), "wmo_wind", "")))
    pres = num("USA_PRES").fillna(num("WMO_PRES"))
    t = pd.to_datetime(x["ISO_TIME"].str.strip())
    return pd.DataFrame({"event": x.NAME, "storm_id": x["SID"].astype(str), "time": t.dt.strftime("%Y%m%d%H%M"),
                         "lat": num("LAT"), "lon": _wrap(num("LON")), "pmin_hpa": pres, "wind_ms": wind,
                         "wind_src": wind_src, "rmw_km": num("USA_RMW") * NMILE_KM,
                         "source": "noaa_ibtracs_csv"}).reset_index(drop=True)


def decode_tracker(bufr_path):
    """One row per storm, member and step from an ECMWF tracker BUFR file (compressed ENS messages hold one
    subset per member). lat/lon are the centre, msl the central pressure (hPa), wind the 10 m maximum (m/s)."""
    import eccodes as ec
    import pandas as pd

    def bad(v):
        return (not np.isfinite(v)) or abs(v) > 1e9

    rows = []
    with open(bufr_path, "rb") as f:
        while True:
            h = ec.codes_bufr_new_from_file(f)
            if h is None:
                break
            ec.codes_set(h, "unpack", 1)
            nsub = ec.codes_get(h, "numberOfSubsets")

            def arr(key):
                try:
                    v = np.asarray(ec.codes_get_array(h, key), np.float64)
                except ec.KeyValueNotFoundError:
                    return None
                return np.repeat(v, nsub) if v.size == 1 else v

            def strs(key):
                try:
                    v = ec.codes_get_string_array(h, key)
                except Exception:  # noqa: BLE001 - scalar string in uncompressed messages
                    v = [ec.codes_get_string(h, key)]
                v = [s.strip() for s in v]
                return v * nsub if len(v) == 1 else v

            def occurrences(name):
                out, k = [], 1
                while ec.codes_is_defined(h, f"#{k}#{name}"):
                    out.append(arr(f"#{k}#{name}")); k += 1
                return out

            sid, name = strs("stormIdentifier"), strs("longStormName")
            mem = arr("ensembleMemberNumber")
            lat, lon = occurrences("latitude"), occurrences("longitude")
            msl, ws, tp = (occurrences("pressureReducedToMeanSeaLevel"), occurrences("windSpeedAt10M"),
                           occurrences("timePeriod"))
            # latitude/longitude per subset: [initial position, analysis centre, analysis max-wind point, then per
            # time step: centre, max-wind point]; msl and wind: one value per step including the analysis
            for j in range(nsub):
                steps = [0] + [int(t[j]) for t in tp if not bad(t[j])]
                for i in range(len(msl)):
                    c = 1 + 2 * i
                    rows.append(dict(storm=sid[j], name=name[j], member=int(mem[j]) if mem is not None else -1,
                                     step_h=steps[i] if i < len(steps) else -1,
                                     lat=np.nan if c >= len(lat) or bad(lat[c][j]) else lat[c][j],
                                     lon=np.nan if c >= len(lon) or bad(lon[c][j]) else float(_wrap(lon[c][j])),
                                     msl_hpa=np.nan if bad(msl[i][j]) else msl[i][j] / 100,
                                     wind_ms=np.nan if i >= len(ws) or bad(ws[i][j]) else ws[i][j]))
            ec.codes_release(h)
    cols = ["storm", "name", "member", "step_h", "lat", "lon", "msl_hpa", "wind_ms"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).dropna(subset=["msl_hpa"], how="all")


def retrieve_tracker(stream, date, target):
    req = (f"retrieve, class=od, expver=0001, stream={stream}, type=tf, date={date}, time=00, obstype=32, "
           f"target=\"{target}\"\n")
    subprocess.run(["mars"], input=req, text=True, check=True)


def fetch_tc(storms: dict, dates, out, *, zarr_path=IBTRACS_ZARR, noaa_csv=None, streams=("oper", "enfo")):
    """Write the tc_obs reference cache (see the module docstring). Reads only; MARS for the tracker BUFR."""
    import pandas as pd

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    log = []
    bt = [read_ibtracs_zarr(zarr_path, storms)]
    if noaa_csv:
        t0 = pd.Timestamp(str(min(dates)))
        bt.append(read_noaa_csv(noaa_csv, list(storms), window=(t0 - pd.Timedelta(days=1),
                                                              pd.Timestamp(str(max(dates))) + pd.Timedelta(days=11))))
    bt = pd.concat(bt, ignore_index=True)
    bt.to_csv(out / "best_track.csv", index=False, float_format="%.4f")
    log.append(f"best track: {len(bt)} rows for {sorted(set(bt.event))}")
    files, not_found = {}, []
    for date in dates:
        for stream in streams:
            bufr = out / f"tf_{stream}_{date}.bufr"
            if not bufr.exists():
                retrieve_tracker(stream, date, bufr)
            df = decode_tracker(bufr)
            path = out / f"tracker_{stream}_{date}.csv"
            df.to_csv(path, index=False)
            files[path.name] = int(len(df))
            if df.empty:
                not_found.append(f"{stream}_{date}")
            log.append(f"tracker {stream} {date}: {len(df)} rows, {df.storm.nunique()} storms")
            print(log[-1], flush=True)
    meta = {"cache_version": CACHE_VERSION, "kind": "tropical cyclone references (tc_obs)",
            "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "host": socket.gethostname(),
            "storms": {k: int(v) for k, v in storms.items()}, "tracker_not_found": not_found, "dates": list(dates), "streams": list(streams),
            "ibtracs_zarr": str(zarr_path), "noaa_csv": str(noaa_csv) if noaa_csv else None,
            "tracker_request": "class=od, expver=0001, type=tf, obstype=32, time=00", "rows": files,
            "hres_member": HRES_MEMBER, "log": log}
    (out / "tc_meta.json").write_text(json.dumps(meta, indent=2))
    return out


def best_track_at(bt, event, when, *, max_gap_h: float = 6.0):
    """Best-track position, pressure, wind and RMW of ``event`` at ``when`` (datetime): exact fix, or linear
    interpolation in time between the two fixes around it when they are at most ``max_gap_h`` apart. Only fixes
    with a finite pressure and position count. The NOAA rows are used first; the anemoi zarr rows only when the
    NOAA rows give nothing, and then ``noaa_incomplete`` is True if the NOAA file had a fix there without
    pressure or position. Returns a dict (with ``source``) or None."""
    b = bt[bt.event.str.lower() == event.lower()]
    if b.empty:
        return None
    sources = ["noaa_ibtracs_csv"] + sorted(set(b.source) - {"noaa_ibtracs_csv"}) if "source" in b else [None]
    incomplete = False
    for src in sources:
        bs = b if src is None else b[b.source == src]
        if bs.empty:
            continue
        ok = np.isfinite(bs.pmin_hpa.astype(float)) & np.isfinite(bs.lat.astype(float)) & np.isfinite(bs.lon.astype(float))
        r = _fix_at(bs[ok], when, max_gap_h)
        if r is not None:
            return {**r, "source": src, "noaa_incomplete": incomplete}
        if src == "noaa_ibtracs_csv" and (~ok).any() and _fix_at(bs, when, max_gap_h) is not None:
            incomplete = True
    return None


def _fix_at(b, when, max_gap_h):
    import pandas as pd

    if b.empty:
        return None
    t = pd.to_datetime(b.time.astype(str).str.ljust(12, "0"), format="%Y%m%d%H%M")  # YYYYMMDDHH or ...HHMM
    b = b.assign(_t=t).sort_values("_t").drop_duplicates("_t")

    def get(r, k):
        return r[k] if k in r and r[k] is not None else np.nan

    exact = b[b._t == when]
    if len(exact):
        r = exact.iloc[0]
        return {"lat": r.lat, "lon": r.lon, "pmin_hpa": r.pmin_hpa, "wind_ms": get(r, "wind_ms"),
                "wind_src": r.get("wind_src", ""), "rmw_km": get(r, "rmw_km"), "interpolated": False}
    before, after = b[b._t < when], b[b._t > when]
    if before.empty or after.empty:
        return None
    r0, r1 = before.iloc[-1], after.iloc[0]
    gap = (r1._t - r0._t).total_seconds() / 3600.0
    if gap > max_gap_h:
        return None
    w = (when - r0._t).total_seconds() / 3600.0 / gap
    dlon = ((r1.lon - r0.lon + 180.0) % 360.0) - 180.0
    lin = lambda k: get(r0, k) + w * (get(r1, k) - get(r0, k))  # noqa: E731
    return {"lat": r0.lat + w * (r1.lat - r0.lat), "lon": float(_wrap(r0.lon + w * dlon)), "pmin_hpa": lin("pmin_hpa"),
            "wind_ms": lin("wind_ms"), "wind_src": r0.get("wind_src", ""), "rmw_km": lin("rmw_km"),
            "interpolated": True}


def load_cache(cache_dir):
    """(best track, {(stream, date): tracker frame}, meta) from a tc_obs cache directory."""
    import pandas as pd

    cache_dir = Path(cache_dir)
    bt = pd.read_csv(cache_dir / "best_track.csv", dtype={"time": str, "storm_id": str, "wind_src": str})
    trackers = {}
    for p in sorted(cache_dir.glob("tracker_*_*.csv")):
        _, stream, date = p.stem.split("_")
        trackers[(stream, date)] = pd.read_csv(p)
    meta_path = cache_dir / "tc_meta.json"
    return bt, trackers, (json.loads(meta_path.read_text()) if meta_path.exists() else {})
