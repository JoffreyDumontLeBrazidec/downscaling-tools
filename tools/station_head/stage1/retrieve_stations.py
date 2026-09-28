"""Retrieve STVL station observations for every valid time of the pairing manifest.

Reproduces retrieve_observations() of
/home/ecm5702/dev/downscaling-tools/eval/_backends/obs_crps/obs_crps_compute.py:
table "observation", forecast length zero, the valid time as the reference time.
Accumulated precipitation needs the `period` key instead (six hours here), as
established in the 2026-09-08 observation inventory.

One cache file per (parameter, valid time) under outputs/cache/, so a crash or a
resubmission loses nothing; the per-parameter parquet files are concatenated at
the end by --finalise.
"""
from __future__ import annotations
import argparse, datetime as dt, json, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path("/home/ecm5702/agent-work/20260909-station-head-adapter")
MANIFEST = ROOT / "outputs" / "pairing_manifest.parquet"
CACHE = ROOT / "outputs" / "cache"
OUTDIR = ROOT / "outputs" / "stations_2026"

# instantaneous parameters, plus precipitation with a six-hour accumulation period
PARAMETERS = {"2t": 0, "2d": 0, "10ff": 0, "msl": 0, "tp": 6}


def retrieve(parameter: str, valid: dt.datetime, period_h: int) -> pd.DataFrame:
    import vtb.media as vmedia
    kw = {}
    if period_h:
        kw["period"] = dt.timedelta(hours=period_h)
    fieldset = vmedia.stvl_retrieve(
        table="observation",
        parameter=parameter,
        reference_datetimes=[valid.strftime("%Y-%m-%dT%H:%M:%S")],
        forecast_lengths=[dt.timedelta(hours=0)],
        **kw,
    )
    if len(fieldset) == 0:
        return pd.DataFrame()
    df = fieldset[0].to_dataframe().rename(columns={"value_0": "value"})
    return df.dropna(subset=["value"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N valid times")
    ap.add_argument("--parameters", default=",".join(PARAMETERS))
    ap.add_argument("--pause", type=float, default=0.0)
    ap.add_argument("--finalise", action="store_true")
    args = ap.parse_args()

    man = pd.read_parquet(MANIFEST).sort_values("synthetic_index").reset_index(drop=True)
    if args.limit:
        man = man.head(args.limit)
    params = [p for p in args.parameters.split(",") if p]
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    failures = []
    t0 = time.time()
    if not args.finalise:
        for row in man.itertuples():
            valid = pd.Timestamp(row.valid).to_pydatetime()
            stamp = valid.strftime("%Y%m%dT%H")
            for p in params:
                cdir = CACHE / p
                cdir.mkdir(parents=True, exist_ok=True)
                cfile = cdir / f"{stamp}.parquet"
                if cfile.exists():
                    continue
                try:
                    df = retrieve(p, valid, PARAMETERS[p])
                except Exception as exc:  # keep going, record it
                    failures.append(dict(parameter=p, valid=valid.isoformat(),
                                         kind="raised", detail=repr(exc)[:300]))
                    print(f"ERROR {p} {valid} {exc!r}"[:300], flush=True)
                    continue
                if df is None or len(df) == 0:
                    failures.append(dict(parameter=p, valid=valid.isoformat(),
                                         kind="empty", detail=""))
                    print(f"EMPTY {p} {valid}", flush=True)
                    df = pd.DataFrame(columns=["latitude", "longitude", "stnid",
                                               "elevation", "value"])
                out = pd.DataFrame({
                    "synthetic_index": row.synthetic_index,
                    "init": pd.Timestamp(row.init),
                    "lead_h": row.lead_h,
                    "valid": pd.Timestamp(valid),
                    "parameter": p,
                    "stnid": df["stnid"].astype(str) if len(df) else pd.Series(dtype=str),
                    "latitude": df["latitude"] if len(df) else pd.Series(dtype=float),
                    "longitude": df["longitude"] if len(df) else pd.Series(dtype=float),
                    "elevation": df["elevation"] if len(df) else pd.Series(dtype=float),
                    "value": df["value"] if len(df) else pd.Series(dtype=float),
                    "period_h": PARAMETERS[p],
                })
                out.to_parquet(cfile, index=False)
                print(f"{p:5s} {valid:%Y-%m-%d %HZ} n={len(out):6d} "
                      f"t={time.time()-t0:8.1f}s", flush=True)
                if args.pause:
                    time.sleep(args.pause)
        fpath = ROOT / "logs" / f"failures_{dt.datetime.utcnow():%Y%m%dT%H%M%S}.json"
        fpath.write_text(json.dumps(failures, indent=1))
        print("failures recorded:", len(failures), "->", fpath, flush=True)

    # concatenate
    for p in params:
        files = sorted((CACHE / p).glob("*.parquet"))
        if not files:
            print("no cache files for", p, flush=True)
            continue
        big = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        big = big.sort_values(["synthetic_index", "stnid"]).reset_index(drop=True)
        dest = OUTDIR / f"{p}.parquet"
        big.to_parquet(dest, index=False)
        print(f"WROTE {dest} rows={len(big)} valid_times={big.valid.nunique()} "
              f"stations={big.stnid.nunique()}", flush=True)
    print("elapsed %.1f s" % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
