"""Build the pairing manifest from the fake_forecasts attribute of the AIFS ENS v2
validtime store.  One row per synthetic index of the store axis."""
import datetime as dt
import json
import numpy as np
import pandas as pd
import zarr

ZPATH = "/home/ecm5702/scratch/data/anemoi_datasets_aifsens2_20260907/downscaling-od-an-oper-0001-mars-o1280-2026-2026-12h-6h-v1-aifsens2-validtime.zarr"
OUT = "/home/ecm5702/agent-work/20260909-station-head-adapter/outputs/pairing_manifest.parquet"

attrs = json.load(open(ZPATH + "/.zattrs"))
ff = attrs["fake_forecasts"]

g = zarr.open(ZPATH, mode="r")
dates = np.asarray(g["dates"][:])
print("store dates:", dates.shape, dates.dtype, dates[0], dates[-1])
dates_str = [pd.Timestamp(d).strftime("%Y-%m-%dT%H:%M:%S") for d in dates]

# the store axis order is the order of the dates array; fake_forecasts is keyed by
# the same synthetic dates.
missing = [d for d in dates_str if d not in ff]
assert not missing, "synthetic dates on the axis with no fake_forecasts entry: %s" % missing[:5]
assert len(dates_str) == len(ff), (len(dates_str), len(ff))

rows = []
for i, sd in enumerate(dates_str):
    init_s, lead = ff[sd]
    init = dt.datetime.fromisoformat(init_s)
    rows.append(dict(synthetic_index=i, synthetic_date=sd, init=init,
                     lead_h=int(lead), valid=init + dt.timedelta(hours=int(lead))))
df = pd.DataFrame(rows)

assert len(df) == 446, len(df)
assert df["init"].nunique() == 223, df["init"].nunique()
assert sorted(df["lead_h"].unique()) == [6, 12], df["lead_h"].unique()
assert df["valid"].is_unique, "valid times are not unique"
print("rows", len(df), "inits", df["init"].nunique(), "leads", sorted(df.lead_h.unique()))
print("init range", df["init"].min(), "->", df["init"].max())
print("valid range", df["valid"].min(), "->", df["valid"].max())
vs = df["valid"].sort_values()
gaps = vs.diff().dropna().value_counts()
print("valid-time spacing:"); print(gaps)
df.to_parquet(OUT, index=False)
print("written", OUT)
print(df.head(4).to_string())
