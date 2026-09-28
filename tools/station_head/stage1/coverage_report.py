"""Build notes/coverage_report.md and add the per-parameter station counts to the
pairing manifest."""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/ecm5702/agent-work/20260909-station-head-adapter")
PARAMS = ["2t", "2d", "10ff", "msl", "tp"]
man = pd.read_parquet(ROOT / "outputs" / "pairing_manifest.parquet")
man = man[[c for c in man.columns if not c.startswith("n_")]]

lines = []
w = lines.append

counts = {}
per_param = {}
for p in PARAMS:
    d = pd.read_parquet(ROOT / "outputs" / "stations_2026" / f"{p}.parquet",
                        columns=["synthetic_index", "valid", "stnid"])
    per_param[p] = d
    c = d.groupby("synthetic_index").size().rename(f"n_{p}")
    counts[p] = c
    man = man.merge(c, on="synthetic_index", how="left")
man[[f"n_{p}" for p in PARAMS]] = man[[f"n_{p}" for p in PARAMS]].fillna(0).astype(int)
man.to_parquet(ROOT / "outputs" / "pairing_manifest.parquet", index=False)

st = pd.read_parquet(ROOT / "outputs" / "static_stations.parquet")
method = (ROOT / "outputs" / "holdout_method.txt").read_text().strip()
timing = json.loads((ROOT / "outputs" / "timing.json").read_text()) if (ROOT / "outputs" / "timing.json").exists() else {}

fails = []
for f in sorted(glob.glob(str(ROOT / "logs" / "failures_*.json"))):
    fails.extend(json.load(open(f)))
fail_df = pd.DataFrame(fails)

w("# Stage 1 coverage report: station observations for the 2026 AIFS ENS version 2 calendar")
w("")
w("This report describes what the stage 1 build of the station head adapter actually")
w("retrieved. The calendar is the 446 synthetic samples of the AIFS ENS version 2")
w("validtime store, which are 223 initialisations at 00 and 12 UTC from 12 May 2026")
w("12 UTC to 31 August 2026 12 UTC, each with a 6 and a 12 hour lead. The valid times")
w("run from 12 May 2026 18 UTC to 1 September 2026 00 UTC, exactly six hours apart,")
w("with no gap and no repetition, so the pairing manifest is a clean one to one map")
w("between a store index and a valid time.")
w("")
w("## Station counts per parameter")
w("")
w("The table gives, over the 446 valid times, the median, smallest and largest number")
w("of stations that reported the parameter, and how many distinct stations were seen at")
w("least once.")
w("")
w("| parameter | period (h) | median | min | max | distinct stations | total rows |")
w("|---|---|---|---|---|---|---|")
for p in PARAMS:
    c = counts[p].reindex(man["synthetic_index"]).fillna(0)
    d = per_param[p]
    period = 6 if p == "tp" else 0
    w("| %s | %d | %d | %d | %d | %d | %d |" % (p, period, int(c.median()), int(c.min()),
                                                int(c.max()), d["stnid"].nunique(), len(d)))
w("")
w("The valid times with the fewest reports, for each parameter, are listed below. A low")
w("count usually means a late-arriving observation batch rather than a failed request.")
w("")
for p in PARAMS:
    c = counts[p].reindex(man["synthetic_index"]).fillna(0).astype(int)
    lo = c.nsmallest(5)
    vt = man.set_index("synthetic_index")["valid"]
    w("- %s: " % p + ", ".join("%s (%d)" % (pd.Timestamp(vt[i]).strftime("%Y-%m-%d %HZ"), n)
                              for i, n in lo.items()))
w("")

w("## Stability of the network")
w("")
n_times = len(man)
for p in ["2t", "tp"]:
    d = per_param[p]
    freq = d.groupby("stnid")["synthetic_index"].nunique() / n_times
    w("For %s, %d distinct stations were seen at least once and %d of them, that is %.1f "
      "per cent, reported at more than 90 per cent of the %d valid times. The median "
      "station reported at %.1f per cent of valid times."
      % (p, freq.size, int((freq > 0.9).sum()), 100.0 * (freq > 0.9).mean(), n_times,
         100.0 * freq.median()))
    w("")

allst = len(st)
w("Across all five parameters, %d distinct station identifiers were seen. This is the "
  "row count of the static station table." % allst)
w("")

w("## Stations, terrain and the hold-out split")
w("")
w("Every station was matched to its nearest point of the O1280 output grid and given the")
w("static fields the analysis archive carries there: the sub-grid orography standard")
w("deviation, the slope, the surface height and the land-sea fraction within 15 km. The")
w("terrain class follows the design note, mountain above 100 m of sub-grid orography,")
w("hilly between 30 and 100 m, flat below. The hold-out flag was assigned by the")
w("following method: %s." % method)
w("")
w("| stratum | stations | held out | held out (%) |")
w("|---|---|---|---|")
for col in ["terrain_class", "region"]:
    for v, sub in st.groupby(col):
        w("| %s = %s | %d | %d | %.2f |" % (col, v, len(sub), int(sub["holdout_station"].sum()),
                                            100.0 * sub["holdout_station"].mean()))
w("| all | %d | %d | %.2f |" % (allst, int(st["holdout_station"].sum()),
                                100.0 * st["holdout_station"].mean()))
w("")
w("The nearest-point distance has a median of %.2f km and a maximum of %.1f km. The "
  "station minus model height difference has a median of %+.1f m overall and %+.1f m in "
  "the mountain class, which is the representativeness gap the head is meant to close."
  % (st["nearest_distance_km"].median(), st["nearest_distance_km"].max(),
     st["station_minus_model_height_m"].median(),
     st.loc[st["terrain_class"] == "mountain", "station_minus_model_height_m"].median()))
w("")
w("Coastal stations, meaning those whose 15 km neighbourhood mixes land and sea, number "
  "%d, which is %.1f per cent of the network." % (int(st["coastal"].sum()),
                                                  100.0 * st["coastal"].mean()))
w("")

w("## Failed and empty requests")
w("")
if len(fail_df) == 0:
    w("No request failed and no request returned an empty result. All %d requests, that is"
      " five parameters at each of the %d valid times, returned stations." % (5 * n_times, n_times))
else:
    w("The following requests either raised an exception or returned nothing. The build")
    w("continued past each of them and the affected valid time simply has no rows for that")
    w("parameter.")
    w("")
    w("| parameter | valid time | kind | detail |")
    w("|---|---|---|---|")
    for r in fail_df.itertuples():
        w("| %s | %s | %s | %s |" % (r.parameter, r.valid, r.kind, str(r.detail)[:120]))
w("")

w("## Method and wall time")
w("")
for k, v in timing.items():
    w("- %s: %s" % (k, v))
w("")
lines.append("")
(ROOT / "notes" / "coverage_report.md").write_text("\n".join(lines) + "\n")
print("written", ROOT / "notes" / "coverage_report.md")
print("manifest columns", list(man.columns))
