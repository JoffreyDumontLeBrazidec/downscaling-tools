"""Summary of a tc_intensity TSV in the format of notes/tc_intensity_stage1a_summary.txt.
Usage: python tc_intensity_summary.py <tsv> [arm ...]  (truth first if present, then the arms in order)."""
import collections
import csv
import sys

import numpy as np

rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
g = collections.defaultdict(list)
for r in rows:
    g[(r["arm"], r["storm"], int(r["lead"]))].append((float(r["wind_max"]), float(r["mslp_min"])))
arms = sys.argv[2:] or sorted({r["arm"] for r in rows})
order = (["truth"] if any(r["arm"] == "truth" for r in rows) else []) + [a for a in arms if a != "truth"]
print("storm lead | arm | n | mean wind_max | p90 wind_max | mean mslp_min | p10 mslp_min")
for s in ("idalia", "franklin"):
    for L in (24, 120):
        for a in order:
            v = np.array(g[(a, s, L)])
            print(f"{s:8s} {L:3d} | {a:10s} | {len(v):3d} | {v[:, 0].mean():6.2f} | {np.percentile(v[:, 0], 90):6.2f} | "
                  f"{v[:, 1].mean():7.2f} | {np.percentile(v[:, 1], 10):7.2f}")
