"""Independent gates on the early-2026 target and forcings stores, without the input store.

verify_early.py opens all three stores together and cannot run until the forecast input store
exists. These are the checks that do not need it: the shared synthetic axis, its origin, that the
declared gap of the lost 2026-01-15 12 UTC start is exactly the two synthetic dates of that start,
the ensemble size, absence of NaN, and that a sample reads and is finite.
"""
import sys

import numpy as np
import zarr
from anemoi.datasets import open_dataset

D = "/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/training"
STORES = [
    ("target", "downscaling-od-an-oper-0001-mars-o1280-2026-2026-12h-6h-v1-aifsens2-early-validtime"),
    ("forcings", "downscaling-od-an-oper-0001-mars-o1280-2026-2026-12h-6h-v1-aifsens2-early-validtime-forcings"),
]
LOST = "2026-01-15T12:00:00"

ok = True


def chk(cond, msg):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + msg, flush=True)


maps = {}
for tag, name in STORES:
    path = f"{D}/{name}.zarr"
    z = zarr.open(path, mode="r")
    attrs = dict(z.attrs)
    ds = open_dataset(path)
    fake = attrs["fake_forecasts"]
    maps[tag] = fake
    missing = sorted(attrs.get("missing_dates", []))
    expected = sorted(k for k, v in fake.items() if v[0] == LOST)

    print(f"--- {tag}: shape={ds.shape} synthetic_dates={len(fake)} missing={missing}")
    chk(ds.shape[0] == 526, f"{tag} has 526 synthetic dates for 263 starts at two leads: got {ds.shape[0]}")
    chk(str(ds.dates[0]).startswith("1900-01-19T14"), f"{tag} axis starts at 1900-01-19T14: got {ds.dates[0]}")
    deltas = {(ds.dates[i + 1] - ds.dates[i]) for i in range(len(ds.dates) - 1)}
    chk(len(deltas) == 1, f"{tag} axis is regular with one step: {deltas}")
    chk(len(expected) == 2, f"{tag} the lost start maps to two synthetic dates: {expected}")
    chk(missing == expected, f"{tag} missing_dates are exactly those two")
    chk(attrs.get("variables_with_nans", []) == [], f"{tag} no variable recorded as holding NaN: {attrs.get('variables_with_nans')}")
    chk(ds.shape[2] == 1, f"{tag} ensemble size is 1: got {ds.shape[2]}")

    present = [i for i in range(ds.shape[0]) if i not in set(ds.missing)]
    chk(len(present) == 524, f"{tag} 524 of 526 samples present: got {len(present)}")
    for i in (present[0], present[len(present) // 2], present[-1]):
        chk(np.isfinite(ds[i]).all(), f"{tag} sample {i} is finite")

chk(maps["target"] == maps["forcings"], "target and forcings share an identical synthetic mapping")

print("TARGET_FORCINGS_RC=" + ("0" if ok else "1"))
sys.exit(0 if ok else 1)
