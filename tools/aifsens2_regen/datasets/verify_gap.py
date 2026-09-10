"""Verify a fake_forecasts store that declares a missing real forecast start.

The script checks, on the store given as the first argument:
  1. the synthetic axis is regular and has one date per (start, lead) pair, with no hole;
  2. the synthetic dates of every start declared in missing_starts are listed in missing_dates;
  3. every other sample is present and finite;
  4. one (start, lead, member) sample is bit identical to the source O320 GRIB file;
  5. open_dataset opens the store, reports the missing dates, refuses to read them, and
     skip_missing_dates drops the samples that touch them.
"""

import datetime
import sys

import numpy as np
import zarr

from anemoi.datasets import open_dataset

PATH = sys.argv[1]
GRIB_DIR = sys.argv[2] if len(sys.argv) > 2 else (
    "/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/derived_o320"
)

failures = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" : " + str(detail)) if detail else ""))
    if not ok:
        failures.append(name)


z = zarr.open(PATH, "r")
attrs = dict(z.attrs)

fake = attrs["fake_forecasts"]
missing_starts = attrs.get("fake_forecasts_missing_starts", [])
missing_dates = attrs.get("missing_dates", [])

print("store            :", PATH)
print("n synthetic dates:", len(fake))
print("missing_starts   :", missing_starts)
print("missing_dates    :", missing_dates)


def as_dt(x):
    if isinstance(x, str):
        return datetime.datetime.fromisoformat(x)
    return x


mapping = {}
for k, v in fake.items():
    start, step = v[0], v[1]
    mapping[as_dt(k)] = (as_dt(start), int(step))

synth = sorted(mapping)
starts = sorted({s for s, _ in mapping.values()})
steps = sorted({t for _, t in mapping.values()})

# 1. regular axis, no hole
deltas = {(b - a) for a, b in zip(synth, synth[1:])}
check(
    "axis is regular hourly with one date per (start, lead)",
    deltas == {datetime.timedelta(hours=1)} and len(synth) == len(starts) * len(steps),
    f"{len(synth)} dates = {len(starts)} starts x {len(steps)} leads, deltas={deltas}",
)

# 2. the missing start contributes exactly its synthetic dates to missing_dates
expected_missing = sorted(d for d, (s, _) in mapping.items() if s in {as_dt(m) for m in missing_starts})
check(
    "missing_dates are exactly the synthetic dates of the missing starts",
    sorted(as_dt(m) for m in missing_dates) == expected_missing,
    f"expected {[d.isoformat() for d in expected_missing]}",
)

# 3. present samples finite, missing samples not read
ds = open_dataset(PATH)
print("open_dataset shape:", ds.shape, "variables:", len(ds.variables))
check("open_dataset reports the missing indices", len(ds.missing) == len(expected_missing), sorted(ds.missing))

missing_idx = sorted(ds.missing)
for i in range(len(ds)):
    if i in missing_idx:
        continue
    arr = ds[i]
    if not np.isfinite(arr).all():
        check(f"sample {i} ({synth[i].isoformat()}) is finite", False)
        break
else:
    check(f"the {len(ds) - len(missing_idx)} present samples are finite", True)

from anemoi.datasets.data import MissingDateError  # noqa: E402

try:
    _ = ds[missing_idx[0]]
    check("reading a missing date raises MissingDateError", False, "no exception")
except MissingDateError as e:
    check("reading a missing date raises MissingDateError", True, str(e))

# 5. skip_missing_dates, the option a training config must use
n_lead = len(steps)
skipped = open_dataset(PATH, skip_missing_dates=True, expected_access=slice(0, n_lead))
expected_windows = sum(
    1
    for i in range(len(ds) - n_lead + 1)
    if not set(range(i, i + n_lead)) & set(missing_idx)
)
check(
    "skip_missing_dates keeps only the windows that avoid the missing dates",
    len(skipped) == expected_windows,
    f"len(skip_missing_dates)={len(skipped)}, expected {expected_windows}, len(ds)={len(ds)}",
)

# 4. bit identity against the source GRIB, for the first present sample
import earthkit.data as ekd  # noqa: E402

idx = next(i for i in range(len(ds)) if i not in missing_idx)
start, step = mapping[synth[idx]]
grib = f"{GRIB_DIR}/{start.strftime('%Y%m%d_%H')}.grib"
print(f"bit identity check on start={start} step={step} file={grib}")

variables = list(ds.variables)
name = variables[0]
if "_" in name and name.split("_")[-1].isdigit():
    param, level = name.rsplit("_", 1)
    sel = dict(param=param, level=int(level))
else:
    param, level = name, None
    sel = dict(param=param)
member = 1

fs = ekd.from_source("file", grib).sel(step=step, number=member, **sel)
check(f"exactly one GRIB field for {name} step={step} number={member}", len(fs) == 1, len(fs))
if len(fs) == 1:
    ref = fs[0].to_numpy(flatten=True).astype(np.float32)
    got = np.asarray(ds[idx][variables.index(name), member - 1, :], dtype=np.float32)
    check(
        f"store sample is bit identical to the GRIB for {name} start={start} step={step} member={member}",
        ref.shape == got.shape and np.array_equal(ref, got),
        f"shapes {ref.shape} {got.shape}, max abs diff "
        f"{np.abs(ref - got).max() if ref.shape == got.shape else 'n/a'}",
    )

print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
