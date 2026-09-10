"""Acceptance gates for the three early-2026 AIFS ENS version 2 stores.

This is the early-period counterpart of
/home/ecm5702/hpcperm/sandbox/20260905-aifs-analysis-target/verify_full.py.

It checks, in order:

  1. the three stores carry an identical fake_forecasts mapping;
  2. the synthetic axis has the expected number of dates, starts at the expected origin, and
     advances by exactly one hour;
  3. the ensemble dimensions are 10 (input) / 1 (target) / 1 (forcings);
  4. the input and target carry 68 variables in the same order as the summer stores;
  5. every variable is finite: the store-wide variables_with_nans attribute is empty, and a
     spot check over a few dates reports the NaN count per variable;
  6. one target sample and one forcing sample equal the O1280 analysis at the sample valid time,
     retrieved directly from MARS in this script;
  7. one input sample equals, to the bit, the corresponding message of the source O320 GRIB file
     for that (start, step, member);
  8. the 12 May 00 UTC boundary start is present with both leads (full period only).

Everything is parameterised, so the same script also validates a block subset (January alone, or
the two-start fixture) by passing --suffix, --expect-dates and --grib-dir.

Usage examples:
  python verify_early.py --dir /home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/training
  python verify_early.py --dir <test area>/stores --suffix -test --expect-dates 4 \
      --grib-dir <test area>/derived_o320 --allow-identical-members --no-boundary
"""

import argparse
import datetime
import json
import subprocess
import sys
import tempfile

import numpy as np
from anemoi.datasets import open_dataset

SUMMER_DIR = "/home/ecm5702/scratch/data/anemoi_datasets_aifsens2_20260907"
SUMMER_IN = f"{SUMMER_DIR}/downscaling-ai-pf-enfo-0001-mars-o320-2026-2026-12h-6h-v1-aifsens2.zarr"

IN_STEM = "downscaling-ai-pf-enfo-rgn2-mars-o320-2026-2026-12h-6h-v1-aifsens2-early"
TG_STEM = "downscaling-od-an-oper-0001-mars-o1280-2026-2026-12h-6h-v1-aifsens2-early-validtime"
FO_STEM = TG_STEM + "-forcings"

OK = True


def check(condition, message):
    global OK
    print(("PASS " if condition else "FAIL ") + message, flush=True)
    OK = OK and bool(condition)


def mars_field(request):
    """Retrieve a single field with MARS and return its values as a flat array."""
    import eccodes as ec

    with tempfile.TemporaryDirectory() as td:
        target = f"{td}/f.grib"
        subprocess.run(
            ["mars"], input=(request + f', target="{target}"').encode(), check=True, capture_output=True
        )
        with open(target, "rb") as f:
            handle = ec.codes_grib_new_from_file(f)
            values = ec.codes_get_values(handle)
            ec.codes_release(handle)
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="directory holding the three .zarr stores")
    p.add_argument("--suffix", default="", help="suffix appended to the three store names")
    p.add_argument("--expect-dates", type=int, default=526)
    p.add_argument("--origin", default="1900-01-19T14:00:00")
    p.add_argument("--members", type=int, default=10)
    p.add_argument("--grib-dir", default="/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/derived_o320")
    p.add_argument("--nan-sample", type=int, default=4, help="how many dates to scan for NaNs")
    p.add_argument("--allow-identical-members", action="store_true",
                   help="skip the 'members differ' gate (the two-start fixture duplicates one member)")
    p.add_argument("--no-boundary", action="store_true", help="skip the 12 May 00 UTC boundary gate")
    p.add_argument("--no-mars", action="store_true", help="skip the two direct MARS comparisons")
    args = p.parse_args()

    paths = {
        "input": f"{args.dir}/{IN_STEM}{args.suffix}.zarr",
        "target": f"{args.dir}/{TG_STEM}{args.suffix}.zarr",
        "forcings": f"{args.dir}/{FO_STEM}{args.suffix}.zarr",
    }
    zi, zt, zf = (open_dataset(paths[k]) for k in ("input", "target", "forcings"))
    for name in ("input", "target", "forcings"):
        z = {"input": zi, "target": zt, "forcings": zf}[name]
        print(f"{name:9s} {paths[name]}")
        print(f"{'':9s} shape {z.shape} variables {len(z.variables)}")

    attrs = {k: json.load(open(f"{v}/.zattrs")) for k, v in paths.items()}
    mi, mt, mf = (attrs[k]["fake_forecasts"] for k in ("input", "target", "forcings"))

    # 1. one shared synthetic axis
    check(mi == mt == mf, "identical fake_forecasts mapping across the three stores")

    # 2. length, origin and one-hour spacing
    check(len(mi) == args.expect_dates, f"{args.expect_dates} synthetic dates (got {len(mi)})")
    keys = sorted(mi)
    check(keys[0] == args.origin, f"synthetic axis starts at {args.origin} (got {keys[0]})")
    stamps = [datetime.datetime.fromisoformat(k) for k in keys]
    gaps = {(b - a) for a, b in zip(stamps, stamps[1:])}
    check(gaps in ({datetime.timedelta(hours=1)}, set()), f"synthetic dates one hour apart (gaps seen: {gaps})")

    # 3. ensemble dimensions
    check(
        zi.shape[2] == args.members and zt.shape[2] == 1 and zf.shape[2] == 1,
        f"ensemble dims {args.members}/1/1 (got {zi.shape[2]}/{zt.shape[2]}/{zf.shape[2]})",
    )
    check(zi.shape[0] == zt.shape[0] == zf.shape[0] == args.expect_dates, "the three stores have the same length")

    # 4. variables, in the summer order
    check(zi.shape[1] == 68 and zt.shape[1] == 68, f"68 variables in input and target (got {zi.shape[1]}/{zt.shape[1]})")
    check(list(zi.variables) == list(zt.variables), "same variable order in input and target")
    try:
        summer_vars = json.load(open(f"{SUMMER_IN}/.zattrs"))["variables"]
        check(list(zi.variables) == list(summer_vars), "same variable order as the summer input store")
    except OSError as exc:
        check(False, f"could not read the summer store for the variable-order comparison: {exc}")

    # 5. finiteness
    for name in ("input", "target", "forcings"):
        with_nans = attrs[name].get("variables_with_nans", []) or []
        check(len(with_nans) == 0, f"{name}: store-wide variables_with_nans is empty (got {with_nans})")
    n = min(args.nan_sample, zi.shape[0])
    idx = sorted({int(round(i)) for i in np.linspace(0, zi.shape[0] - 1, n)})
    print(f"NaN spot check over dates {idx}")
    for label, z in (("input", zi), ("target", zt), ("forcings", zf)):
        counts = {}
        for i in idx:
            block = np.asarray(z[i])  # (variables, ensemble, gridpoints)
            bad = (~np.isfinite(block)).sum(axis=(1, 2))
            for v, c in zip(z.variables, bad):
                counts[v] = counts.get(v, 0) + int(c)
        offenders = {v: c for v, c in counts.items() if c}
        check(not offenders, f"{label}: every variable finite over the scanned dates (offenders: {offenders})")

    # members must actually differ, unless the caller says the fixture duplicates one member
    if args.allow_identical_members:
        print("SKIP  members-differ gate (--allow-identical-members)")
    elif zi.shape[2] > 1:
        a = zi[0, zi.name_to_index["msl"], 0, :]
        b = zi[0, zi.name_to_index["msl"], 1, :]
        check(not np.array_equal(a, b), "input members differ (msl member 1 vs member 2 of the first sample)")

    # a synthetic date -> (start, lead) inverse map, used by the remaining gates
    inverse = {(v[0], int(v[1])): k for k, v in mi.items()}
    dates = [str(d).replace(" ", "T")[:19] for d in zt.dates]

    # 6. target and forcings against a direct MARS retrieval at the sample valid time
    if args.no_mars:
        print("SKIP  direct MARS comparisons (--no-mars)")
    else:
        start_iso, lead = sorted(mi.items())[0][1][0], int(sorted(mi.items())[0][1][1])
        start = datetime.datetime.fromisoformat(start_iso)
        valid = start + datetime.timedelta(hours=lead)
        i = dates.index(sorted(mi)[0][:19])
        an = mars_field(
            "retrieve,class=od,stream=oper,expver=0001,type=an,"
            f"date={valid:%Y%m%d},time={valid:%H}00,step=0,levtype=sfc,param=167,grid=O1280"
        )
        d = np.abs(zt[i, zt.name_to_index["2t"], 0, :].astype(np.float64) - an).max()
        check(d < 1e-3, f"target sample {i} (start {start_iso} +{lead}h) == analysis {valid:%Y-%m-%dT%H} 2t from MARS (max abs diff {d:.2e} K)")
        lsm = mars_field(
            "retrieve,class=od,stream=oper,expver=0001,type=an,"
            f"date={valid:%Y%m%d},time={valid:%H}00,step=0,levtype=sfc,param=172,grid=O1280"
        )
        d = np.abs(zf[i, zf.name_to_index["lsm"], 0, :].astype(np.float64) - lsm).max()
        check(d < 1e-6, f"forcing sample {i} lsm == analysis {valid:%Y-%m-%dT%H} lsm from MARS (max abs diff {d:.2e})")
        # insolation is a valid-time forcing: the 6 h and 12 h samples of one start must differ
        other = inverse.get((start_iso, 12 if lead == 6 else 6))
        if other is not None:
            j = dates.index(other[:19])
            ins = zf.name_to_index["insolation"]
            check(
                not np.array_equal(zf[i, ins, 0, :], zf[j, ins, 0, :]),
                "insolation differs between the 6 h and 12 h samples of one start (valid-time based)",
            )

    # 7. the input equals the source GRIB message, to the bit
    try:
        import eccodes as ec
        from earthkit.data import from_source

        key = sorted(mi)[0]
        start_iso, lead = mi[key][0], int(mi[key][1])
        start = datetime.datetime.fromisoformat(start_iso)
        i = dates.index(key[:19])
        member = min(7, args.members)  # ensemble slot member-1
        path = f"{args.grib_dir}/{start:%Y%m%d_%H}.grib"
        sel = from_source("file", path).sel(
            date=int(start.strftime("%Y%m%d")), time=int(start.strftime("%H%M")),
            step=lead, number=member, levtype="sfc", param="2t",
        )
        check(len(sel) == 1, f"exactly one 2t message for start {start:%Y%m%d_%H} step {lead} member {member} (got {len(sel)})")
        grib_values = sel[0].to_numpy(flatten=True)
        store_values = np.asarray(zi[i, zi.name_to_index["2t"], member - 1, :])
        check(
            np.array_equal(store_values.astype(np.float64), grib_values.astype(np.float64)),
            f"input sample {i} 2t member {member} is bit-identical to {path} "
            f"(max abs diff {np.abs(store_values.astype(np.float64) - grib_values).max():.3e} K)",
        )
        del ec
    except Exception as exc:  # noqa: BLE001
        check(False, f"input-versus-GRIB comparison failed: {exc!r}")

    # 8. the 12 May 00 UTC boundary start
    if args.no_boundary:
        print("SKIP  12 May 00 UTC boundary gate (--no-boundary)")
    else:
        for lead in (6, 12):
            k = inverse.get(("2026-05-12T00:00:00", lead))
            check(k is not None, f"boundary sample present: start 2026-05-12 00 UTC lead {lead} h -> {k}")

    print("ALL_PASS" if OK else "SOME_FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
