"""Open the combined summer + early-2026 training view and check it is what training expects.

The view itself is described in combined_view.yaml, which is not a store but the set of
arguments handed to anemoi.datasets.open_dataset. This script opens it read-only and checks:

  * the length (446 summer samples + 526 early samples = 972);
  * the ensemble size of each of the three views (10 for the input, 1 for target and forcings);
  * the synthetic dates are contiguous, one hour apart, with no gap and no duplicate;
  * the union of the two stores' fake_forecasts mappings resolves every synthetic date of the
    view to exactly one (forecast start, lead) pair, over 486 distinct starts (223 + 263);
  * the 12 May 2026 boundary: the last early sample (start 12 May 00 UTC, lead 12 h) and the
    first summer sample (start 12 May 12 UTC, lead 6 h) are both present, and the two halves
    meet at that boundary with no missing and no repeated start;
  * the statistics of the view are the summer store's, not a recomputed or concatenated set;
  * a sample read from the summer half and a sample read from the early half are both finite.

It also reports whether the combined view is consistent with the ensemble-member pick that the
training loader performs (sandbox /home/ecm5702/hpcperm/sandbox/20260908-aifs-arm-a, branch
exp/aifs-arm-a-20260908, MultiDataset.get_sample). That code draws one member index per sample
from the FIRST store whose sample has more than one member and applies the same index to every
other multi-member store, so its recorded limitation is two multi-member stores of DIFFERENT
ensemble sizes. This script checks that at most one of the three views is multi-member.

Modes:
  --mode combined     open the whole view (needs the early stores to exist)
  --mode summer-only  open only the summer half, with the same number subset and statistics

Overrides, used to run the test against a tiny early store built from the fixture:
  --early-dir D --early-suffix S --expect-early N
"""

import argparse
import datetime
import json
import sys

import numpy as np
import yaml
from anemoi.datasets import open_dataset

OK = True


def check(condition, message):
    global OK
    print(("PASS " if condition else "FAIL ") + message, flush=True)
    OK = OK and bool(condition)


def retarget(spec, early_dir, early_suffix):
    """Point the second member of each concat at another directory and store-name suffix."""
    if early_dir is None and not early_suffix:
        return spec
    entry = spec["concat"][1]
    path = entry["dataset"]
    name = path.rsplit("/", 1)[1][: -len(".zarr")]
    directory = early_dir if early_dir else path.rsplit("/", 1)[0]
    entry["dataset"] = f"{directory}/{name}{early_suffix}.zarr"
    return spec


def summer_only(spec):
    out = {k: v for k, v in spec.items() if k != "concat"}
    out.update(spec["concat"][0])
    return out


def fake_forecasts_of(spec):
    """Union of the fake_forecasts mappings of the stores named in a concat specification."""
    union = {}
    duplicates = []
    for entry in spec["concat"]:
        attrs = json.load(open(f"{entry['dataset']}/.zattrs"))
        for key, value in attrs["fake_forecasts"].items():
            if key in union:
                duplicates.append(key)
            union[key] = (value[0], int(value[1]))
    return union, duplicates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--view", default="/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/datasets-build/combined_view.yaml")
    p.add_argument("--mode", choices=["combined", "summer-only"], default="combined")
    p.add_argument("--early-dir", default=None)
    p.add_argument("--early-suffix", default="")
    p.add_argument("--expect-summer", type=int, default=446)
    p.add_argument("--expect-early", type=int, default=526)
    p.add_argument("--expect-starts", type=int, default=486)
    p.add_argument("--members", type=int, default=10)
    p.add_argument("--no-boundary", action="store_true",
                   help="the early half is only a block of the period, not all of it: skip the "
                        "start-contiguity and 12 May 2026 boundary gates")
    args = p.parse_args()

    view = yaml.safe_load(open(args.view))
    specs = {k: retarget(view[k], args.early_dir, args.early_suffix) for k in ("input", "target", "forcings")}

    if args.mode == "summer-only":
        expected = args.expect_summer
        specs = {k: summer_only(v) for k, v in specs.items()}
    else:
        expected = args.expect_summer + args.expect_early

    print(f"mode {args.mode}, expecting {expected} samples")
    for k, v in specs.items():
        print(f"  {k}: {json.dumps(v)}")

    opened = {k: open_dataset(v) for k, v in specs.items()}
    for k, z in opened.items():
        print(f"{k:9s} shape {z.shape} variables {len(z.variables)} "
              f"dates {z.dates[0]} .. {z.dates[-1]}")

    zi, zt, zf = opened["input"], opened["target"], opened["forcings"]

    # length
    check(zi.shape[0] == expected, f"combined length {expected} (got {zi.shape[0]})")
    check(zt.shape[0] == expected and zf.shape[0] == expected,
          f"target and forcings have the same length (got {zt.shape[0]} and {zf.shape[0]})")

    # ensemble sizes
    check(zi.shape[2] == args.members, f"input ensemble size {args.members} (got {zi.shape[2]})")
    check(zt.shape[2] == 1 and zf.shape[2] == 1,
          f"target and forcings have one member (got {zt.shape[2]} and {zf.shape[2]})")

    # contiguous hourly synthetic axis
    stamps = [d.astype("datetime64[s]").astype(datetime.datetime) if hasattr(d, "astype") else d for d in zi.dates]
    gaps = {(b - a) for a, b in zip(stamps, stamps[1:])}
    check(gaps == {datetime.timedelta(hours=1)}, f"synthetic dates one hour apart (gaps seen: {sorted(gaps)})")

    # the union mapping
    if args.mode == "combined":
        union, duplicates = fake_forecasts_of(specs["input"])
        check(not duplicates, f"the two stores share no synthetic date (duplicates: {duplicates[:5]})")
        check(len(union) == expected, f"the union mapping has {expected} entries (got {len(union)})")
        unresolved = [str(d) for d in stamps if d.isoformat() not in union]
        check(not unresolved, f"every synthetic date of the view resolves to a (start, lead) pair "
                              f"(unresolved: {unresolved[:5]})")
        pairs = list(union.values())
        check(len(set(pairs)) == len(pairs), "every (start, lead) pair is used exactly once")
        starts = sorted({datetime.datetime.fromisoformat(s) for s, _ in pairs})
        check(len(starts) == args.expect_starts, f"{args.expect_starts} distinct forecast starts (got {len(starts)})")
        # Contiguity of the starts and the 12 May boundary are both properties of the FULL early
        # period, so they are skipped together when the early half is only a block of it.
        if args.no_boundary:
            print("SKIP  start-contiguity gate (the early half is only a block of the period)")
        else:
            start_gaps = {(b - a) for a, b in zip(starts, starts[1:])}
            check(start_gaps == {datetime.timedelta(hours=12)},
                  f"the union of starts is contiguous at 12 h (gaps seen: {sorted(start_gaps)})")

        # the 12 May 2026 boundary
        if args.no_boundary:
            print("SKIP  12 May 2026 boundary gate (--no-boundary)")
        else:
            inverse = {v: k for k, v in union.items()}
            early_last = inverse.get(("2026-05-12T00:00:00", 12))
            summer_first = inverse.get(("2026-05-12T12:00:00", 6))
            check(early_last is not None, f"early boundary sample present: start 12 May 00 UTC lead 12 h -> {early_last}")
            check(summer_first is not None, f"summer boundary sample present: start 12 May 12 UTC lead 6 h -> {summer_first}")
            if early_last and summer_first:
                iso = [d.isoformat() for d in stamps]
                print(f"      synthetic positions: early boundary at index {iso.index(early_last)} ({early_last}), "
                      f"summer boundary at index {iso.index(summer_first)} ({summer_first})")
                # The concat puts the summer half first, so the two boundary samples sit at the two
                # ends of the synthetic axis. What must be adjacent is the real calendar: the last
                # early start and the first summer start are 12 h apart with nothing in between.
                check(
                    datetime.datetime.fromisoformat("2026-05-12T12:00:00")
                    - datetime.datetime.fromisoformat("2026-05-12T00:00:00")
                    == datetime.timedelta(hours=12)
                    and starts[starts.index(datetime.datetime(2026, 5, 12, 0, 0)) + 1] == datetime.datetime(2026, 5, 12, 12, 0),
                    "the early and summer halves meet at 12 May 2026 with no missing and no repeated start",
                )

    # statistics come from the summer store, explicitly
    summer_stats = open_dataset(view["input"]["statistics"]).statistics
    same = all(np.array_equal(np.asarray(zi.statistics[k]), np.asarray(summer_stats[k])) for k in summer_stats)
    check(same, "the input view's statistics are the summer store's, key by key")
    print("      summer 2t mean %.4f stdev %.4f ; view 2t mean %.4f stdev %.4f" % (
        float(summer_stats["mean"][zi.name_to_index["2t"]]),
        float(summer_stats["stdev"][zi.name_to_index["2t"]]),
        float(zi.statistics["mean"][zi.name_to_index["2t"]]),
        float(zi.statistics["stdev"][zi.name_to_index["2t"]]),
    ))

    # a finite sample from each half
    probes = [0] if args.mode == "summer-only" else [0, expected - 1]
    for i in probes:
        half = "summer" if i < args.expect_summer else "early"
        for label, z in (("input", zi), ("target", zt), ("forcings", zf)):
            block = np.asarray(z[i])
            check(np.isfinite(block).all(), f"{half} half, {label} sample {i} is finite "
                                            f"(shape {block.shape}, mean {float(np.nanmean(block)):.4g})")

    # consistency with the training loader's member pick
    multi = [k for k, z in opened.items() if z.shape[2] > 1]
    sizes = {k: opened[k].shape[2] for k in multi}
    check(len(multi) <= 1, f"at most one multi-member view, so the loader's shared member draw is "
                           f"unambiguous (multi-member views: {sizes})")
    check(len(set(sizes.values())) <= 1, f"no two multi-member views of different ensemble sizes, which is "
                                         f"the limitation recorded in the aifs-arm-a experiment (sizes: {sizes})")

    print("ALL_PASS" if OK else "SOME_FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
