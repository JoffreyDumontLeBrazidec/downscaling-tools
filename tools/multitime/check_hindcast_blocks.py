#!/usr/bin/env python3
"""Check that a fake_hindcasts view of a hindcast store is made of whole forecast blocks.

The multi-time (multi-in multi-out) training on Vera Gahlen's `feat/spatial_downscaler`
branch reads the Europe box E2 stores with `HindcastBlockDataset`, which ASSUMES that the
rows kept by the lane's `fake_hindcasts` selection come in blocks of `steps_per_sequence`
consecutive rows, one block per reference date, steps in increasing order. This script
verifies that assumption on the real store before anything is trained.

It needs the fork's anemoi-datasets (the one with `fake_hindcasts`) on the path, and a
store whose zarr attributes carry the `fake_hindcasts` mapping (row date -> reference
date, hindcast date, step), as every lane store does.

Usage (Atos, from a sandbox with the fork's anemoi-datasets):
  python3 check_hindcast_blocks.py /ec/res4/scratch/ecm5702/o320_o1280_europe_e2_v4/data/out_hres_o1280.zarr \
      --end 2021 --steps 12,24,36,48,60,72 --steps-per-sequence 6

Exit code 0 and the line "BLOCKS OK" mean the assumption holds for that selection;
anything else means stop and look.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _parse_steps(text: str) -> list[int]:
    return [int(s) for s in text.split(",") if s.strip()]


def _date_key(date: np.datetime64) -> str:
    return str(np.datetime_as_string(date, unit="s"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zarr", help="path of the hindcast store (any of the triplet; they share the axis)")
    parser.add_argument("--start", default=None, help="fake_hindcasts start (hindcast year), e.g. 2022")
    parser.add_argument("--end", default=None, help="fake_hindcasts end (hindcast year), e.g. 2021")
    parser.add_argument("--steps", default="12,24,36,48,60,72", help="fake_hindcasts steps in hours")
    parser.add_argument("--steps-per-sequence", type=int, default=6)
    parser.add_argument("--json", default=None, help="write the summary here as JSON")
    args = parser.parse_args()

    import zarr
    from anemoi.datasets import open_dataset

    steps = _parse_steps(args.steps)
    selection = {"steps": steps}
    if args.start is not None:
        selection["start"] = int(args.start) if args.start.isdigit() else args.start
    if args.end is not None:
        selection["end"] = int(args.end) if args.end.isdigit() else args.end

    raw = zarr.open(args.zarr, mode="r")
    attrs = dict(raw.attrs)
    if "fake_hindcasts" not in attrs:
        print(f"FAIL: {args.zarr} has no 'fake_hindcasts' attribute; cannot map rows to (refdate, step).")
        return 2
    mapping = attrs["fake_hindcasts"]  # {date string: {...}} as written by the materialiser
    raw_dates = np.asarray(raw["dates"][:]).astype("datetime64[s]")

    view = open_dataset(args.zarr, fake_hindcasts=selection)
    view_dates = np.asarray(view.dates).astype("datetime64[s]")
    n = len(view_dates)
    k = args.steps_per_sequence
    print(f"view rows: {n}; steps per sequence: {k}; selection: {json.dumps(selection)}")
    if n % k != 0:
        print(f"FAIL: {n} rows is not a multiple of {k}.")
        return 2

    # Map view rows back to raw rows through the (unique, hourly) synthetic date axis.
    raw_index = {int(d.astype("int64")): i for i, d in enumerate(raw_dates)}
    rows = np.array([raw_index[int(d.astype("int64"))] for d in view_dates])

    def entry(row: int) -> dict:
        key = _date_key(raw_dates[row])
        value = mapping.get(key)
        if value is None:
            # Some stores key the mapping by position.
            value = mapping.get(str(row))
        if value is None:
            raise KeyError(f"no fake_hindcasts entry for raw row {row} ({key}); keys look like {list(mapping)[:3]}")
        return value

    def refdate_and_step(value) -> tuple[str, int]:
        if isinstance(value, dict):
            ref = value.get("reference_date") or value.get("refdate") or value.get("base_date") or value.get("date")
            step = value.get("step")
            if step is None:
                raise KeyError(f"no step in fake_hindcasts entry {value}")
            return str(ref), int(step)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            # (reference date, hindcast date, step) or (reference date, step)
            return str(value[0]), int(value[-1])
        raise TypeError(f"unexpected fake_hindcasts entry {value!r}")

    bad_blocks = []
    monotone_rows = bool(np.all(np.diff(rows) > 0))
    for b in range(n // k):
        block = rows[b * k : (b + 1) * k]
        refs, stps = zip(*(refdate_and_step(entry(int(r))) for r in block))
        same_ref = len(set(refs)) == 1
        ordered = list(stps) == sorted(stps) and list(stps) == steps
        if not (same_ref and ordered):
            bad_blocks.append({"block": b, "rows": block.tolist(), "refdates": list(refs), "steps": list(stps)})
            if len(bad_blocks) <= 5:
                print(f"BAD block {b}: refdates {set(refs)} steps {list(stps)}")

    summary = {
        "zarr": args.zarr,
        "selection": selection,
        "view_rows": int(n),
        "blocks": int(n // k),
        "steps_per_sequence": k,
        "rows_monotone": monotone_rows,
        "bad_blocks": len(bad_blocks),
        "first_bad": bad_blocks[:5],
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2, default=str)
    print(json.dumps({key: value for key, value in summary.items() if key != "first_bad"}))
    if bad_blocks or not monotone_rows:
        print("BLOCKS FAIL")
        return 1
    print("BLOCKS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
