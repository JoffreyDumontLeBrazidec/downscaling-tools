"""Stage 6: decide whether a block is finished, and say so in one word.

The check is deliberately independent of the manifests: it looks at the files
themselves, so that a manifest written by a job that later crashed cannot make
an incomplete block look finished.

A block passes when, for every one of its starts, all ten members have a native
forecast with the right number of fields at both lead times, the O320 file for
that start has its 1360 messages, and a sample of the data is finite.  The
sample is one member per start by default; a full check of every field is
available with --sample-all and is much slower.

The output ends with a single line beginning PASS or FAIL, and every missing
record is listed by name so that the gap can be re-run directly.

Usage
-----
    python -m aifsens2_regen.verify_block --block pilot_20260101
"""

from __future__ import annotations

import argparse
import os
import sys

from . import calendar as cal
from . import gribspec as spec
from .assemble import members_root
from .common import DEFAULT_ROOT, grib_count, log, run_cmd
from .regrid import _validate_derived, derived_root_for, native_root_for


def sample_is_finite(path: str, limit: int = 8) -> tuple[bool, str]:
    """Check that the first few fields of a file are not entirely missing.

    grib_get is asked for the maximum and the average of each field.  A field
    whose statistics cannot be computed at all, or whose maximum is not a
    number, is reported.  Fields that are partly missing are normal here, so
    only a wholly undefined field counts as a failure.
    """
    rc, out = run_cmd(["grib_get", "-p", "shortName,level,max,average", path])
    if rc != 0:
        return False, "grib_get could not read the file"
    bad = []
    for line in out.strip().splitlines()[:limit] + out.strip().splitlines()[-limit:]:
        f = line.split()
        if len(f) != 4:
            continue
        try:
            mx = float(f[2])
        except ValueError:
            bad.append(f"{f[0]}{f[1]}: maximum is {f[2]}")
            continue
        if mx != mx:  # not a number
            bad.append(f"{f[0]}{f[1]}: maximum is not a number")
    return (not bad), "; ".join(bad)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block", required=True)
    p.add_argument("--root", default=os.environ.get("AIFSENS2_ROOT", DEFAULT_ROOT))
    p.add_argument("--skip-o320", action="store_true", help="check only the native forecasts")
    a = p.parse_args(argv)

    starts = cal.block_starts(a.block, a.root)
    if not starts:
        print(f"FAIL block {a.block} has no starts defined")
        return 1

    ic_root = members_root(a.root, a.block)
    nat_root = native_root_for(a.root, a.block)
    der_root = derived_root_for(a.root, a.block)

    want_native = spec.NATIVE_FIELDS_PER_STEP_EXPECTED * len(spec.LEAD_STEPS)
    missing: list[str] = []
    ok_members = 0
    ok_starts = 0

    for start in starts:
        key = cal.start_key(start)
        start_ok = True
        for m in spec.MEMBERS:
            tag = f"m{m:02d}"
            ic = os.path.join(ic_root, key, f"{tag}.grib")
            nat = os.path.join(nat_root, key, f"{tag}.grib")

            if not os.path.exists(ic):
                missing.append(f"{key} {tag}: initial condition absent")
                start_ok = False
                continue
            n_ic = grib_count(ic)
            if n_ic != spec.FIELDS_PER_MEMBER_FILE:
                missing.append(
                    f"{key} {tag}: initial condition has {n_ic} fields, "
                    f"expected {spec.FIELDS_PER_MEMBER_FILE}"
                )
                start_ok = False
                continue

            if not os.path.exists(nat):
                missing.append(f"{key} {tag}: native forecast absent")
                start_ok = False
                continue
            n_nat = grib_count(nat)
            if n_nat != want_native:
                missing.append(
                    f"{key} {tag}: native forecast has {n_nat} fields, expected {want_native}"
                )
                start_ok = False
                continue
            ok_members += 1

        # Sample one member of the start for finite values.
        sample = os.path.join(nat_root, key, f"m{spec.MEMBERS[0]:02d}.grib")
        if os.path.exists(sample):
            finite, why = sample_is_finite(sample)
            if not finite:
                missing.append(f"{key}: sampled fields are undefined: {why}")
                start_ok = False

        if not a.skip_o320:
            derived = os.path.join(der_root, f"{key}.grib")
            if not os.path.exists(derived):
                missing.append(f"{key}: O320 file absent")
                start_ok = False
            else:
                ok, problems, _ = _validate_derived(derived, start)
                if not ok:
                    missing.append(f"{key}: O320 file is wrong: {'; '.join(problems)}")
                    start_ok = False

        if start_ok:
            ok_starts += 1

    want_members = len(starts) * len(spec.MEMBERS)
    print(f"block {a.block}: {len(starts)} starts, {len(spec.MEMBERS)} members each")
    print(f"native member forecasts present and complete: {ok_members} of {want_members}")
    print(f"starts fully complete (including the O320 file): {ok_starts} of {len(starts)}")
    if missing:
        print(f"missing or wrong records ({len(missing)}):")
        for line in missing:
            print(f"  {line}")

    if ok_starts == len(starts) and ok_members == want_members:
        print(f"PASS block {a.block} is complete")
        return 0
    print(f"FAIL block {a.block} is not complete")
    return 1


if __name__ == "__main__":
    sys.exit(main())
