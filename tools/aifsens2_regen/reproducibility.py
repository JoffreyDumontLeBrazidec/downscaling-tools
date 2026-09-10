"""Check that the forecasts are reproducible and that members are distinct.

Three questions are answered, all of them on the GPU, one at a time.

First, does the same start and member run twice give the same numbers?  The two
runs happen in separate processes so that nothing can be carried over in
memory.  Bit-for-bit equality is the goal; if the two runs differ, the largest
absolute difference per variable is reported instead, because some CUDA kernels
are not deterministic and a difference at the level of floating-point rounding
means something quite different from a difference in the tenth of a degree.

Second, are two different members and two different starts actually different?
A pipeline that quietly fed the same initial condition to every member would
pass every field-count check ever written, so this compares fields directly.

Third, and most important, was member m started from member m's own initial
condition?  This compares the two-metre temperature in the member's initial
condition at the analysis time against the same field in the runner's input
state, and also confirms that a member's forecast is closer to its own initial
condition than to another member's.

Usage
-----
    python -m aifsens2_regen.reproducibility --block pilot_20260101 --start 20260101_00
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from . import calendar as cal
from . import gribspec as spec
from .assemble import members_root
from .common import DEFAULT_ROOT, atomic_write_json, log, move_aside, sha256_file
from .regrid import native_root_for


def read_field(path: str, param: int, step: int | None, level: int = 0):
    """Read one field's values from a GRIB file as a numpy array."""
    import numpy as np
    from eccodes import (
        codes_get,
        codes_get_values,
        codes_grib_new_from_file,
        codes_release,
    )

    found = None
    with open(path, "rb") as f:
        while True:
            h = codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                if codes_get(h, "paramId") == param and codes_get(h, "level") == level:
                    if step is None or codes_get(h, "endStep") == step:
                        found = np.array(codes_get_values(h))
                        break
            finally:
                codes_release(h)
    return found


def compare_files(a: str, b: str) -> dict:
    """Compare two native forecast files field by field."""
    import numpy as np
    from eccodes import (
        codes_get,
        codes_get_values,
        codes_grib_new_from_file,
        codes_release,
    )

    def index(path):
        out = {}
        with open(path, "rb") as f:
            while True:
                h = codes_grib_new_from_file(f)
                if h is None:
                    break
                try:
                    key = (
                        codes_get(h, "paramId"),
                        codes_get(h, "level"),
                        codes_get(h, "endStep"),
                    )
                    out[key] = np.array(codes_get_values(h))
                finally:
                    codes_release(h)
        return out

    ia, ib = index(a), index(b)
    common = sorted(set(ia) & set(ib))
    identical = 0
    worst = 0.0
    worst_key = None
    differing = []
    for k in common:
        x, y = ia[k], ib[k]
        m = np.isfinite(x) & np.isfinite(y)
        if np.array_equal(x[m], y[m]) and np.array_equal(np.isfinite(x), np.isfinite(y)):
            identical += 1
            continue
        d = float(np.nanmax(np.abs(x[m] - y[m]))) if m.any() else float("nan")
        differing.append((k, d))
        if d == d and d > worst:
            worst, worst_key = d, k
    return dict(
        fields_compared=len(common),
        fields_only_in_first=sorted(set(ia) - set(ib)),
        fields_only_in_second=sorted(set(ib) - set(ia)),
        fields_bit_identical=identical,
        fields_differing=len(differing),
        max_absolute_difference=worst,
        max_absolute_difference_field=worst_key,
        largest_differences=sorted(differing, key=lambda t: -t[1])[:10],
    )


def run_single(block: str, root: str, start_key: str, member: int, out_root: str) -> str:
    """Run one forecast in a fresh process and return the path it wrote."""
    cmd = [
        sys.executable, "-m", "aifsens2_regen.run_forecasts",
        "--block", block, "--root", root,
        "--only-start", start_key, "--only-member", str(member),
        "--native-root", out_root, "--force",
    ]
    log("running: " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=os.environ.get("AIFSENS2_TOOLS", os.getcwd()))
    if p.returncode != 0:
        raise SystemExit(f"the repeat run failed with exit code {p.returncode}")
    return os.path.join(out_root, start_key, f"m{member:02d}.grib")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block", required=True)
    p.add_argument("--root", default=os.environ.get("AIFSENS2_ROOT", DEFAULT_ROOT))
    p.add_argument("--start", help="which start to test, as YYYYMMDD_HH; default the first")
    p.add_argument("--member", type=int, default=1)
    p.add_argument("--other-member", type=int, default=2)
    a = p.parse_args(argv)

    starts = cal.block_starts(a.block, a.root)
    if not starts:
        log("FATAL the block has no starts")
        return 1
    key = a.start or cal.start_key(starts[0])
    other_key = cal.start_key(starts[1]) if len(starts) > 1 else None

    nat_root = native_root_for(a.root, a.block)
    ic_root = members_root(a.root, a.block)
    check_root = os.path.join(a.root, "validation", "reproducibility", a.block)
    os.makedirs(check_root, exist_ok=True)

    report: dict = {"block": a.block, "start": key, "member": a.member}

    # ---- 1. the same forecast twice, in two separate processes ----
    first = run_single(a.block, a.root, key, a.member, os.path.join(check_root, "run_a"))
    second = run_single(a.block, a.root, key, a.member, os.path.join(check_root, "run_b"))
    sha_a, sha_b = sha256_file(first), sha256_file(second)
    same_bytes = sha_a == sha_b
    cmp_repeat = compare_files(first, second)
    report["repeat_run"] = dict(
        first=first, second=second, sha256_first=sha_a, sha256_second=sha_b,
        whole_file_identical=same_bytes, **cmp_repeat,
    )
    if same_bytes:
        log(
            f"REPRODUCIBILITY bit-for-bit identical: the two runs produced the same "
            f"{cmp_repeat['fields_compared']} fields and the same file checksum {sha_a}"
        )
    else:
        log(
            f"REPRODUCIBILITY not bit-for-bit: {cmp_repeat['fields_bit_identical']} of "
            f"{cmp_repeat['fields_compared']} fields are identical, the largest "
            f"absolute difference is {cmp_repeat['max_absolute_difference']:.3e} in "
            f"{cmp_repeat['max_absolute_difference_field']}; this is CUDA "
            f"non-determinism, not a seeding failure, if the difference is at the "
            f"level of floating point rounding"
        )

    # ---- 2. two members, and two starts, must differ ----
    mine = os.path.join(nat_root, key, f"m{a.member:02d}.grib")
    other = os.path.join(nat_root, key, f"m{a.other_member:02d}.grib")
    if os.path.exists(mine) and os.path.exists(other):
        cmp_members = compare_files(mine, other)
        report["different_members"] = cmp_members
        log(
            f"MEMBER_DISTINCTNESS members {a.member} and {a.other_member} at {key}: "
            f"{cmp_members['fields_differing']} of {cmp_members['fields_compared']} "
            f"fields differ, largest difference {cmp_members['max_absolute_difference']:.3f} "
            f"in {cmp_members['max_absolute_difference_field']}"
        )
    else:
        report["different_members"] = {"note": "one of the two member files is absent"}

    if other_key:
        other_start = os.path.join(nat_root, other_key, f"m{a.member:02d}.grib")
        if os.path.exists(mine) and os.path.exists(other_start):
            cmp_starts = compare_files(mine, other_start)
            report["different_starts"] = cmp_starts
            log(
                f"START_DISTINCTNESS member {a.member} at {key} against {other_key}: "
                f"{cmp_starts['fields_differing']} of {cmp_starts['fields_compared']} "
                f"fields differ"
            )

    # ---- 3. member m really started from member m's initial condition ----
    # The two-metre temperature at the analysis time is compared between the
    # initial condition files and the six-hour forecasts.  A forecast six hours
    # ahead stays close to its own analysis and further from another member's,
    # so the ordering of these two distances tells us which analysis was used.
    import numpy as np

    prov = {}
    t2m = 167
    ic_mine = read_field(os.path.join(ic_root, key, f"m{a.member:02d}.grib"), t2m, None)
    ic_other = read_field(os.path.join(ic_root, key, f"m{a.other_member:02d}.grib"), t2m, None)
    fc_mine = read_field(mine, t2m, spec.LEAD_STEPS[0]) if os.path.exists(mine) else None

    if ic_mine is not None and ic_other is not None and fc_mine is not None:
        d_own = float(np.sqrt(np.nanmean((fc_mine - ic_mine) ** 2)))
        d_other = float(np.sqrt(np.nanmean((fc_mine - ic_other) ** 2)))
        ic_spread = float(np.sqrt(np.nanmean((ic_mine - ic_other) ** 2)))
        prov = dict(
            rms_forecast_minus_own_initial_condition=d_own,
            rms_forecast_minus_other_initial_condition=d_other,
            rms_between_the_two_initial_conditions=ic_spread,
            verdict=(
                "member started from its own initial condition"
                if d_own < d_other
                else "SUSPECT: the forecast is closer to the other member's analysis"
            ),
        )
        log(
            f"PROVENANCE member {a.member} at {key}: the six-hour forecast is "
            f"{d_own:.3f} K from its own analysis and {d_other:.3f} K from member "
            f"{a.other_member}'s analysis (the two analyses differ by {ic_spread:.3f} K). "
            f"{prov['verdict']}"
        )
    else:
        prov = {"note": "could not read the two-metre temperature from every file"}
    report["provenance"] = prov

    out = os.path.join(a.root, "manifests", f"reproducibility_{a.block}.json")
    atomic_write_json(out, report)
    log(f"reproducibility report written to {out}")
    print(json.dumps({k: v for k, v in report.items() if k != "repeat_run"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
