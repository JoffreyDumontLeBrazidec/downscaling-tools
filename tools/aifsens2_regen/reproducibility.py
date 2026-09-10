"""Check that the forecasts are reproducible and that members are distinct.

Three questions are answered.  Only the first needs a GPU, so the other two can
be re-run cheaply with --skip-repeat.

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
condition?  Comparing a forecast against an analysis cannot answer that: in six
hours the two-metre temperature moves by about 3.6 K in the root mean square
while two members' analyses differ by only about 0.9 K, so the comparison is
swamped by the common evolution.  The question is instead asked in terms of
differences between members, where that common evolution cancels.  If member i
started from analysis i, the way member i's forecast differs from member j's
must carry the imprint of the way analysis i differs from analysis j, so the
two difference fields should correlate strongly and positively.  Two
deliberately mismatched pairings are computed as controls, and the matched pair
has to give the clearly largest correlation.

Usage
-----
    python -m aifsens2_regen.reproducibility --block pilot_20260101 --start 20260101_00
    python -m aifsens2_regen.reproducibility --block pilot_20260101 --skip-repeat
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
from .common import DEFAULT_ROOT, atomic_write_json, log, read_json, sha256_file
from .regrid import native_root_for


def read_field(path: str, param: int, step: int | None, level: int = 0, latest: bool = False):
    """Read one field's values from a GRIB file as a numpy array.

    With latest=True the message with the greatest date and time is returned
    rather than the first match.  That matters for the initial-condition files,
    which hold two analysis times: the analysis at the start is the later of the
    two, and taking the first match would silently give the analysis six hours
    earlier instead.
    """
    import numpy as np
    from eccodes import (
        codes_get,
        codes_get_values,
        codes_grib_new_from_file,
        codes_release,
    )

    best_stamp = None
    found = None
    with open(path, "rb") as f:
        while True:
            h = codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                if codes_get(h, "paramId") != param or codes_get(h, "level") != level:
                    continue
                if step is not None and codes_get(h, "endStep") != step:
                    continue
                stamp = (codes_get(h, "dataDate"), codes_get(h, "dataTime"))
                if not latest:
                    found = np.array(codes_get_values(h))
                    break
                if best_stamp is None or stamp > best_stamp:
                    best_stamp, found = stamp, np.array(codes_get_values(h))
            finally:
                codes_release(h)
    return found


def correlation(x, y) -> float:
    """Pearson correlation of two fields, ignoring points that are not finite."""
    import numpy as np

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    a, b = x[m], y[m]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


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
    p.add_argument(
        "--skip-repeat",
        action="store_true",
        help="do not rerun the forecast twice; keep the earlier result and run "
             "only the checks that read existing files, which need no GPU",
    )
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
    # This is the only part that needs a GPU.  With --skip-repeat the earlier
    # result is carried over from the saved report, so the checks that only
    # read files can be re-run on a CPU node.
    out = os.path.join(a.root, "manifests", f"reproducibility_{a.block}.json")
    if a.skip_repeat:
        earlier = read_json(out, {}) or {}
        report["repeat_run"] = earlier.get(
            "repeat_run", {"note": "not run, and no earlier result was saved"}
        )
        log("skipping the repeat run and keeping the earlier result")
    else:
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
                f"REPRODUCIBILITY bit-for-bit identical: the two runs produced the "
                f"same {cmp_repeat['fields_compared']} fields and the same file "
                f"checksum {sha_a}"
            )
        else:
            log(
                f"REPRODUCIBILITY not bit-for-bit: {cmp_repeat['fields_bit_identical']} "
                f"of {cmp_repeat['fields_compared']} fields are identical, the largest "
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
    #
    # Comparing a forecast against an analysis directly does not answer this.
    # In six hours the two-metre temperature moves by about 3.6 K in the root
    # mean square, while two ensemble members' analyses differ by only about
    # 0.9 K, so both distances come out nearly equal and the comparison decides
    # nothing.  That was the first attempt and it produced a meaningless
    # "suspect" verdict.
    #
    # The question has to be asked in terms of differences between members,
    # where the common evolution cancels out.  If member i really started from
    # analysis i, then the way member i's forecast differs from member j's must
    # carry the imprint of the way analysis i differs from analysis j.  So the
    # analysis difference field and the forecast difference field should be
    # strongly and positively correlated.
    #
    # Two controls are computed alongside it.  Pairing the analysis difference
    # between members 1 and 2 with the forecast difference between members 1
    # and a third member tests whether the correlation is specific to the right
    # pair, or merely reflects some structure shared by every member.  The
    # correct pairing must give the clearly largest correlation.
    t2m = 167
    third = next(m for m in spec.MEMBERS if m not in (a.member, a.other_member))

    def ic_field(m):
        return read_field(
            os.path.join(ic_root, key, f"m{m:02d}.grib"), t2m, None, latest=True
        )

    def fc_field(m):
        p = os.path.join(nat_root, key, f"m{m:02d}.grib")
        return read_field(p, t2m, spec.LEAD_STEPS[0]) if os.path.exists(p) else None

    A_i, A_j, A_k = ic_field(a.member), ic_field(a.other_member), ic_field(third)
    F_i, F_j, F_k = fc_field(a.member), fc_field(a.other_member), fc_field(third)

    if all(v is not None for v in (A_i, A_j, A_k, F_i, F_j, F_k)):
        matched = correlation(A_i - A_j, F_i - F_j)
        control_one = correlation(A_i - A_j, F_i - F_k)
        control_two = correlation(A_i - A_k, F_i - F_j)
        ok = matched > 0.3 and matched > control_one and matched > control_two
        prov = dict(
            variable="2t",
            members=[a.member, a.other_member, third],
            correlation_matched_pair=matched,
            correlation_control_mismatched_forecast=control_one,
            correlation_control_mismatched_analysis=control_two,
            verdict=(
                "member m's forecast carries member m's analysis perturbation"
                if ok
                else "SUSPECT: the forecast perturbation does not follow the analysis "
                "perturbation for the matched pair"
            ),
        )
        log(
            f"PROVENANCE at {key}: the two-metre temperature difference between "
            f"members {a.member} and {a.other_member} correlates at {matched:.3f} "
            f"between their analyses and their six-hour forecasts. The mismatched "
            f"controls give {control_one:.3f} and {control_two:.3f}. {prov['verdict']}"
        )
    else:
        prov = {"note": "could not read the two-metre temperature from every file"}
    report["provenance"] = prov

    atomic_write_json(out, report)
    log(f"reproducibility report written to {out}")
    print(json.dumps({k: v for k, v in report.items() if k != "repeat_run"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
