"""Stage 4: select the lane variables and regrid them from N320 to O320.

The native forecast holds all 119 output fields of the checkpoint on the
model's own reduced Gaussian N320 grid.  The dataset build that consumes this
campaign needs a much smaller selection, 68 variables, on the O320 octahedral
grid.  The 68 are eight surface fields and six upper-air fields on ten pressure
levels.

The output layout is one file per forecast start, holding every member and both
lead times together:

    derived_o320/<YYYYMMDD_HH>.grib     1360 messages
                                        = 10 members x 2 lead times x 68 fields

That layout is what the dataset build reads, so it must not be changed without
telling whoever owns that stage.

The work runs on a normal compute node, not a GPU, because mir is a CPU tool.
Selection is done with grib_copy, which does not decode the data, and the
regridding with mir.  As everywhere else in this pipeline the result is written
under a temporary name, validated and only then renamed, so an interrupted job
never leaves a half-written file that a later stage would trust.

Usage
-----
    python -m aifsens2_regen.regrid --block pilot_20260101
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time

from . import calendar as cal
from . import gribspec as spec
from .common import (
    DEFAULT_ROOT,
    atomic_write_json,
    concat_files,
    grib_count,
    log,
    move_aside,
    run_cmd,
)


def selection_rules() -> list[list[str]]:
    """The grib_copy selections that together pick the 68 lane variables.

    Two selections are needed because the surface fields and the pressure-level
    fields cannot be expressed in one -w clause: a level constraint that suits
    the pressure levels would exclude the surface fields.
    """
    steps = "/".join(str(s) for s in spec.LEAD_STEPS)
    sfc = "/".join(str(p) for p in spec.LANE_PARAM_SFC)
    pl = "/".join(str(p) for p in spec.LANE_PARAM_PL)
    levels = "/".join(str(l) for l in spec.LANE_LEVELS_PL)
    return [
        ["-w", f"paramId={sfc},levelType=sfc,step={steps}"],
        ["-w", f"paramId={pl},levelType=pl,level={levels},step={steps}"],
    ]


def native_root_for(root: str, block: str) -> str:
    if block in cal.VALIDATION_BLOCKS:
        return cal.validation_dir(root, "native_n320", block)
    return os.path.join(root, "native_n320")


def derived_root_for(root: str, block: str) -> str:
    if block in cal.VALIDATION_BLOCKS:
        return cal.validation_dir(root, "derived_o320", block)
    return os.path.join(root, "derived_o320")


def _validate_derived(path: str, start) -> tuple[bool, list[str], dict]:
    """Check that a start's O320 file holds exactly what the dataset build expects."""
    keys = "paramId,level,typeOfLevel,step,number,gridType,dataDate,dataTime,expver,marsClass"
    rc, out = run_cmd(["grib_get", "-p", keys, path])
    problems: list[str] = []
    if rc != 0:
        return False, ["grib_get could not read the file"], {}

    rows = [l.split() for l in out.strip().splitlines()]
    rows = [r for r in rows if len(r) == 10]
    n = len(rows)
    if n != spec.LANE_FIELDS_PER_START:
        problems.append(f"has {n} messages, expected {spec.LANE_FIELDS_PER_START}")

    grids = {r[5] for r in rows}
    if grids != {"reduced_gg"}:
        problems.append(f"grid types present are {sorted(grids)}, expected reduced_gg")

    numbers = sorted({int(r[4]) for r in rows if r[4].isdigit()})
    if numbers != spec.MEMBERS:
        problems.append(f"member numbers are {numbers}, expected {spec.MEMBERS}")

    steps = sorted({int(r[3]) for r in rows})
    if steps != sorted(spec.LEAD_STEPS):
        problems.append(f"lead times are {steps}, expected {sorted(spec.LEAD_STEPS)}")

    # Column order matches the -p list: paramId, level, typeOfLevel, step,
    # number, gridType, dataDate, dataTime, expver, marsClass.
    dates = {int(r[6]) for r in rows}
    times = {int(r[7]) for r in rows}
    if dates != {int(start.strftime("%Y%m%d"))}:
        problems.append(f"dataDate is {sorted(dates)}, expected {start:%Y%m%d}")
    if times != {start.hour * 100}:
        problems.append(f"dataTime is {sorted(times)}, expected {start.hour * 100}")

    expvers = {r[8] for r in rows}
    if expvers != {spec.OUTPUT_EXPVER}:
        problems.append(f"expver is {sorted(expvers)}, expected {spec.OUTPUT_EXPVER}")

    classes = {r[9] for r in rows}
    if classes != {spec.OUTPUT_CLASS}:
        problems.append(f"class is {sorted(classes)}, expected {spec.OUTPUT_CLASS}")

    per_member_step = collections.Counter((r[4], r[3]) for r in rows)
    wrong = {k: v for k, v in per_member_step.items() if v != spec.LANE_FIELDS_PER_MEMBER_PER_STEP}
    if wrong:
        problems.append(
            f"these member and lead-time pairs do not carry "
            f"{spec.LANE_FIELDS_PER_MEMBER_PER_STEP} fields: {dict(list(wrong.items())[:6])}"
        )

    stats = dict(messages=n, members=numbers, steps=steps)
    return not problems, problems, stats


def regrid_start(
    root: str,
    block: str,
    start,
    workdir: str,
    native_root: str | None = None,
    derived_root: str | None = None,
) -> tuple[bool, list[str], dict]:
    """Build one start's O320 file from the ten member native files."""
    key = cal.start_key(start)
    native_dir = os.path.join(native_root or native_root_for(root, block), key)
    dest = os.path.join(derived_root or derived_root_for(root, block), f"{key}.grib")
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest):
        ok, problems, stats = _validate_derived(dest, start)
        if ok:
            log(f"REGRID skip {key}: existing file validates ({stats['messages']} messages)")
            return True, [], dict(stats, skipped=True)
        move_aside(dest, "present but failed validation: " + "; ".join(problems)[:200])

    work = os.path.join(workdir, key)
    os.makedirs(work, exist_ok=True)
    pieces = []
    problems: list[str] = []

    for m in spec.MEMBERS:
        native = os.path.join(native_dir, f"m{m:02d}.grib")
        if not os.path.exists(native):
            problems.append(f"member {m} has no native forecast at {native}")
            continue

        selected = os.path.join(work, f"sel_m{m:02d}.grib")
        chunks = []
        for i, rule in enumerate(selection_rules()):
            chunk = os.path.join(work, f"sel_m{m:02d}_{i}.grib")
            rc, out = run_cmd(["grib_copy", *rule, native, chunk])
            if rc != 0 or not os.path.exists(chunk):
                problems.append(f"member {m} selection {i} failed: {out.strip()[:200]}")
                break
            chunks.append(chunk)
        if len(chunks) != len(selection_rules()):
            continue
        concat_files(chunks, selected)

        want = spec.LANE_FIELDS_PER_MEMBER_PER_STEP * len(spec.LEAD_STEPS)
        got = grib_count(selected)
        if got != want:
            problems.append(
                f"member {m}: selected {got} native fields, expected {want}; "
                f"the native file does not contain the full lane selection"
            )
            continue

        regridded = os.path.join(work, f"o320_m{m:02d}.grib")
        rc, out = run_cmd(["mir", "--grid=O320", selected, regridded])
        if rc != 0:
            problems.append(f"member {m}: mir failed: {out.strip()[:200]}")
            continue
        pieces.append(regridded)

    if len(pieces) != len(spec.MEMBERS):
        problems.insert(0, f"only {len(pieces)} of {len(spec.MEMBERS)} members could be regridded")
        return False, problems, {}

    tmp = f"{dest}.tmp.{os.getpid()}"
    concat_files(pieces, tmp)
    ok, more, stats = _validate_derived(tmp, start)
    if not ok:
        move_aside(tmp, "failed validation: " + "; ".join(more)[:200])
        return False, problems + more, stats
    os.replace(tmp, dest)
    stats["bytes"] = os.path.getsize(dest)
    return True, problems, stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block", required=True)
    p.add_argument("--root", default=os.environ.get("AIFSENS2_ROOT", DEFAULT_ROOT))
    p.add_argument("--workdir", help="scratch space for the intermediate selections")
    p.add_argument("--native-root", help="override the directory the native forecasts are read from")
    p.add_argument("--derived-root", help="override the destination for the O320 files")
    a = p.parse_args(argv)

    starts = cal.block_starts(a.block, a.root)
    if not starts:
        log(f"FATAL block {a.block} has no starts")
        return 1

    workdir = a.workdir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"aifsens2_regrid_{a.block}_{os.getpid()}"
    )
    os.makedirs(workdir, exist_ok=True)
    log(f"block {a.block}: regridding {len(starts)} starts, working in {workdir}")

    ok_count = 0
    results = {}
    for start in starts:
        key = cal.start_key(start)
        t0 = time.time()
        ok, problems, stats = regrid_start(
            a.root, a.block, start, workdir, a.native_root, a.derived_root
        )
        elapsed = time.time() - t0
        if ok:
            ok_count += 1
            log(
                f"REGRID_OK {key}: {stats.get('messages')} messages, "
                f"{stats.get('bytes', 0) / (1 << 20):.0f} MiB, {elapsed:.0f} s"
                + (" (skipped, already present)" if stats.get("skipped") else "")
            )
        else:
            log(f"REGRID_FAILED {key}: {'; '.join(problems)}")
        results[key] = dict(ok=ok, problems=problems, seconds=round(elapsed, 1), **stats)

    atomic_write_json(
        os.path.join(a.root, "manifests", f"regrid_{a.block}.json"),
        dict(block=a.block, starts=results, complete=ok_count, total=len(starts)),
    )
    log(f"REGRID_SUMMARY block={a.block} complete={ok_count} of {len(starts)}")
    if ok_count != len(starts):
        log(f"REGRID_RC=1 block={a.block}")
        return 1
    log(f"REGRID_RC=0 block={a.block}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
