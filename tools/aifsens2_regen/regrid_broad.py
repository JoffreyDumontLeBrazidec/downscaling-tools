"""Stage 4, broad variant: select 108 variables, regrid them to O320 and
re-encode them the way the MARS archive encodes the same fields.

Why this module exists
----------------------
The training campaign needs one input store that covers the whole of 2026 from
January to the end of August.  The part from 12 May onwards is retrieved
straight from the MARS archive, because the operational AIFS ensemble version 2
began on that date.  The part before it does not exist in the archive at all
and was regenerated on our own machine, so it only exists as native N320 GRIB
files.  The regeneration's own stage 4, aifsens2_regen.regrid, converts those
native files to O320 but keeps a narrow selection of 68 variables.  The archive
half of the campaign holds 108.  Two stores can only be opened as one dataset
if they hold the same variables under the same names, so this module produces
the same 108 variables from the native files, encoded so that the names come
out identical.

What it does differently from the narrow stage 4
------------------------------------------------
Three things.

First, the selection is wider: fifteen instantaneous surface fields, six
accumulations, four soil fields and the upper-air fields on fourteen pressure
levels, with specific humidity on thirteen, which is 108 fields per member and
lead time.

Second, some headers are rewritten after the regrid so that the parameters
carry the identifiers the archive uses.  The soil fields are the awkward case:
the native forecast writes volumetric soil water and soil temperature as
swvl1, swvl2, stl1 and stl2 on typeOfLevel depthBelowLandLayer in GRIB edition
1, whereas the archive writes them as vsw and sot on typeOfLevel soilLayer,
layers 1 and 2, which only exist in GRIB edition 2.  Those messages are
therefore raised to edition 2 before the parameter and level are set.  Runoff,
total precipitation, convective precipitation and snowfall are simply renamed
to the identifiers the archive uses.  Everything else already agrees, either
exactly or in the only respect that matters, which is the shortName the dataset
build turns into a variable name.

Third, the accumulations are re-windowed.  The model accumulates from the start
of the forecast, so the native step-12 field is the total over the first twelve
hours, while every accumulated variable in the finished store has to mean the
six hours before the valid time.  The step-6 field already means that and is
kept as it is.  The step-12 field is replaced by the step-12 values minus the
step-6 values and is stamped with stepRange 6-12.  The subtraction is done on
the O320 values, after the interpolation, so that the arithmetic and the
interpolation cannot interact.

Output layout
-------------
One file per forecast start, holding every member and both lead times:

    derived_o320_broad/<YYYYMMDD_HH>.grib    2160 messages
                                             = 10 members x 2 steps x 108 fields

The directory name deliberately differs from the regeneration's
derived_o320/ so that the two selections can never be confused.

As everywhere else in this pipeline, the result is written under a temporary
name, validated, and only then renamed into place, so that an interrupted job
never leaves a half-written file behind that a later stage would trust.

Usage
-----
    python -m aifsens2_regen.regrid_broad --block 202601
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time

import numpy as np

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

# The name of the destination directory.  It must not collide with the
# regeneration's own derived_o320/.
DERIVED_DIRNAME = "derived_o320_broad"

# The level types whose level number becomes part of the store variable name.
LEVELLED_TYPES = {"isobaricInhPa", "soilLayer"}


def selection_rules() -> list[list[str]]:
    """The grib_copy selections that together pick the 108 broad variables.

    Three selections are needed.  The surface, soil and accumulated fields all
    live on levelType sfc and can be asked for together, without any level
    constraint, because each of them is identified by its parameter alone.  The
    five upper-air parameters need a level constraint of fourteen levels, and
    specific humidity a different one of thirteen, so they cannot share a rule.
    """
    steps = "/".join(str(s) for s in spec.LEAD_STEPS)
    sfc_params = (
        spec.BROAD_PARAM_SFC_INSTANT
        + spec.BROAD_PARAM_SFC_ACCUM
        + spec.BROAD_PARAM_SOIL
    )
    sfc = "/".join(str(p) for p in sfc_params)
    pl = "/".join(str(p) for p in spec.BROAD_PARAM_PL)
    pl_levels = "/".join(str(l) for l in spec.BROAD_LEVELS_PL)
    q = "/".join(str(p) for p in spec.BROAD_PARAM_Q)
    q_levels = "/".join(str(l) for l in spec.BROAD_LEVELS_Q)
    return [
        ["-w", f"paramId={sfc},levelType=sfc,step={steps}"],
        ["-w", f"paramId={pl},levelType=pl,level={pl_levels},step={steps}"],
        ["-w", f"paramId={q},levelType=pl,level={q_levels},step={steps}"],
    ]


def native_root_for(root: str, block: str) -> str:
    if block in cal.VALIDATION_BLOCKS:
        return cal.validation_dir(root, "native_n320", block)
    return os.path.join(root, "native_n320")


def derived_root_for(root: str, block: str) -> str:
    if block in cal.VALIDATION_BLOCKS:
        return cal.validation_dir(root, DERIVED_DIRNAME, block)
    return os.path.join(root, DERIVED_DIRNAME)


def store_variable_name(short_name: str, type_of_level: str, level) -> str:
    """The name the dataset build would give a field with this header.

    The build remaps every field to "{param}_{levelist}" and then drops the
    level part for fields that have no level, which in practice means every
    field whose level type is not a pressure level or a soil layer.
    """
    if type_of_level in LEVELLED_TYPES:
        return f"{short_name}_{int(level)}"
    return short_name


# --------------------------------------------------------------------------
# The header rewrite and the accumulation differencing.
# --------------------------------------------------------------------------


def rewrite_member(src: str, dst: str) -> list[str]:
    """Re-encode one member's regridded file into its final form.

    The file arrives holding 216 messages, 108 for each of the two lead times,
    exactly as mir wrote them.  It leaves holding the same 216 messages with
    the soil fields renamed to the archive's parameters and level type, the
    four renamed accumulations carrying the archive's identifiers, and the
    step-12 accumulations replaced by the difference against step 6 and stamped
    with stepRange 6-12.

    Returns a list of problems; an empty list means the rewrite succeeded.
    """
    import eccodes as ec

    problems: list[str] = []

    # First pass: remember the step-6 value array of every accumulated field,
    # keyed by the parameter identifier the native file uses.  Only six arrays
    # are held at a time, so the memory cost is small.
    base: dict[int, np.ndarray] = {}
    with open(src, "rb") as f:
        while True:
            h = ec.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                param = int(ec.codes_get(h, "paramId"))
                if param in spec.BROAD_PARAM_SFC_ACCUM:
                    if int(ec.codes_get(h, "endStep")) == 6:
                        base[param] = ec.codes_get_values(h)
            finally:
                ec.codes_release(h)

    missing = [p for p in spec.BROAD_PARAM_SFC_ACCUM if p not in base]
    if missing:
        problems.append(
            "these accumulated parameters have no step-6 field to subtract: "
            f"{missing}"
        )
        return problems

    # Second pass: write every message out, changing the ones that need it.
    with open(src, "rb") as f, open(dst, "wb") as out:
        while True:
            h = ec.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                param = int(ec.codes_get(h, "paramId"))

                if param in spec.BROAD_SOIL_ENCODING:
                    new_param, new_level = spec.BROAD_SOIL_ENCODING[param]
                    # The parameter and the level type only exist in GRIB
                    # edition 2, so the message is raised first.  The values
                    # survive the change untouched because the packing stays
                    # grid_simple at the same number of bits.
                    ec.codes_set(h, "editionNumber", 2)
                    ec.codes_set(h, "paramId", new_param)
                    ec.codes_set(h, "typeOfLevel", "soilLayer")
                    ec.codes_set(h, "level", new_level)

                elif param in spec.BROAD_PARAM_SFC_ACCUM:
                    end_step = int(ec.codes_get(h, "endStep"))
                    if end_step == 12:
                        values = ec.codes_get_values(h)
                        missing_value = ec.codes_get(h, "missingValue")
                        earlier = base[param]
                        if earlier.shape != values.shape:
                            problems.append(
                                f"parameter {param}: the step-6 field has "
                                f"{earlier.size} values and the step-12 field "
                                f"{values.size}, so they cannot be differenced"
                            )
                            continue
                        gap = values == missing_value
                        difference = np.where(
                            gap, missing_value, values - earlier
                        )
                        ec.codes_set(h, "startStep", 6)
                        ec.codes_set(h, "endStep", 12)
                        ec.codes_set_values(h, difference)
                    new_param = spec.BROAD_ACCUM_PARAM_RENAME.get(param)
                    if new_param is not None:
                        ec.codes_set(h, "paramId", new_param)

                ec.codes_write(h, out)
            finally:
                ec.codes_release(h)

    return problems


# --------------------------------------------------------------------------
# Validation.
# --------------------------------------------------------------------------


def _validate_derived(path: str, start) -> tuple[bool, list[str], dict]:
    """Check that a start's broad O320 file holds exactly what the build expects."""
    keys = (
        "paramId,shortName,level,typeOfLevel,step,stepRange,number,gridType,"
        "dataDate,dataTime,expver,marsClass"
    )
    rc, out = run_cmd(["grib_get", "-p", keys, path])
    problems: list[str] = []
    if rc != 0:
        return False, ["grib_get could not read the file"], {}

    rows = [l.split() for l in out.strip().splitlines()]
    rows = [r for r in rows if len(r) == 12]
    n = len(rows)
    if n != spec.BROAD_FIELDS_PER_START:
        problems.append(f"has {n} messages, expected {spec.BROAD_FIELDS_PER_START}")

    grids = {r[7] for r in rows}
    if grids != {"reduced_gg"}:
        problems.append(f"grid types present are {sorted(grids)}, expected reduced_gg")

    numbers = sorted({int(r[6]) for r in rows if r[6].lstrip("-").isdigit()})
    if numbers != spec.MEMBERS:
        problems.append(f"member numbers are {numbers}, expected {spec.MEMBERS}")

    steps = sorted({int(r[4]) for r in rows})
    if steps != sorted(spec.LEAD_STEPS):
        problems.append(f"lead times are {steps}, expected {sorted(spec.LEAD_STEPS)}")

    dates = {int(r[8]) for r in rows}
    times = {int(r[9]) for r in rows}
    if dates != {int(start.strftime("%Y%m%d"))}:
        problems.append(f"dataDate is {sorted(dates)}, expected {start:%Y%m%d}")
    if times != {start.hour * 100}:
        problems.append(f"dataTime is {sorted(times)}, expected {start.hour * 100}")

    expvers = {r[10] for r in rows}
    if expvers != {spec.OUTPUT_EXPVER}:
        problems.append(f"expver is {sorted(expvers)}, expected {spec.OUTPUT_EXPVER}")

    classes = {r[11] for r in rows}
    if classes != {spec.OUTPUT_CLASS}:
        problems.append(f"class is {sorted(classes)}, expected {spec.OUTPUT_CLASS}")

    per_member_step = collections.Counter((r[6], r[4]) for r in rows)
    wrong = {
        k: v
        for k, v in per_member_step.items()
        if v != spec.BROAD_FIELDS_PER_MEMBER_PER_STEP
    }
    if wrong:
        problems.append(
            "these member and lead-time pairs do not carry "
            f"{spec.BROAD_FIELDS_PER_MEMBER_PER_STEP} fields: "
            f"{dict(list(wrong.items())[:6])}"
        )

    names = sorted({store_variable_name(r[1], r[3], r[2]) for r in rows})
    if names != spec.BROAD_STORE_VARIABLES:
        extra = sorted(set(names) - set(spec.BROAD_STORE_VARIABLES))
        absent = sorted(set(spec.BROAD_STORE_VARIABLES) - set(names))
        problems.append(
            f"the variable names do not match the archive half: "
            f"unexpected {extra}, absent {absent}"
        )

    # Every accumulated field must carry the six-hour window that ends at its
    # own valid time, which means 0-6 at step 6 and 6-12 at step 12.
    accum_params = set(spec.BROAD_ACCUM_PARAM_RENAME.values()) | {169, 175}
    windows = collections.Counter(
        (r[4], r[5]) for r in rows if int(r[0]) in accum_params
    )
    unexpected = {k: v for k, v in windows.items() if k not in {("6", "0-6"), ("12", "6-12")}}
    if unexpected:
        problems.append(
            f"these accumulation windows are wrong (step, stepRange): {unexpected}"
        )

    stats = dict(messages=n, members=numbers, steps=steps, variables=len(names))
    return not problems, problems, stats


# --------------------------------------------------------------------------
# The per-start work.
# --------------------------------------------------------------------------


def regrid_start(
    root: str,
    block: str,
    start,
    workdir: str,
    native_root: str | None = None,
    derived_root: str | None = None,
) -> tuple[bool, list[str], dict]:
    """Build one start's broad O320 file from the ten member native files."""
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
    rules = selection_rules()

    for m in spec.MEMBERS:
        native = os.path.join(native_dir, f"m{m:02d}.grib")
        if not os.path.exists(native):
            problems.append(f"member {m} has no native forecast at {native}")
            continue

        selected = os.path.join(work, f"sel_m{m:02d}.grib")
        chunks = []
        for i, rule in enumerate(rules):
            chunk = os.path.join(work, f"sel_m{m:02d}_{i}.grib")
            rc, out = run_cmd(["grib_copy", *rule, native, chunk])
            if rc != 0 or not os.path.exists(chunk):
                problems.append(f"member {m} selection {i} failed: {out.strip()[:200]}")
                break
            chunks.append(chunk)
        if len(chunks) != len(rules):
            continue
        concat_files(chunks, selected)

        want = spec.BROAD_FIELDS_PER_MEMBER_PER_STEP * len(spec.LEAD_STEPS)
        got = grib_count(selected)
        if got != want:
            problems.append(
                f"member {m}: selected {got} native fields, expected {want}; "
                f"the native file does not contain the full broad selection"
            )
            continue

        regridded = os.path.join(work, f"o320_m{m:02d}.grib")
        rc, out = run_cmd(["mir", "--grid=O320", selected, regridded])
        if rc != 0:
            problems.append(f"member {m}: mir failed: {out.strip()[:200]}")
            continue

        final = os.path.join(work, f"final_m{m:02d}.grib")
        trouble = rewrite_member(regridded, final)
        if trouble:
            problems.extend(f"member {m}: {t}" for t in trouble)
            continue
        if grib_count(final) != want:
            problems.append(
                f"member {m}: the rewrite produced {grib_count(final)} messages, "
                f"expected {want}"
            )
            continue
        pieces.append(final)

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
    p.add_argument(
        "--manifest-root",
        help="override the directory the JSON manifest is written into "
        "(it defaults to manifests/ under --root)",
    )
    p.add_argument(
        "--start",
        action="append",
        help="restrict the work to this start, given as YYYYMMDD_HH; may be repeated",
    )
    a = p.parse_args(argv)

    starts = cal.block_starts(a.block, a.root)
    if not starts:
        log(f"FATAL block {a.block} has no starts")
        return 1
    if a.start:
        wanted = set(a.start)
        starts = [s for s in starts if cal.start_key(s) in wanted]
        if not starts:
            log(f"FATAL none of the requested starts belong to block {a.block}")
            return 1

    workdir = a.workdir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"aifsens2_regrid_broad_{a.block}_{os.getpid()}"
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

    manifest_root = a.manifest_root or os.path.join(a.root, "manifests")
    atomic_write_json(
        os.path.join(manifest_root, f"regrid_broad_{a.block}.json"),
        dict(
            block=a.block,
            variables=spec.BROAD_STORE_VARIABLES,
            starts=results,
            complete=ok_count,
            total=len(starts),
        ),
    )
    log(f"REGRID_SUMMARY block={a.block} complete={ok_count} of {len(starts)}")
    if ok_count != len(starts):
        log(f"REGRID_RC=1 block={a.block}")
        return 1
    log(f"REGRID_RC=0 block={a.block}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
