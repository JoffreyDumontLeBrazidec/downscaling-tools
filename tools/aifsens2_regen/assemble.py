"""Stage 2: turn the grouped raw files into one initial condition per member.

The retrieval leaves a handful of large files, each holding one field group at
one time of day for every date and every member of the block.  The inference
step needs the opposite arrangement: one file per forecast start and per
ensemble member, holding the 111 fields of the analysis six hours before the
start followed by the 111 fields of the analysis at the start.

The work is done in two phases.  The first phase splits every raw file into
small pieces keyed by date, time and member, using grib_copy, which is fast
because it copies messages without decoding them.  The second phase
concatenates the right four pieces for each start and member, validates the
result and only then renames it into place.

Validation is deliberately strict, because a silently wrong initial condition
would produce a forecast that looks fine and is not.  A member file is accepted
only if it has exactly 222 messages, exactly two analysis times six hours
apart, one single ensemble member number on all the perturbed messages and none
on the constants, the reduced Gaussian N320 grid throughout, and exactly the
expected set of parameters and levels at each of the two times.

A member that cannot be built, because a field is missing from the archive, is
recorded as incomplete and skipped.  It is never patched with the control
member or with a neighbouring member: a gap must stay visible.  The job still
builds every other member and exits non-zero at the end.

Usage
-----
    python -m aifsens2_regen.assemble --block pilot_20260101
    python -m aifsens2_regen.assemble --block summer_validation --members-root validation/ic_members
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import os
import sys

from . import calendar as cal
from . import gribspec as spec
from .common import (
    DEFAULT_ROOT,
    atomic_write_json,
    concat_files,
    grib_count,
    log,
    move_aside,
    read_json,
    run_cmd,
    sha256_file,
)
from .retrieve import raw_dir


def expected_paramid_counts() -> dict[int, int]:
    """How many messages of each parameter one input time must contain."""
    c: collections.Counter[int] = collections.Counter()
    for p in spec.PARAM_SFC:
        c[p] += 1
    for p in spec.PARAM_PL:
        c[p] += len(spec.LEVELS_PL)
    for p in spec.PARAM_Q:
        c[p] += len(spec.LEVELS_Q)
    for p in spec.PARAM_WAVE:
        c[p] += 1
    for p in spec.PARAM_CON:
        c[p] += 1
    return dict(c)


def expected_pl_pairs() -> set[tuple[int, int]]:
    """The parameter and level combinations on pressure levels."""
    pairs = {(p, l) for p in spec.PARAM_PL for l in spec.LEVELS_PL}
    pairs |= {(p, l) for p in spec.PARAM_Q for l in spec.LEVELS_Q}
    return pairs


# --------------------------------------------------------------------------
# Phase 1: split the raw files.
# --------------------------------------------------------------------------


def is_constants_file(path: str) -> bool:
    """Whether a raw file holds the invariant analysis fields.

    Decided by reading the stream of the first message rather than by trusting
    the file name, because the pilot day, the summer probe and the monthly
    blocks all use slightly different naming.
    """
    rc, out = run_cmd(["grib_get", "-w", "count=1", "-p", "marsStream", path])
    return rc == 0 and out.strip().split()[0] == "oper"


def split_raw(root: str, block: str, parts: str) -> list[str]:
    """Split every raw file of the block into per-date, per-time, per-member pieces.

    Returns the list of raw files that were used.  A raw file whose split has
    already been done, judged by a marker recording its size, is not split
    again, which is what makes the stage cheap to re-run.
    """
    os.makedirs(parts, exist_ok=True)
    markers = os.path.join(parts, ".split_done")
    os.makedirs(markers, exist_ok=True)

    d = raw_dir(root, block)
    # The summer block keeps one directory per date, and four of its groups
    # live read-only in the earlier study directory.
    patterns = [os.path.join(d, "*.grib"), os.path.join(d, "*", "*.grib")]
    raws = sorted({p for pat in patterns for p in glob.glob(pat)})
    # The second pattern also matches this stage's own output, and re-splitting
    # a piece onto itself makes grib_copy abort.  Anything inside the parts
    # directory, or inside a directory moved aside earlier, is not raw input.
    raws = [
        p
        for p in raws
        if os.path.dirname(os.path.abspath(p)) != os.path.abspath(parts)
        and not os.path.basename(os.path.dirname(p)).startswith("_aside_")
    ]
    if block == cal.SUMMER_BLOCK:
        raws += sorted(
            glob.glob(
                "/home/ecm5702/scratch/eval/aifs_v1_vs_regen_20260909/regen/"
                "ic_v2_2026_pf/grp_{sfc_t0,wave_t6,wave_t0,con}.grib"
            )
        ) or [
            f"/home/ecm5702/scratch/eval/aifs_v1_vs_regen_20260909/regen/ic_v2_2026_pf/grp_{n}.grib"
            for n in ("sfc_t0", "wave_t6", "wave_t0", "con")
        ]
        raws = sorted({p for p in raws if os.path.exists(p)})

    if not raws:
        log(f"FATAL no raw files found for block {block} under {d}")
        sys.exit(1)

    # All the perturbed groups have to be split in ONE grib_copy invocation, and
    # likewise the constants.  When grib_copy writes to a name template it
    # truncates each output file the first time that invocation opens it, so a
    # second invocation writing to the same template does not add to the pieces
    # from the first, it replaces them.  Splitting group by group therefore
    # leaves only the group that happened to run last, which is exactly the
    # failure this code hit on 2026-09-10: every piece held eleven wave fields
    # and nothing else.
    perturbed = [r for r in raws if not is_constants_file(r)]
    constants = [r for r in raws if is_constants_file(r)]

    for label, group, template in (
        ("perturbed", perturbed, "pf_[dataDate]_[dataTime]_[number].grib"),
        ("constants", constants, "con_[dataDate]_[dataTime].grib"),
    ):
        if not group:
            log(f"SPLIT no {label} raw files found for block {block}")
            continue

        signature = sorted([os.path.basename(p), os.path.getsize(p)] for p in group)
        marker = os.path.join(markers, f"{label}.json")
        prev = read_json(marker)
        if prev and prev.get("signature") == signature:
            log(
                f"SPLIT skip {label}: the same {len(group)} raw files were already "
                f"split into {parts}"
            )
            continue

        log(f"SPLIT {label}: {len(group)} raw files -> {template}")
        rc, out = run_cmd(["grib_copy", *group, os.path.join(parts, template)])
        if rc != 0:
            log(f"SPLIT_FAILED {label}: {out.strip()[:600]}")
            sys.exit(1)
        atomic_write_json(marker, {"signature": signature, "sources": group})

    return raws


def part_path(parts: str, when: dt.datetime, member: int | None) -> str:
    """The piece holding one analysis time, for one member or for the constants.

    grib_copy expands [dataTime] as four digits with leading zeros, so 00 UTC
    becomes 0000 and 06 UTC becomes 0600.  This was checked against the files
    grib_copy actually wrote rather than assumed; an earlier version of this
    function guessed "0" and "600" and found nothing.
    """
    date = when.strftime("%Y%m%d")
    time = f"{when.hour * 100:04d}"
    if member is None:
        return os.path.join(parts, f"con_{date}_{time}.grib")
    return os.path.join(parts, f"pf_{date}_{time}_{member}.grib")


# --------------------------------------------------------------------------
# Phase 2: build and validate the member files.
# --------------------------------------------------------------------------


def inspect_messages(path: str) -> list[dict]:
    """Read the identifying keys of every message in a file.

    eccodes is used through its command line rather than its Python bindings so
    that the stage runs on a plain compute node without the inference
    environment.
    """
    keys = "paramId,dataDate,dataTime,level,typeOfLevel,shortName,gridType,N,number"
    rc, out = run_cmd(["grib_get", "-p", keys, path])
    if rc != 0:
        return []
    msgs = []
    for line in out.strip().splitlines():
        f = line.split()
        if len(f) != 9:
            continue
        msgs.append(
            dict(
                paramId=int(f[0]),
                dataDate=int(f[1]),
                dataTime=int(f[2]),
                level=int(f[3]),
                typeOfLevel=f[4],
                shortName=f[5],
                gridType=f[6],
                N=f[7],
                number=f[8],
            )
        )
    return msgs


def validate_member(path: str, start: dt.datetime, member: int) -> tuple[bool, list[str]]:
    """Check one assembled member file.  Returns whether it passed and why not."""
    problems: list[str] = []
    msgs = inspect_messages(path)

    if len(msgs) != spec.FIELDS_PER_MEMBER_FILE:
        problems.append(
            f"has {len(msgs)} messages, expected {spec.FIELDS_PER_MEMBER_FILE}"
        )
        if not msgs:
            return False, problems

    # The two analysis times, six hours apart, ending at the start.
    stamps = sorted({(m["dataDate"], m["dataTime"]) for m in msgs})
    if len(stamps) != 2:
        problems.append(f"has {len(stamps)} distinct analysis times, expected 2: {stamps}")
    else:
        def to_dt(s):
            return dt.datetime.strptime(f"{s[0]:08d}{s[1] // 100:02d}", "%Y%m%d%H")

        t6, t0 = to_dt(stamps[0]), to_dt(stamps[1])
        if t0 - t6 != dt.timedelta(hours=6):
            problems.append(f"analysis times are {t0 - t6} apart, expected 6 hours")
        if t0 != start:
            problems.append(f"latest analysis time is {t0}, expected the start {start}")

    # One member number on the perturbed messages, and no member number on the
    # constants.  The constants come from the operational analysis, which is not
    # an ensemble, and eccodes reports their number as 0 rather than leaving the
    # key unset.  A 0 here therefore means "not an ensemble field", not "the
    # control member": the control is never retrieved by this pipeline, so a
    # perturbed message can never legitimately carry 0.
    no_member = ("", "0", "MISSING", "None")
    numbers = {m["number"] for m in msgs if m["number"] not in no_member}
    if numbers != {str(member)}:
        problems.append(f"carries member numbers {sorted(numbers)}, expected only {member}")
    n_constants = sum(1 for m in msgs if m["number"] in no_member)
    if n_constants != 2 * spec.N_CON_PER_TIME:
        problems.append(
            f"has {n_constants} messages that are not ensemble fields, "
            f"expected {2 * spec.N_CON_PER_TIME} constants"
        )
    n_perturbed = len(msgs) - n_constants
    if n_perturbed != 2 * spec.N_PERTURBED_PER_MEMBER_PER_TIME:
        problems.append(
            f"has {n_perturbed} perturbed messages, "
            f"expected {2 * spec.N_PERTURBED_PER_MEMBER_PER_TIME}"
        )

    # The grid.
    grids = {(m["gridType"], m["N"]) for m in msgs}
    if grids != {("reduced_gg", "320")}:
        problems.append(f"grid is {sorted(grids)}, expected reduced_gg N320")

    # The parameter composition of each analysis time.
    want_counts = expected_paramid_counts()
    want_pairs = expected_pl_pairs()
    for stamp in stamps:
        at = [m for m in msgs if (m["dataDate"], m["dataTime"]) == stamp]
        got = collections.Counter(m["paramId"] for m in at)
        if dict(got) != want_counts:
            missing = {p: n for p, n in want_counts.items() if got.get(p, 0) != n}
            extra = {p: n for p, n in got.items() if p not in want_counts}
            problems.append(
                f"at {stamp} the parameter counts are wrong; "
                f"wrong or missing {missing}, unexpected {extra}"
            )
        got_pairs = {
            (m["paramId"], m["level"]) for m in at if m["typeOfLevel"] == "isobaricInhPa"
        }
        if got_pairs != want_pairs:
            problems.append(
                f"at {stamp} the pressure-level combinations are wrong; "
                f"missing {sorted(want_pairs - got_pairs)[:8]}, "
                f"unexpected {sorted(got_pairs - want_pairs)[:8]}"
            )
        # Soil temperature must sit on its soil layer, not on the surface.
        soil = [m for m in at if m["paramId"] == 139]
        if len(soil) != 1:
            problems.append(f"at {stamp} found {len(soil)} soil temperature messages, expected 1")
        elif "depthBelowLand" not in soil[0]["typeOfLevel"]:
            problems.append(
                f"at {stamp} soil temperature is on level type "
                f"{soil[0]['typeOfLevel']}, expected a depth below land layer"
            )

    return not problems, problems


def build_member(parts: str, dest: str, start: dt.datetime, member: int) -> tuple[bool, list[str]]:
    """Assemble one member file and validate it, writing it atomically."""
    t6 = start - cal.INPUT_LEAD
    sources = [
        part_path(parts, t6, member),
        part_path(parts, t6, None),
        part_path(parts, start, member),
        part_path(parts, start, None),
    ]
    missing = [s for s in sources if not os.path.exists(s) or os.path.getsize(s) == 0]
    if missing:
        return False, [f"missing input pieces: {[os.path.basename(m) for m in missing]}"]

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = f"{dest}.tmp.{os.getpid()}"
    concat_files(sources, tmp)
    ok, problems = validate_member(tmp, start, member)
    if ok:
        os.replace(tmp, dest)
    else:
        move_aside(tmp, "failed validation: " + "; ".join(problems)[:200])
    return ok, problems


def members_root(root: str, block: str) -> str:
    if block == cal.SUMMER_BLOCK:
        return os.path.join(root, "validation", "ic_members")
    return os.path.join(root, "ic", "members")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block", required=True)
    p.add_argument("--root", default=os.environ.get("AIFSENS2_ROOT", DEFAULT_ROOT))
    p.add_argument("--members-root", help="override the destination for member files")
    a = p.parse_args(argv)

    starts = cal.block_starts(a.block, a.root)
    if not starts:
        log(f"FATAL block {a.block} has no starts")
        return 1
    log(f"block {a.block}: {len(starts)} starts, {len(spec.MEMBERS)} members each")

    parts = os.path.join(raw_dir(a.root, a.block), "parts")
    split_raw(a.root, a.block, parts)

    dest_root = a.members_root or members_root(a.root, a.block)
    complete = incomplete = skipped = 0

    for start in starts:
        key = cal.start_key(start)
        out_dir = os.path.join(dest_root, key)
        entries = {}
        for m in spec.MEMBERS:
            dest = os.path.join(out_dir, f"m{m:02d}.grib")
            if os.path.exists(dest):
                ok, problems = validate_member(dest, start, m)
                if ok:
                    skipped += 1
                    complete += 1
                    entries[f"m{m:02d}"] = dict(
                        status="complete",
                        path=dest,
                        fields=spec.FIELDS_PER_MEMBER_FILE,
                        sha256=None,
                        note="already present and valid, not rebuilt",
                    )
                    continue
                move_aside(dest, "present but failed validation: " + "; ".join(problems)[:200])

            ok, problems = build_member(parts, dest, start, m)
            if ok:
                complete += 1
                entries[f"m{m:02d}"] = dict(
                    status="complete",
                    path=dest,
                    fields=spec.FIELDS_PER_MEMBER_FILE,
                    bytes=os.path.getsize(dest),
                    sha256=sha256_file(dest),
                )
            else:
                incomplete += 1
                log(f"INCOMPLETE {key} member {m}: {'; '.join(problems)}")
                entries[f"m{m:02d}"] = dict(status="incomplete", problems=problems)

        atomic_write_json(
            os.path.join(out_dir, "manifest.json"),
            dict(
                block=a.block,
                start=cal.start_key(start),
                start_iso=start.isoformat(),
                input_times=[
                    (start - cal.INPUT_LEAD).isoformat(),
                    start.isoformat(),
                ],
                fields_per_member=spec.FIELDS_PER_MEMBER_FILE,
                members=entries,
                complete=sum(1 for e in entries.values() if e["status"] == "complete"),
            ),
        )
        log(
            f"ASSEMBLE {key}: "
            f"{sum(1 for e in entries.values() if e['status'] == 'complete')} of "
            f"{len(spec.MEMBERS)} members complete"
        )

    want = len(starts) * len(spec.MEMBERS)
    log(
        f"ASSEMBLE_SUMMARY block={a.block} complete={complete} of {want} "
        f"(of which {skipped} were already present), incomplete={incomplete}"
    )
    if incomplete:
        log(f"ASSEMBLE_RC=1 block={a.block}: {incomplete} member files could not be built")
        return 1
    log(f"ASSEMBLE_RC=0 block={a.block}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
