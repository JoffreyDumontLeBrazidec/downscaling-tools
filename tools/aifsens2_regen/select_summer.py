"""Decide which summer 2026 validation dates the archive was able to serve.

The summer validation set exists to run the production code on dates outside
the campaign calendar before the campaign itself is produced.  Its initial
conditions come from two places.  Five field groups were fetched date by date
by a probe, because the whole-summer request failed twice on an unavailable
tape.  The remaining four groups survive from the earlier study of 2026-09-09
and are read, never written, from that study's directory.

A date is usable only if all nine groups are present and complete for it.  A
date missing even one group is excluded rather than patched, because a forecast
started from an incomplete initial condition is wrong in a way that does not
announce itself.

The result is written to validation/selected_dates.json together with the
criterion used and the per-date evidence, so that the choice can be re-read
later without re-running the check.

Usage
-----
    python -m aifsens2_regen.select_summer
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import os
import sys

from . import calendar as cal
from . import gribspec as spec
from .common import DEFAULT_ROOT, atomic_write_json, log, run_cmd

# The four groups that survive from the earlier study.  They cover all ten
# candidate dates in one file each, so their contents must be indexed by date
# before a per-date verdict can be given.
SHARED_DIR = "/home/ecm5702/scratch/eval/aifs_v1_vs_regen_20260909/regen/ic_v2_2026_pf"
SHARED_GROUPS = {
    # name in that directory -> (which analysis time it holds, fields per date)
    "grp_sfc_t0.grib": ("t0", spec.GROUPS["sfc"]["per_date_per_time"]),
    "grp_wave_t6.grib": ("t6", spec.GROUPS["wave"]["per_date_per_time"]),
    "grp_wave_t0.grib": ("t0", spec.GROUPS["wave"]["per_date_per_time"]),
    "grp_con.grib": ("both", None),
}

# The five groups the probe fetched, one directory per date.
PROBE_GROUPS = {
    "grp_sfc_t6.grib": spec.GROUPS["sfc"]["per_date_per_time"],
    "grp_pl_t6.grib": spec.GROUPS["pl"]["per_date_per_time"],
    "grp_pl_t0.grib": spec.GROUPS["pl"]["per_date_per_time"],
    "grp_q_t6.grib": spec.GROUPS["q"]["per_date_per_time"],
    "grp_q_t0.grib": spec.GROUPS["q"]["per_date_per_time"],
}


def index_by_date_and_time(path: str) -> dict[tuple[int, int], int]:
    """Count the messages of a file by analysis date and analysis time.

    Counting by date alone is not enough for the invariant fields.  The
    2026-09-09 request asked for both 0000 and 1800 on every date, so that file
    holds eight messages per date, of which only four belong to any one
    analysis time.  A date-only count made every date look as though it had
    twice the constants it needed.
    """
    rc, out = run_cmd(["grib_get", "-p", "dataDate,dataTime", path])
    if rc != 0:
        return {}
    c: collections.Counter = collections.Counter()
    for line in out.strip().splitlines():
        f = line.split()
        if len(f) != 2:
            continue
        c[(int(f[0]), int(f[1]))] += 1
    return dict(c)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=os.environ.get("AIFSENS2_ROOT", DEFAULT_ROOT))
    a = p.parse_args(argv)

    probe_root = os.path.join(a.root, "validation", "ic_raw")

    # Index the shared files once.
    shared_counts: dict[str, dict[int, int]] = {}
    for name in SHARED_GROUPS:
        path = os.path.join(SHARED_DIR, name)
        if not os.path.exists(path):
            log(f"shared group {name} is absent from {SHARED_DIR}")
            shared_counts[name] = {}
            continue
        counts = index_by_date_and_time(path)
        shared_counts[name] = counts
        dates = {d for d, _ in counts}
        log(
            f"shared group {name}: {sum(counts.values())} messages over "
            f"{len(dates)} dates and {len({tm for _, tm in counts})} analysis times"
        )

    evidence = {}
    selected = []
    for date in cal.SUMMER_CANDIDATE_DATES:
        d = int(date)
        day = _dt.datetime.strptime(date, "%Y%m%d")
        prev = int((day - _dt.timedelta(days=1)).strftime("%Y%m%d"))

        rec = {"probe": {}, "shared": {}, "usable": True, "reasons": []}

        for name, want in PROBE_GROUPS.items():
            path = os.path.join(probe_root, date, name)
            got = 0
            if os.path.exists(path):
                rc, out = run_cmd(["grib_count", path])
                got = int(out.strip().split()[0]) if rc == 0 and out.strip() else 0
            rec["probe"][name] = {"fields": got, "expected": want}
            if got != want:
                rec["usable"] = False
                rec["reasons"].append(f"{name} has {got} of {want} fields")

        for name, (which, want) in SHARED_GROUPS.items():
            counts = shared_counts.get(name, {})
            if name == "grp_con.grib":
                # Four invariant fields at the start itself, and four at the
                # analysis six hours before it, which is 18 UTC the day before.
                got = counts.get((d, 0), 0) + counts.get((prev, 1800), 0)
                want_here = 2 * spec.N_CON_PER_TIME
            elif which == "t0":
                got = counts.get((d, 0), 0)
                want_here = want
            else:
                got = counts.get((prev, 1800), 0)
                want_here = want
            rec["shared"][name] = {"fields": got, "expected": want_here}
            if got != want_here:
                rec["usable"] = False
                rec["reasons"].append(f"{name} has {got} of {want_here} fields for this date")

        evidence[date] = rec
        if rec["usable"]:
            selected.append(date)
        log(
            f"SUMMER {date}: {'usable' if rec['usable'] else 'NOT usable'}"
            + ("" if rec["usable"] else "; " + "; ".join(rec["reasons"]))
        )

    out_path = os.path.join(a.root, "validation", "selected_dates.json")
    atomic_write_json(
        out_path,
        {
            "criterion": (
                "a date is selected only if all nine initial-condition groups are "
                "present with exactly the expected number of fields for that date: "
                "the five groups fetched by the per-date probe (surface at t-6, "
                "pressure levels at t-6 and t0, specific humidity at t-6 and t0) and "
                "the four groups surviving from the 2026-09-09 study (surface at t0, "
                "waves at t-6 and t0, and the invariant analysis fields). A date "
                "missing any group is excluded rather than completed from another "
                "source."
            ),
            "candidates": cal.SUMMER_CANDIDATE_DATES,
            "dates": selected,
            "evidence": evidence,
            "probe_root": probe_root,
            "shared_dir": SHARED_DIR,
        },
    )
    log(f"SUMMER_SELECTION {len(selected)} of {len(cal.SUMMER_CANDIDATE_DATES)} dates usable -> {out_path}")
    log(f"selected: {selected}")
    if len(selected) < 6:
        log(
            "WARNING fewer than six dates are usable, so the summer run is a smoke "
            "test of the code, not a fidelity sample of the model"
        )
    return 0 if selected else 1


if __name__ == "__main__":
    sys.exit(main())
