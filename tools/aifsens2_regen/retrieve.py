"""Stage 1: retrieve the initial-condition fields from the MARS archive.

The work is cut into requests.  One request covers one field group (surface,
pressure levels, specific humidity, waves or constants) at one time of day, for
every date of the block at once.  Grouping this way matters for the wave
fields: for January to May 2026 most of the wave data sits on tape, and a
request that names every date of a month at one time of day causes each tape to
be mounted once instead of once per date.

The atmospheric groups and the wave group are submitted as two separate SLURM
arrays.  That is the second half of the same concern: the atmospheric fields
are mostly online and return in minutes, and they must not sit in a queue
behind a tape mount that can take hours.

Each request is idempotent.  If the target file already exists with exactly the
expected number of fields, the request returns immediately.  Otherwise the
existing file is moved aside, never deleted, and the request is attempted up to
three times with a ten-minute pause between attempts.  If MARS reports that the
tape holding the data is unavailable, the message TAPE_UNAVAILABLE appears in
the log so that the operator can tell an archive problem from a code problem.

Usage
-----
    python -m aifsens2_regen.retrieve --block 202601 --groupset atm --list
    python -m aifsens2_regen.retrieve --block 202601 --groupset atm --index 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from . import calendar as cal
from . import gribspec as spec
from .common import DEFAULT_ROOT, grib_count, log, move_aside, run_cmd

TAPE_MESSAGE = "tape on which the data reside is unavailable"
MAX_ATTEMPTS = 3
PAUSE_SECONDS = 600


def raw_dir(root: str, block: str) -> str:
    """Where the grouped raw files of a block live.

    The summer validation block keeps its raw files under validation/, because
    that data was probed date by date and is kept separate from the production
    calendar.
    """
    if block == cal.SUMMER_BLOCK:
        return os.path.join(root, "validation", "ic_raw")
    return os.path.join(root, "ic", "raw", block)


def build_requests(block: str, groupset: str, root: str) -> list[dict]:
    """Every MARS request needed for one block and one group set.

    Returns a list of dictionaries, one per request, each carrying the group,
    the time of day, the dates, the expected field count and the target path.
    """
    starts = cal.block_starts(block, root)
    if not starts:
        raise SystemExit(
            f"block {block} has no starts; for the summer block this means "
            f"selected_dates.json is missing or empty"
        )
    by_time = cal.input_times_by_time_of_day(starts)

    groups = spec.ATMOSPHERIC_GROUPS if groupset == "atm" else spec.WAVE_GROUPS
    d = raw_dir(root, block)

    requests = []
    for group in groups:
        for time_of_day, dates in by_time.items():
            g = spec.GROUPS[group]
            requests.append(
                dict(
                    group=group,
                    time=time_of_day,
                    dates=dates,
                    expected=g["per_date_per_time"] * len(dates),
                    target=os.path.join(d, f"{group}_{time_of_day}.grib"),
                )
            )
    return requests


def mars_text(req: dict, target: str) -> str:
    """The MARS request language text for one request."""
    g = spec.GROUPS[req["group"]]
    parts = [
        "retrieve",
        "class=od",
        "domain=g",
        "expver=0001",
        "step=0",
        f"stream={g['stream']}",
        f"type={g['type']}",
        f"levtype={g['levtype']}",
        f"param={'/'.join(str(p) for p in g['param'])}",
        f"date={'/'.join(req['dates'])}",
        f"time={req['time']}",
        f"grid={spec.GRID}",
        f"area={spec.AREA}",
    ]
    if g["levelist"]:
        parts.insert(-2, f"levelist={'/'.join(str(l) for l in g['levelist'])}")
    if g["perturbed"]:
        parts.insert(6, f"number={spec.MEMBERS[0]}/to/{spec.MEMBERS[-1]}")
    parts.append(f'target="{target}"')
    return ",\n  ".join(parts) + "\n"


def execute_request(req: dict, root: str) -> int:
    """Fetch one request.  Returns 0 on success, 1 on failure."""
    target = req["target"]
    os.makedirs(os.path.dirname(target), exist_ok=True)
    label = f"{req['group']}_{req['time']}"

    have = grib_count(target)
    if have == req["expected"]:
        log(f"RETRIEVE {label}: already complete with {have} fields, skipping")
        return 0
    if have:
        move_aside(target, f"incomplete: {have} of {req['expected']} fields")
    elif os.path.exists(target):
        move_aside(target, "empty or unreadable")

    work = os.path.dirname(target)
    req_path = os.path.join(work, f"req_{label}.mars")
    log_path = os.path.join(work, f"mars_{label}.log")
    tmp = f"{target}.tmp"
    if os.path.exists(tmp):
        move_aside(tmp, "leftover temporary file from an earlier attempt")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        with open(req_path, "w") as f:
            f.write(mars_text(req, tmp))
        t0 = time.time()
        rc, _ = run_cmd(["bash", "-c", f"mars {req_path} > {log_path} 2>&1"])
        elapsed = time.time() - t0
        n = grib_count(tmp)
        log(
            f"RETRIEVE {label} attempt={attempt} rc={rc} fields={n} "
            f"expected={req['expected']} dates={len(req['dates'])} "
            f"seconds={elapsed:.0f}"
        )
        try:
            with open(log_path) as f:
                text = f.read()
        except OSError:
            text = ""
        if TAPE_MESSAGE in text:
            log(f"TAPE_UNAVAILABLE {label}: MARS reports the tape is unavailable")

        if rc == 0 and n == req["expected"]:
            os.replace(tmp, target)
            log(f"RETRIEVE_OK {label}: {n} fields written to {target}")
            return 0

        if n:
            move_aside(tmp, f"attempt {attempt} returned {n} of {req['expected']} fields")
        if attempt < MAX_ATTEMPTS:
            log(f"RETRIEVE {label}: pausing {PAUSE_SECONDS} s before attempt {attempt + 1}")
            time.sleep(PAUSE_SECONDS)

    log(f"RETRIEVE_FAILED {label}: gave up after {MAX_ATTEMPTS} attempts")
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block", required=True)
    p.add_argument("--groupset", choices=["atm", "wave"], required=True)
    p.add_argument("--root", default=os.environ.get("AIFSENS2_ROOT", DEFAULT_ROOT))
    p.add_argument("--index", type=int, help="1-based index of the request to run")
    p.add_argument("--list", action="store_true", help="print the request list and exit")
    p.add_argument("--check", action="store_true", help="report completeness and exit")
    a = p.parse_args(argv)

    requests = build_requests(a.block, a.groupset, a.root)

    if a.list:
        for i, r in enumerate(requests, 1):
            print(
                f"{i}\t{r['group']}\t{r['time']}\t{len(r['dates'])} dates\t"
                f"{r['expected']} fields\t{r['target']}"
            )
        print(f"# {len(requests)} requests for block {a.block} groupset {a.groupset}")
        return 0

    if a.check:
        missing = 0
        for r in requests:
            n = grib_count(r["target"])
            ok = n == r["expected"]
            missing += 0 if ok else 1
            print(
                f"{'OK  ' if ok else 'MISS'} {r['group']}_{r['time']}: "
                f"{n} of {r['expected']} fields"
            )
        print(
            f"{'PASS' if missing == 0 else 'FAIL'}: {len(requests) - missing} of "
            f"{len(requests)} requests complete for block {a.block} groupset {a.groupset}"
        )
        return 0 if missing == 0 else 1

    if a.index is None:
        p.error("give --index, --list or --check")
    if not 1 <= a.index <= len(requests):
        p.error(f"--index must be between 1 and {len(requests)}")

    return execute_request(requests[a.index - 1], a.root)


if __name__ == "__main__":
    sys.exit(main())
