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

There is a second, chunked mode for the case where the archive will not
schedule a grouped monthly request at all.  A request that names a whole month
of one group is sometimes never even assigned a server task, while smaller
requests for the same fields are served promptly.  The chunked mode cuts the
dates of one (block, group, time of day) into contiguous ranges of a few days
and fetches one range per SLURM array task.  The requests are built by the same
code as the monthly ones and from the same group definitions in gribspec, so
the fields cannot drift apart; only the list of dates differs.  Each chunk is
written into its own subdirectory of the block's raw directory, which stage two
already picks up, and never into the grouped monthly file's path.

Usage
-----
    python -m aifsens2_regen.retrieve --block 202601 --groupset atm --list
    python -m aifsens2_regen.retrieve --block 202601 --groupset atm --index 3

    python -m aifsens2_regen.retrieve --block 202602 --group pl --time 0000 --list
    python -m aifsens2_regen.retrieve --block 202602 --group pl --time 0000 --index 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from . import calendar as cal
from . import gribspec as spec
from .common import DEFAULT_ROOT, grib_count, log, move_aside, run_cmd

TAPE_MESSAGE = "tape on which the data reside is unavailable"
MAX_ATTEMPTS = 3
PAUSE_SECONDS = 600
MARS_TIMEOUT_SECONDS = 36000

# The default width of a chunk, in days.  Eight days gives about four chunks per
# calendar month, which is small enough that the archive has been willing to
# schedule the requests and large enough that a month does not turn into thirty
# separate queue entries.
CHUNK_DAYS_DEFAULT = 8


def raw_dir(root: str, block: str) -> str:
    """Where the grouped raw files of a block live.

    The validation blocks keep their raw files under validation/, because that
    data was probed date by date and is kept separate from the production
    calendar.
    """
    if block in cal.VALIDATION_BLOCKS:
        return cal.validation_dir(root, "ic_raw", block)
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


def chunk_dir(root: str, block: str, group: str, time_of_day: str) -> str:
    """The subdirectory one (block, group, time of day) keeps its chunks in.

    It sits inside the block's raw directory, so stage two finds the files
    without any change: assemble.split_raw globs both "<block>/*.grib" and
    "<block>/*/*.grib" and ignores directories whose name starts with
    "_aside_".  The name ends in "_bychunk" so that an operator can see at a
    glance which groups were fetched the slow way.
    """
    return os.path.join(raw_dir(root, block), f"{group}_{time_of_day}_bychunk")


def build_chunk_requests(
    block: str,
    group: str,
    time_of_day: str,
    root: str,
    chunk_days: int = CHUNK_DAYS_DEFAULT,
) -> list[dict]:
    """The chunked requests for one block, one group and one time of day.

    The dates are exactly the dates the grouped monthly request would have
    named, taken from the same calendar function, cut into contiguous ranges of
    at most chunk_days days.  The ranges therefore tile the month once, with no
    gap and no overlap, whatever the length of the month.  Every other field of
    the request, including the expected count, is built the same way as in
    build_requests, so a chunk differs from the monthly request only in its
    dates.
    """
    if group not in spec.GROUPS:
        raise SystemExit(
            f"unknown group {group!r}; expected one of {sorted(spec.GROUPS)}"
        )
    if chunk_days < 1:
        raise SystemExit("--chunk-days must be at least 1")

    starts = cal.block_starts(block, root)
    if not starts:
        raise SystemExit(
            f"block {block} has no starts; for the summer block this means "
            f"selected_dates.json is missing or empty"
        )
    by_time = cal.input_times_by_time_of_day(starts)
    if time_of_day not in by_time:
        raise SystemExit(
            f"block {block} needs no input at time {time_of_day}; it needs "
            f"{sorted(by_time)}"
        )

    dates = by_time[time_of_day]
    g = spec.GROUPS[group]
    d = chunk_dir(root, block, group, time_of_day)

    requests = []
    for i in range(0, len(dates), chunk_days):
        part = dates[i : i + chunk_days]
        label = f"{group}_{time_of_day}_{part[0]}_{part[-1]}"
        requests.append(
            dict(
                group=group,
                time=time_of_day,
                dates=part,
                expected=g["per_date_per_time"] * len(part),
                label=label,
                target=os.path.join(d, f"{label}.grib"),
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
    # The chunked requests carry their own label, which includes the date
    # range, so that two chunks of the same group and time of day cannot
    # overwrite each other's request and log files.
    label = req.get("label") or f"{req['group']}_{req['time']}"

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
        # A tape-bound request can sit in the MARS server queue for hours before a
        # single field arrives, so the ceiling is ten hours (inside the twelve-hour
        # job limit) and a timeout counts as a failed attempt, not as a crash.
        try:
            rc, _ = run_cmd(["bash", "-c", f"mars {req_path} > {log_path} 2>&1"], timeout=MARS_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            rc = 124
            log(f"RETRIEVE {label} attempt={attempt}: mars exceeded {MARS_TIMEOUT_SECONDS} s and was stopped")
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
    p.add_argument(
        "--groupset",
        choices=["atm", "wave"],
        help="monthly mode: fetch every group of this set, one request per "
        "group and time of day, covering the whole block at once",
    )
    p.add_argument(
        "--group",
        choices=sorted(spec.GROUPS),
        help="chunked mode: fetch this one group, cut into date ranges of "
        "--chunk-days days; requires --time",
    )
    p.add_argument(
        "--time",
        help="chunked mode: the input time of day, as four digits, for "
        "example 0000 or 1800",
    )
    p.add_argument(
        "--chunk-days",
        type=int,
        default=CHUNK_DAYS_DEFAULT,
        help=f"chunked mode: days per chunk (default {CHUNK_DAYS_DEFAULT})",
    )
    p.add_argument("--root", default=os.environ.get("AIFSENS2_ROOT", DEFAULT_ROOT))
    p.add_argument("--index", type=int, help="1-based index of the request to run")
    p.add_argument("--list", action="store_true", help="print the request list and exit")
    p.add_argument("--check", action="store_true", help="report completeness and exit")
    a = p.parse_args(argv)

    # Exactly one of the two modes must be chosen.  They are kept apart here and
    # nowhere else: from this point on both modes are the same list of requests
    # run by the same code.
    if (a.groupset is None) == (a.group is None):
        p.error("give either --groupset (monthly mode) or --group (chunked mode)")
    if a.group is not None and not a.time:
        p.error("--group needs --time, for example --time 0000")
    if a.groupset is not None and a.time:
        p.error("--time belongs to the chunked mode; use it together with --group")

    if a.group is not None:
        requests = build_chunk_requests(
            a.block, a.group, a.time, a.root, a.chunk_days
        )
        what = f"block {a.block} group {a.group} time {a.time} in chunks of {a.chunk_days} days"
    else:
        requests = build_requests(a.block, a.groupset, a.root)
        what = f"block {a.block} groupset {a.groupset}"

    if a.list:
        for i, r in enumerate(requests, 1):
            print(
                f"{i}\t{r['group']}\t{r['time']}\t{r['dates'][0]}..{r['dates'][-1]}\t"
                f"{len(r['dates'])} dates\t{r['expected']} fields\t{r['target']}"
            )
        print(f"# {len(requests)} requests for {what}")
        return 0

    if a.check:
        missing = 0
        for r in requests:
            n = grib_count(r["target"])
            ok = n == r["expected"]
            missing += 0 if ok else 1
            print(
                f"{'OK  ' if ok else 'MISS'} "
                f"{r.get('label') or r['group'] + '_' + r['time']}: "
                f"{n} of {r['expected']} fields"
            )
        print(
            f"{'PASS' if missing == 0 else 'FAIL'}: {len(requests) - missing} of "
            f"{len(requests)} requests complete for {what}"
        )
        return 0 if missing == 0 else 1

    if a.index is None:
        p.error("give --index, --list or --check")
    if not 1 <= a.index <= len(requests):
        p.error(f"--index must be between 1 and {len(requests)}")

    return execute_request(requests[a.index - 1], a.root)


if __name__ == "__main__":
    sys.exit(main())
