"""The campaign calendar: which forecast starts belong to which block.

A "start" is one forecast initialisation time, for example 2026-01-03 at 12
UTC.  Every start is run for ten perturbed members and for two lead times, six
and twelve hours.

The production calendar runs from 2026-01-01 00 UTC to 2026-05-12 00 UTC
inclusive, every twelve hours, which is 263 starts.  It is divided into five
monthly blocks so that a block can be retrieved, assembled, forecast and
verified as one unit and restarted on its own.  Two further blocks exist for
testing the code before production: the January pilot day and a set of summer
2026 validation dates.

Every block also knows which input times it needs.  A start at time T needs the
analysis at T minus six hours and the analysis at T, so a day with starts at 00
and 12 UTC needs the previous day at 18 UTC and the same day at 00, 06 and 12
UTC.
"""

from __future__ import annotations

import datetime as dt
import json
import os

# --------------------------------------------------------------------------
# The production calendar.
# --------------------------------------------------------------------------

CAMPAIGN_FIRST_START = dt.datetime(2026, 1, 1, 0)
CAMPAIGN_LAST_START = dt.datetime(2026, 5, 12, 0)
START_INTERVAL = dt.timedelta(hours=12)
INPUT_LEAD = dt.timedelta(hours=6)

# The summer validation dates, all at 00 UTC.  Which of them are actually
# usable depends on what the archive can serve; see selected_summer_dates().
SUMMER_CANDIDATE_DATES = [
    "20260605", "20260615", "20260625",
    "20260705", "20260715", "20260725",
    "20260804", "20260814", "20260824", "20260830",
]

MONTHLY_BLOCKS = ["202601", "202602", "202603", "202604", "202605"]
PILOT_BLOCK = "pilot_20260101"
SUMMER_BLOCK = "summer_validation"


def campaign_starts() -> list[dt.datetime]:
    """Every start of the production calendar, in order."""
    out = []
    t = CAMPAIGN_FIRST_START
    while t <= CAMPAIGN_LAST_START:
        out.append(t)
        t += START_INTERVAL
    return out


def block_starts(block: str, root: str | None = None) -> list[dt.datetime]:
    """The starts belonging to one block.

    Parameters
    ----------
    block
        A monthly block name such as "202603", or "pilot_20260101", or
        "summer_validation".
    root
        The data root.  Only the summer validation block uses it, to read the
        list of dates the archive was able to serve.
    """
    if block == PILOT_BLOCK:
        return [dt.datetime(2026, 1, 1, 0), dt.datetime(2026, 1, 1, 12)]

    if block == SUMMER_BLOCK:
        dates = selected_summer_dates(root)
        return [dt.datetime.strptime(d, "%Y%m%d") for d in dates]

    if block in MONTHLY_BLOCKS:
        year, month = int(block[:4]), int(block[4:])
        return [s for s in campaign_starts() if s.year == year and s.month == month]

    raise ValueError(
        f"unknown block {block!r}; expected one of "
        f"{MONTHLY_BLOCKS + [PILOT_BLOCK, SUMMER_BLOCK]}"
    )


def selected_summer_dates(root: str | None = None) -> list[str]:
    """The summer validation dates whose initial conditions arrived complete.

    The list is written by the availability check into
    <root>/validation/selected_dates.json.  If that file does not exist yet the
    function returns an empty list rather than guessing, because running a
    forecast from an incomplete initial condition would silently produce a
    wrong member.
    """
    if root is None:
        root = os.environ.get("AIFSENS2_ROOT", "")
    path = os.path.join(root, "validation", "selected_dates.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(json.load(f)["dates"])


def input_times(starts: list[dt.datetime]) -> list[dt.datetime]:
    """Every analysis time the given starts need, sorted and de-duplicated.

    Each start needs t-6 and t0.
    """
    needed = set()
    for s in starts:
        needed.add(s - INPUT_LEAD)
        needed.add(s)
    return sorted(needed)


def input_times_by_time_of_day(starts: list[dt.datetime]) -> dict[str, list[str]]:
    """Group the needed analysis times by time of day.

    Returns a mapping from a four-digit time of day such as "1800" to the sorted
    list of dates, as YYYYMMDD strings, that are needed at that time of day.
    This is the shape a grouped MARS request takes: one request per group and
    time of day, covering every date of the block at once, so that a tape is
    mounted once instead of once per date.
    """
    out: dict[str, set[str]] = {}
    for t in input_times(starts):
        out.setdefault(f"{t.hour:02d}00", set()).add(t.strftime("%Y%m%d"))
    return {k: sorted(v) for k, v in sorted(out.items())}


def start_key(start: dt.datetime) -> str:
    """The directory name for a start, for example "20260103_12"."""
    return start.strftime("%Y%m%d_%H")


def parse_start_key(key: str) -> dt.datetime:
    """Inverse of start_key."""
    return dt.datetime.strptime(key, "%Y%m%d_%H")


def all_blocks() -> list[str]:
    return [PILOT_BLOCK, SUMMER_BLOCK] + MONTHLY_BLOCKS


def describe() -> str:
    """A human-readable summary of the calendar, used by the command line."""
    lines = []
    total = 0
    for b in MONTHLY_BLOCKS:
        n = len(block_starts(b))
        total += n
        s = block_starts(b)
        lines.append(f"  {b}: {n:3d} starts, {s[0]} to {s[-1]}")
    lines.append(f"  total production starts: {total}")
    lines.append(f"  {PILOT_BLOCK}: {len(block_starts(PILOT_BLOCK))} starts")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    assert len(campaign_starts()) == 263, len(campaign_starts())
    expected = {"202601": 62, "202602": 56, "202603": 62, "202604": 60, "202605": 23}
    for b, n in expected.items():
        got = len(block_starts(b))
        assert got == n, f"{b}: expected {n} starts, got {got}"
    print("calendar self-check passed: 263 starts, block sizes as specified")
