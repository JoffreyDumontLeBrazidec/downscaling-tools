#!/usr/bin/env python3
"""Move existing spectra_ecmwf reference caches under their own window key.

Before window addressing, one directory per lane held whichever evaluation
window happened to be computed first, and every later run silently reused it.
This script reads each cache's own recorded metadata, works out which window it
actually holds, and moves it into a directory named for that window.

Nothing is deleted and nothing is rewritten: directories are moved within the
same parent, on the same filesystem.  Running it twice is harmless.

A cache whose summary records no truncation cannot be described honestly, so it
is filed under a key marked ``Tunknown``.  The runner treats such a cache as
unverifiable and recomputes rather than presenting it as truth, which is the
correct outcome: the o1280_o2560 reference in particular holds curves of 319
wavenumbers taken from a T5119 transform, covering only the largest scales.

Dry run by default.  Pass --apply to move anything.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from eval.evaluators.spectra_ecmwf.runner import _window_key

PAYLOAD = ("grb", "spectral_harmonics", "spectra")
SUMMARIES = ("staging_summary.json", "spectra_summary.json")
KEY_RE = re.compile(r"^d\S*_s\S*_m\S*_T\S+_[0-9a-f]{8}$")

TOMBSTONE = """This reference cache was migrated under a window key on {when}.

Recorded window : {window}
Recorded truncation : {truncation}
Curve length actually stored : {length}

{verdict}
"""


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _describe(ref_dir: Path) -> dict:
    """Work out which window a legacy cache directory actually holds."""
    staging = _read_json(ref_dir / "staging_summary.json")
    spectra = _read_json(ref_dir / "spectra_summary.json")

    dates: set[str] = set()
    steps: set[int] = set()
    members: set[int] = set()
    for entry in spectra.get("files") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("date") is not None:
            dates.add(str(entry["date"]))
        if entry.get("step_hours") is not None:
            steps.add(int(entry["step_hours"]))
        if entry.get("member") is not None:
            members.add(int(entry["member"]))

    if not dates:
        dates = {str(d) for d in staging.get("dates") or []}
    if not members:
        members = {int(m) for m in staging.get("ensemble_members") or []}

    truncation = spectra.get("truncation")
    length = ""
    for state_dir in sorted((ref_dir / "spectra").glob("*")):
        for npy in sorted(state_dir.glob("ampl_*.npy")):
            try:
                import numpy as np

                length = str(np.load(npy).size)
            except Exception:  # noqa: BLE001 - a probe, not a contract
                length = "unreadable"
            break
        if length:
            break

    return {
        "dates": sorted(dates),
        "steps": sorted(steps) or [120],
        "members": sorted(members),
        "truncation": truncation,
        "length": length or "(none)",
    }


def _key_for(desc: dict) -> str:
    if desc["truncation"] is None:
        # Keep the readable prefix but mark the truncation honestly, so the
        # runner refuses to treat this cache as truth.
        readable = _window_key(
            dates=desc["dates"], steps=desc["steps"], members=desc["members"], truncation=1
        )
        return readable.replace("_T1_", "_Tunknown_")
    return _window_key(
        dates=desc["dates"],
        steps=desc["steps"],
        members=desc["members"],
        truncation=int(desc["truncation"]),
    )


def migrate(ref_dir: Path, *, apply: bool, when: str) -> str:
    if not ref_dir.is_dir():
        return "absent"
    if not any((ref_dir / name).is_dir() for name in PAYLOAD):
        already = [d.name for d in ref_dir.iterdir() if d.is_dir() and KEY_RE.match(d.name)]
        return f"already migrated ({', '.join(already)})" if already else "nothing to move"

    desc = _describe(ref_dir)
    if not desc["dates"]:
        return "SKIPPED: no recorded dates, cannot name a window honestly"

    key = _key_for(desc)
    target = ref_dir / key
    print(f"    window   : {','.join(desc['dates'])} steps={desc['steps']} "
          f"members={desc['members'] or 'unrecorded'}")
    print(f"    truncation: {desc['truncation']}   stored curve length: {desc['length']}")
    print(f"    -> {target.name}/")
    if not apply:
        return "dry run"

    target.mkdir(exist_ok=True)
    for name in PAYLOAD + SUMMARIES:
        source = ref_dir / name
        if source.exists():
            source.rename(target / name)

    verdict = (
        "This cache records no truncation, so what it represents cannot be established. "
        "The runner treats it as unverifiable and recomputes rather than using it as truth."
        if desc["truncation"] is None
        else "This cache is valid for the window named above and no other."
    )
    (target / "README.migrated.txt").write_text(
        TOMBSTONE.format(
            when=when,
            window=",".join(desc["dates"]),
            truncation=desc["truncation"] if desc["truncation"] is not None else "not recorded",
            length=desc["length"],
            verdict=verdict,
        ),
        encoding="utf-8",
    )
    return "moved"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root", default="/perm/ecm5702/reference",
        help="directory holding the per-lane reference trees",
    )
    parser.add_argument("--lanes", default="o48_o96,o96_o320,o320_o1280,o1280_o2560")
    parser.add_argument("--apply", action="store_true", help="actually move directories")
    parser.add_argument("--when", default="2026-08-22")
    args = parser.parse_args()

    root = Path(args.reference_root)
    if not args.apply:
        print("DRY RUN - nothing will be moved. Pass --apply to migrate.\n")

    for lane in args.lanes.split(","):
        for label in ("truth", "input"):
            ref_dir = root / lane / "spectra_ecmwf" / label
            print(f"{lane}/{label}:")
            print(f"    {migrate(ref_dir, apply=args.apply, when=args.when)}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
