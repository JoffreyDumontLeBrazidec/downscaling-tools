#!/usr/bin/env python3
"""Put the legacy reference cache back where the original evaluator looks.

The window-addressing migration moved it into _superseded_20260825/. Now that
spectra_ecmwf is restored to its original code, it expects its reference at
reference_dir/<truth|input>/{grb,spectral_harmonics,spectra}. This undoes that
one move so the backup evaluator works without an hour of recomputation.

Nothing is deleted, and the window-key directories that spectra_ecmwf_v2 uses
are left exactly where they are. Dry run unless --apply is given.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PAYLOAD = ("grb", "spectral_harmonics", "spectra", "spectra_summary.json", "staging_summary.json")
PARKED = "_superseded_20260825"


def restore(base: Path, *, apply: bool) -> str:
    parked = base / PARKED
    if not parked.is_dir():
        return "nothing parked here"
    present = [n for n in PAYLOAD if (base / n).exists()]
    if present:
        return f"top level already holds {', '.join(present)}; left alone"
    movable = [n for n in PAYLOAD if (parked / n).exists()]
    if not movable:
        return "parked directory holds none of the expected payload"
    print(f"    would move up: {', '.join(movable)}")
    if not apply:
        return "dry run"
    for name in movable:
        (parked / name).rename(base / name)
    return f"restored {len(movable)} entries"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-root", default="/perm/ecm5702/reference")
    ap.add_argument("--lanes", default="o48_o96,o96_o320,o320_o1280,o1280_o2560")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not args.apply:
        print("DRY RUN - nothing moved. Pass --apply.\n")
    root = Path(args.reference_root)
    for lane in args.lanes.split(","):
        for kind in ("truth", "input"):
            base = root / lane / "spectra_ecmwf" / kind
            print(f"{lane}/{kind}:")
            print(f"    {restore(base, apply=args.apply)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
