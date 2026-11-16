#!/usr/bin/env python3
"""Reduced extract of a NOAA NCEI IBTrACS v04r01 CSV for the storm selection of the extremes set (16 Nov 2026).

Keeps the rows whose ISO_TIME lies in [--start, --end] and the columns the selection and the scoring need
(identity, track type, position, WMO and USA wind / pressure, RMW, status, distance to land). The units row
(second line of the NOAA file) is dropped; a missing value stays a single space as in the source. Standard
library only, so it runs under any python3 (Atos module python3, the Mac).

    python3 ibtracs_extract.py ibtracs.last3years.list.v04r01.csv ibtracs_2024_2025_extract.csv \
        --start 2023-12-20 --end 2026-01-10
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

COLS = ["SID", "SEASON", "NUMBER", "BASIN", "SUBBASIN", "NAME", "ISO_TIME", "NATURE", "LAT", "LON", "WMO_WIND",
        "WMO_PRES", "WMO_AGENCY", "TRACK_TYPE", "DIST2LAND", "LANDFALL", "IFLAG", "USA_AGENCY", "USA_ATCF_ID",
        "USA_LAT", "USA_LON", "USA_STATUS", "USA_WIND", "USA_PRES", "USA_SSHS", "USA_RMW", "USA_ROCI",
        "USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW", "STORM_SPEED", "STORM_DIR"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--start", default="2023-12-20")
    ap.add_argument("--end", default="2026-01-10")
    a = ap.parse_args()
    src = Path(a.src)
    n_in = n_out = 0
    sids, track_types = set(), {}
    with open(src, newline="") as f, open(a.dst, "w", newline="") as g:
        r = csv.reader(f)
        header = next(r)
        next(r)  # the units row
        keep = [c for c in COLS if c in header]
        idx = [header.index(c) for c in keep]
        it = header.index("ISO_TIME")
        w = csv.writer(g)
        w.writerow(keep)
        for row in r:
            n_in += 1
            day = row[it].strip()[:10]
            if not (a.start <= day <= a.end):
                continue
            w.writerow([row[i] for i in idx])
            n_out += 1
            sids.add(row[header.index("SID")])
            if "TRACK_TYPE" in header:
                t = row[header.index("TRACK_TYPE")].strip()
                track_types[t] = track_types.get(t, 0) + 1
    meta = {"source": str(src), "source_bytes": src.stat().st_size,
            "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "start": a.start, "end": a.end,
            "rows_in": n_in, "rows_out": n_out, "storms": len(sids), "track_types": track_types, "columns": keep,
            "missing_columns": [c for c in COLS if c not in header]}
    Path(a.dst + ".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("rows_in", "rows_out", "storms", "track_types", "missing_columns")}))


if __name__ == "__main__":
    main()
