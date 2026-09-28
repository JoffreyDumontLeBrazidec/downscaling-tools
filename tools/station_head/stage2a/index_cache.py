"""Index the surviving MARS retriever cache of the AIFS ENS v2 dataset build.

Each part_* directory of /home/ecm5702/scratch/eval/aifsv2_full_cache/_done_aside/
carries a cache-2.db sqlite database whose `cache` table maps a cache file path to
the JSON MARS request that produced it. This script reads every database, parses
the request, and writes one CSV row per cache file with the columns the stage 2a
bundle build needs: date, time, levtype, the number of members and steps in the
file, and the absolute path of the file as it exists today (the database records
the path before the directory was moved aside, so only the basename is reliable).
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path("/home/ecm5702/scratch/eval/aifsv2_full_cache/_done_aside")


def main(out_csv: str) -> None:
    rows = []
    for part in sorted(ROOT.iterdir()):
        db = part / "cache-2.db"
        if not db.exists():
            continue
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for path, size, args in con.execute("select path, size, args from cache"):
            try:
                req = json.loads(args)
            except Exception:
                continue
            name = Path(path).name
            here = part / name
            rows.append(
                {
                    "date": (req.get("date") or [""])[0].replace("-", ""),
                    "time": str(req.get("time", "")).zfill(2),
                    "levtype": req.get("levtype", ""),
                    "n_members": len(req.get("number", []) or []),
                    "steps": "/".join(str(s) for s in (req.get("step") or [])),
                    "params": "/".join(req.get("param") or []),
                    "size": size,
                    "part": part.name,
                    "path": str(here),
                    "exists": int(here.exists()),
                }
            )
        con.close()
    rows.sort(key=lambda r: (r["date"], r["time"], r["levtype"]))
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1])
