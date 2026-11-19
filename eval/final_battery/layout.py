"""Where a final-model battery run puts its outputs (one root per model row).

    <out>/battery_meta.json        what was run on what (written by the runner)
    <out>/summary.md, summary.json the one-page read (eval.final_battery.summary)
    <out>/logs/<step>.log          one log per step; <out>/.done/<step> marks a finished step
    <out>/terrain/                 FB_TERRAIN_OUT of the terrain kit: statics.npz, parts/, tables/, z/, z6/, d2/, figs/
    <out>/terrain/repeat_check.txt B11 printout
    <out>/consistency/edge1/       realism_consistency.py --outdir (RC_EDGE=1): ratios.csv, scalars.csv, stationary.csv,
                                   arrays.npz, figs/, maps/
    <out>/consistency/edge5/       the same with RC_EDGE=5 (rain large-scale spread without edge contamination)
    <out>/wind/fields/orog.npz     orography and land-sea mask on the output grid
    <out>/wind/fields<LLL>/        reduced 10u 10v msl per date (extract.py), deletable after the run (about 7 GB per lead)
    <out>/wind/out<LLL>/           accumulate.py outputs, spread.csv, spread_dates.csv; small/ = reduce.py tables and maps
    <out>/wind/xset/               xset_tc.py members table (only when an XSET-1 row root is given)
"""
from __future__ import annotations

import re
from pathlib import Path


def terrain_paths(out: Path) -> dict[str, Path]:
    t = Path(out) / "terrain"
    return {"root": t, "tables": t / "tables", "z": t / "z" / "tables", "z6": t / "z6" / "tables", "d2": t / "d2" / "tables",
            "figs": t / "figs"}


def consistency_paths(out: Path) -> dict[str, Path]:
    c = Path(out) / "consistency"
    return {"root": c, "edge1": c / "edge1", "edge5": c / "edge5"}


def wind_paths(out: Path) -> dict:
    w = Path(out) / "wind"
    leads = {}
    if w.exists():
        for d in sorted(w.glob("out*")):
            m = re.fullmatch(r"out(\d+)", d.name)
            if m and d.is_dir():
                leads[int(m.group(1))] = d
    return {"root": w, "fields": w / "fields", "leads": leads, "xset": w / "xset" / "xset_tc_members.csv"}
