"""Part 5: cost per row from the prediction jobs' ``costs.jsonl``.

``<pred root>/<arm>/step_NNNNNN/costs.jsonl``, one JSON line per member written by
predict_members.py: ``{"spec", "sample", "member", "seed", "calls", "seconds"}``, plus a peak
memory field when the job records one (any of peak_mem_gb, peak_memory_gb, peak_mem_gib,
max_memory_allocated_gb, peak_mem_mb, peak_mem_bytes, max_memory_allocated). Lines are kept
when their spec is the row's sampler (or its directory name).
"""
from __future__ import annotations

import json
import math
import statistics

from .common import Context, SectionResult, fmt, fnum, guarded, na, sanitize_spec, table

MEM_KEYS = {"peak_mem_gb": 1.0, "peak_memory_gb": 1.0, "peak_mem_gib": 1.0,
            "max_memory_allocated_gb": 1.0, "peak_gpu_mem_gb": 1.0, "peak_mem_mb": 1e-3,
            "peak_memory_mb": 1e-3, "peak_mem_bytes": 1e-9, "max_memory_allocated": 1e-9,
            "peak_memory_bytes": 1e-9}


def read_costs(path, spec: str) -> list[dict]:
    want = {spec, sanitize_spec(spec)}
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = str(rec.get("spec", ""))
            if s in want or sanitize_spec(s) in want:
                out.append(rec)
    return out


def summarise(recs: list[dict]) -> dict:
    calls = [fnum(r.get("calls")) for r in recs if math.isfinite(fnum(r.get("calls")))]
    secs = [fnum(r.get("seconds")) for r in recs if math.isfinite(fnum(r.get("seconds")))]
    mem = []
    for r in recs:
        for k, scale in MEM_KEYS.items():
            if math.isfinite(fnum(r.get(k))):
                mem.append(fnum(r[k]) * scale)
                break
    return {"n": len(recs),
            "calls": statistics.mean(calls) if calls else math.nan,
            "calls_set": sorted({int(c) for c in calls}),
            "sec_mean": statistics.mean(secs) if secs else math.nan,
            "sec_median": statistics.median(secs) if secs else math.nan,
            "mem_peak_gb": max(mem) if mem else math.nan}


def section(ctx: Context) -> SectionResult:
    res = SectionResult("5. Costs", "")
    md = ["# 5. Costs\n", "Network calls and wall seconds per member, peak memory, from costs.jsonl "
          "(lines of the row's sampler only).\n"]

    def body() -> str:
        header = ["set", "row", "calls (rows file)", "calls (logged, mean)", "seconds/member mean",
                  "seconds/member median", "peak memory GB", "members logged", "file"]
        rows = []
        for row in ctx.rows:
            sets = ctx.set_dirs(row.label) or row.sets or list(ctx.pred_roots) or [""]
            seen = set()
            any_row = False
            for s in sets:
                p = ctx.costs_file(row, s)
                if p is None or p in seen:
                    continue
                seen.add(p)
                recs = read_costs(p, row.sampler)
                if not recs:
                    rows.append([s, row.label, row.calls] + [na(f"no line for spec {row.sampler!r}")] * 5 + [str(p)])
                    continue
                c = summarise(recs)
                mem = fmt(c["mem_peak_gb"], 2) if math.isfinite(c["mem_peak_gb"]) else na("not recorded in costs.jsonl")
                rows.append([s, row.label, row.calls, fmt(c["calls"], 1), fmt(c["sec_mean"], 2),
                             fmt(c["sec_median"], 2), mem, c["n"], str(p)])
                any_row = True
                warn = "" if row.calls is None or c["calls_set"] in ([row.calls], []) else \
                    f" (logged calls {c['calls_set']} != {row.calls})"
                res.add(row.label, f"{s}: {fmt(c['calls'], 0)} calls, {fmt(c['sec_mean'], 2)} s/member, "
                                   f"peak {fmt(c['mem_peak_gb'], 1) if math.isfinite(c['mem_peak_gb']) else 'n/a'} GB{warn}")
            if not any_row and not any(r[1] == row.label for r in rows):
                rows.append(["", row.label, row.calls] + [na("costs.jsonl not found (no prediction root or no "
                                                             "step directory for this row)")] * 6)
                res.add(row.label, f"calls {row.calls}; " + na("costs.jsonl not found"))
        return table(ctx, "costs", header, rows)

    guarded(md, "", body)
    res.md = "\n".join(md)
    return res
