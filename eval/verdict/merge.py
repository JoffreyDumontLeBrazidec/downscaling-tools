"""Merge ``gates_P.csv`` (preflight) and ``gates_A/B/C.csv`` into the run verdict (note s4.0).

INVALID if a preflight check is not PASS (or preflight was not run). FAIL if any hard gate FAILS. PASS if all 14 hard
gates PASS. Otherwise INCONCLUSIVE, listing the gates that did not pass with their MDE. The verdict is never promoted
to "standard checkpoint" here (owner D9).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from eval.verdict import stats as S
from eval.verdict.gates import HARD_GATES, PREFLIGHT, GateLine, read_csv


def load(folder: Path) -> list[GateLine]:
    lines = []
    for tag in ("P", "A", "B", "C"):
        p = Path(folder) / f"gates_{tag}.csv"
        if p.exists():
            lines += read_csv(p)
    return lines


def decide(lines: list[GateLine]) -> dict:
    pre = {p: [ln for ln in lines if ln.gate == p] for p in PREFLIGHT}
    pre_state = {p: (S.combine([ln.state for ln in v]) if v else S.MISSING) for p, v in pre.items()}
    gate_state = {g: S.combine([ln.state for ln in lines if ln.gate == g and ln.hard]) for g in HARD_GATES}
    bad_pre = [p for p, s in pre_state.items() if s != S.PASS]
    if bad_pre:
        verdict = "INVALID"
    elif any(s == S.FAIL for s in gate_state.values()):
        verdict = "FAIL"
    elif all(s == S.PASS for s in gate_state.values()):
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "preflight": pre_state, "gates": gate_state,
            "failed": [g for g, s in gate_state.items() if s == S.FAIL],
            "not_passed": [g for g, s in gate_state.items() if s not in (S.PASS, S.FAIL)],
            "invalid_because": bad_pre}


def verdict_line(d: dict) -> str:
    v = d["verdict"]
    if v == "INVALID":
        return f"VERDICT: INVALID (preflight {', '.join(d['invalid_because'])})"
    if v == "FAIL":
        return f"VERDICT: FAIL (gates {', '.join(d['failed'])})"
    if v == "INCONCLUSIVE":
        return f"VERDICT: INCONCLUSIVE (gates not passed: {', '.join(d['not_passed'])})"
    return "VERDICT: PASS (14 of 14 hard gates)"


def _n(x, nd=3):
    return "" if not isinstance(x, float) or not math.isfinite(x) else f"{x:.{nd}g}"


def render(lines: list[GateLine], d: dict, row: str = "") -> str:
    out = [f"# Verdict{(' of ' + row) if row else ''}", "", verdict_line(d), "",
           "Thresholds are proposals until the calibration of note s7 (B3). Nothing is promoted without the owner's word.", "",
           "| gate | state | what |", "|---|---|---|"]
    for p, s in d["preflight"].items():
        out.append(f"| {p} | {s} | preflight |")
    for g, s in d["gates"].items():
        out.append(f"| {g} | {s} | {HARD_GATES[g]} |")
    out += ["", "## Hard gate lines", "", "| gate | sub | state | value | ref | diff | unit | SE | 95 % interval | margin | MDE "
            "| dates | blocks | note |", "|" + "---|" * 14]
    for ln in lines:
        if not ln.hard:
            continue
        iv = f"{_n(ln.lo)} .. {_n(ln.hi)}" if math.isfinite(ln.lo) else ""
        out.append(f"| {ln.gate} | {ln.sub} | {ln.state} | {_n(ln.value)} | {_n(ln.ref_value)} | {_n(ln.diff)} | {ln.unit} | "
                   f"{_n(ln.se)} | {iv} | {_n(ln.margin)} | {_n(ln.mde)} | {ln.n_dates} | {ln.n_blocks} | {ln.note} |")
    return "\n".join(out) + "\n"


def write(folder: Path, row: str = "") -> tuple[Path, str]:
    folder = Path(folder)
    lines = load(folder)
    d = decide(lines)
    (folder / "verdict.md").write_text(render(lines, d, row))
    (folder / "verdict.json").write_text(json.dumps({**d, "line": verdict_line(d), "row": row}, indent=1))
    return folder / "verdict.md", verdict_line(d)
