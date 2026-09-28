"""Lane BASELINE resolution + vs-baseline scoring.

The baseline of a lane is the TOP of that lane's scoreboard, stamped in
`meta.baseline` of `scoreboard_<lane>/scoreboard.json` (dev/docs hub) by
`docs/scoreboard/baseline.py promote`. Every "how is a run doing?" answer is
relative to it. This module is the single resolver used by:

  * `eval.cli evaluate/scoreboard/run --vs-baseline` -> vs_baseline.md delta table
  * `eval.cli evolution --ref baseline:<lane>`       -> baseline ladder card curve
  * `eval.jobs.ladder loss --baseline-lane <lane>`   -> archived MLflow curve overlay
    (coarse sanity ONLY: train/val loss is not cross-run-comparable skill — judge
    progress on ladder/proxy metrics, not on the loss overlay)

The hub location can be overridden with DS_SCOREBOARD_HUB (default:
/home/ecm5702/dev/docs/docs).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

SCOREBOARD_HUB = Path(os.environ.get("DS_SCOREBOARD_HUB", "/home/ecm5702/dev/docs/docs"))
SCOREBOARD_LANES = ("o1280_o2560", "o320_o1280", "o96_o320", "o48_o96")
PROMOTE_HINT = ("no baseline registered for lane {lane!r} — promote the scoreboard's rank-1 run: "
                "python3 {hub}/scoreboard/baseline.py promote --lane {lane} --slug <rank-1 slug>")


def scoreboard_lane(lane_name: str) -> str:
    """Map an eval lane-config name (e.g. 'o320_o1280_piecewise21_s1k') to its scoreboard lane."""
    for cand in SCOREBOARD_LANES:  # ordered so the longest/most-specific prefix wins
        if lane_name == cand or lane_name.startswith(cand + "_") or lane_name.startswith("_" + cand):
            return cand
        if cand in lane_name:
            return cand
    raise SystemExit(f"cannot infer scoreboard lane from {lane_name!r} "
                     f"(known: {', '.join(SCOREBOARD_LANES)})")


def resolve_baseline(lane_name: str) -> dict:
    """-> {'lane', 'baseline': meta.baseline, 'record': full scoreboard record}."""
    lane = scoreboard_lane(lane_name)
    sb_path = SCOREBOARD_HUB / f"scoreboard_{lane}" / "scoreboard.json"
    if not sb_path.is_file():
        raise SystemExit(f"scoreboard not found: {sb_path} (set DS_SCOREBOARD_HUB?)")
    doc = json.loads(sb_path.read_text())
    baseline = (doc.get("meta") or {}).get("baseline")
    if not baseline:
        raise SystemExit(PROMOTE_HINT.format(lane=lane, hub=SCOREBOARD_HUB))
    record = next((r for r in doc.get("records", []) if r.get("slug") == baseline.get("slug")), None)
    if record is None:
        raise SystemExit(f"baseline slug {baseline.get('slug')!r} not found in {sb_path} — "
                         f"re-promote or fix meta.baseline")
    return {"lane": lane, "baseline": baseline, "record": record}


def baseline_ladder_card(lane_name: str) -> tuple[str, str]:
    """-> (label, path) of the baseline's archived ladder.json, for evolution --ref."""
    res = resolve_baseline(lane_name)
    card = res["baseline"].get("ladder_card")
    if not card or not os.path.isfile(card):
        raise SystemExit(f"baseline {res['baseline'].get('slug')!r} has no archived ladder card "
                         f"({card!r}) — archive one via baseline.py promote --ladder-card")
    ck = (res["record"].get("checkpoint") or {}).get("id") or res["baseline"].get("slug")
    return f"baseline-{str(ck)[:8]}", card


def baseline_mlflow(lane_name: str) -> tuple[str, str]:
    """-> (experiment_dir, run_id) of the baseline's archived MLflow file store."""
    res = resolve_baseline(lane_name)
    arch = res["baseline"].get("mlflow_archive")
    if not arch or not os.path.isdir(arch):
        raise SystemExit(f"baseline {res['baseline'].get('slug')!r} has no archived MLflow store "
                         f"({arch!r}) — see baseline.py promote")
    return os.path.dirname(arch), os.path.basename(arch)


# ---------------------------------------------------------------- vs-baseline delta table
_SURFACE_VARS = ("10v", "2t", "msl", "sp")
_TC_METRICS = ("wind_max", "mslp_min")


def _parse_scores_csv(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                out[(row.get("evaluator"), row.get("metric"))] = float(row.get("value"))
            except (TypeError, ValueError):
                continue
    return out


def _fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def _delta_row(name, run_v, base_v, better, nd=4):
    """One table row. `better`: 'lower' | 'higher' | None (raw extremes have no direction)."""
    if run_v is None or base_v is None:
        return f"| {name} | {_fmt(run_v, nd)} | {_fmt(base_v, nd)} | — | |"
    d = run_v - base_v
    verdict = ""
    if better == "lower":
        verdict = "✅ better" if d < 0 else ("➖ equal" if d == 0 else "❌ worse")
    elif better == "higher":
        verdict = "✅ better" if d > 0 else ("➖ equal" if d == 0 else "❌ worse")
    return f"| {name} | {_fmt(run_v, nd)} | {_fmt(base_v, nd)} | {d:+.{nd}f} | {verdict} |"


def write_vs_baseline(run_root: Path, lane_name: str, run_label: str = "") -> Path:
    """Diff <run_root>/scoreboard/scores.csv against the lane baseline's scoreboard record;
    write <run_root>/scoreboard/vs_baseline.md and print the table."""
    run_root = Path(run_root)
    scores_csv = run_root / "scoreboard" / "scores.csv"
    if not scores_csv.is_file():
        raise SystemExit(f"--vs-baseline: no scores at {scores_csv} — run the scoreboard step first "
                         f"(eval.cli scoreboard --eval-dir {run_root})")
    res = resolve_baseline(lane_name)
    bl, rec = res["baseline"], res["record"]
    sc = _parse_scores_csv(scores_csv)
    surf = lambda m: sc.get(("surface", m))
    spectra = next((v for (e, m), v in sc.items()
                    if (e or "").startswith("spectra") and "mean" in m and "score" in m), None)
    if spectra is None:
        spectra = next((v for (e, m), v in sc.items()
                        if (e or "").startswith("spectra") and "mean" in m), None)

    L = [f"# vs BASELINE — {run_label or run_root.name}",
         "",
         f"Baseline = top of the `{res['lane']}` scoreboard: **{rec.get('display_label')}** "
         f"(slug `{bl.get('slug')}`, promoted `{bl.get('promoted_utc', '?')}`).",
         "All deltas are run − baseline.",
         "",
         "| metric | run | baseline | Δ | verdict |",
         "|---|--:|--:|--:|---|",
         _delta_row("surface_weighted_nmse", surf("surface_weighted_nmse"),
                    rec.get("surface_weighted_nmse"), "lower")]
    base_nmse = rec.get("surface_nmse") or {}
    for v in _SURFACE_VARS:
        L.append(_delta_row(f"surface_{v}_nmse", surf(f"surface_{v}_nmse"), base_nmse.get(v), "lower"))
    L.append(_delta_row("spectra_mean", spectra, rec.get("spectra_mean"), "higher"))

    # RAW TC extremes per event the baseline has been scored on (no direction verdict — read
    # model vs ENFO/OPER by eye per the run-trust contract; Δ shown for orientation only).
    tc_events = rec.get("tc_events") or {}
    if not tc_events and (rec.get("tc") or {}).get("event"):
        tc_events = {rec["tc"]["event"]: rec["tc"]}
    for ev, block in tc_events.items():
        model = ((block or {}).get("extremes") or {}).get("model") or {}
        for m in _TC_METRICS:
            nd = 2 if m.startswith("wind") else 1
            L.append(_delta_row(f"tc_{ev}_{m} (raw)", sc.get(("tc", f"tc_{ev}_{m}")),
                                model.get(m), None, nd))
    L += ["",
          f"> TC rows are RAW extremes (no better/worse verdict): judge model vs ENFO/OPER by eye "
          f"in the lane scoreboard dossiers.",
          f"> Baseline archives: mlflow `{bl.get('mlflow_archive')}` · eval `{bl.get('eval_archive')}` "
          f"· ladder `{bl.get('ladder_card')}`",
          ""]
    out = run_root / "scoreboard" / "vs_baseline.md"
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"vs-baseline table -> {out}")
    return out
