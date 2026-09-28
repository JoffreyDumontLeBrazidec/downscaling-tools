#!/usr/bin/env python3
"""GRIB-route precipitation scorer — the prepml/FDB twin of precip_scores.

Scores model 6h-window tp read from per-date GRIBs (e.g. fields retrieved
from FDB after a prepml month — retrieve-only, never an eval-mode re-push)
against the same `_tp_dea` truth GRIBs and the same o1280 interp baseline as
the manual-inference route, with the identical metric definitions
(eval.evaluators.precip_scores.runner.aggregate_rows/summarize), so numbers
from both routes are directly comparable.

The model GRIB template must contain "{date}" and hold all members and steps
for that date (keys: perturbation number + endStep). Model fields are passed
through the same accumulation auto-detector as everything else, so a raw
running total is corrected loudly rather than scored silently.

Verification status (2026-08-26): the mechanics run end-to-end, but no clean
o2560 prepml month exists yet to score for real — the live Sept-2025 months
(ja6g/ja6y) are o320->o1280 and carry no tp. First real use must eyeball the
per-step log lines against the manual-route numbers.

Usage:
    python -m eval._backends.precip.score_gribs \
        --model-grib-tpl  "/path/model_o2560_tp_date{date}.grib" \
        --truth-grib-tpl  "/path/..._date{date}_..._y_tp_dea.grib" \
        --baseline-grib-tpl "/path/..._date{date}_..._input_tp_dea.grib" \
        --interp-index-cache /path/nn.npz \
        --dates 20250926,20250927 --out-dir /path/out --run-label "j9xx month"
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from eval._backends.precip import metrics as M
from eval._backends.precip.sources import (
    LresInterpBaseline,
    PrecipTruthSource,
    _read_grib_var,
    check_grid_match,
    maybe_deaccumulate_by_member,
)
from eval.evaluators.precip_scores.runner import (
    _render_pdf,
    _write_csv,
    aggregate_rows,
    summarize,
)

LOG = logging.getLogger(__name__)
MM = 1000.0


def score_gribs(
    *,
    model_grib_tpl: str,
    truth_grib_tpl: str,
    dates: list[str],
    out_dir: Path,
    baseline_grib_tpl: str = "",
    interp_index_cache: str = "",
    steps: list[int] | None = None,
    members: list[int] | None = None,
    var: str = "tp",
    wet_threshold_mm: float = M.WET_THRESHOLD_MM,
    run_label: str = "",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    truth_src = PrecipTruthSource(truth_grib_tpl, var=var)
    baseline_src = None

    rows: list[dict] = []
    grid_checked = False
    for date in dates:
        truth_by_step = truth_src.preload(date)
        model_by_key, m_lats, m_lons = _read_grib_var(
            Path(model_grib_tpl.format(date=date)), var)
        model_by_key, _acc = maybe_deaccumulate_by_member(
            model_by_key, context=f"model tp {date}")
        if not grid_checked:
            truth_src.verify_grid(m_lats, m_lons)
            check_grid_match(m_lats, m_lons,
                             truth_src._lats, truth_src._lons,
                             context="model-vs-truth GRIB grids")
            if baseline_grib_tpl:
                baseline_src = LresInterpBaseline(
                    baseline_grib_tpl, interp_index_cache or None, var=var)
                baseline_src.ensure_index(m_lats, m_lons, probe_date=date)
            grid_checked = True

        date_members = sorted({num for num, _ in model_by_key})
        use_members = [m for m in date_members if not members or m in members]
        date_steps = sorted({s for _n, s in model_by_key})
        use_steps = [s for s in date_steps
                     if s in truth_by_step and (not steps or s in steps)]
        skipped = [s for s in date_steps if s not in truth_by_step]
        if skipped:
            LOG.warning("%s: steps %s have no truth — skipped", date, skipped)

        for step in use_steps:
            truth_mm = truth_by_step[step].astype(np.float64) * MM
            ens_sum = np.zeros_like(truth_mm)
            bl_sum = np.zeros_like(truth_mm) if baseline_src is not None else None
            mem_rows = []
            for member in use_members:
                yp_mm = model_by_key[(member, step)].astype(np.float64) * MM
                ens_sum += yp_mm
                mem = {"member": member,
                       "model": {**M.pair_scores(yp_mm, truth_mm),
                                 **M.field_stats(yp_mm, wet_threshold_mm=wet_threshold_mm)}}
                if baseline_src is not None:
                    bl_mm = baseline_src.load(date, step, member).astype(np.float64) * MM
                    bl_sum += bl_mm
                    mem["baseline"] = {**M.pair_scores(bl_mm, truth_mm),
                                       **M.field_stats(bl_mm, wet_threshold_mm=wet_threshold_mm)}
                mem_rows.append(mem)
            row = {"date": date, "step": step,
                   "truth": M.field_stats(truth_mm, wet_threshold_mm=wet_threshold_mm),
                   "model_ens_mean": M.pair_scores(ens_sum / len(use_members), truth_mm),
                   "members": mem_rows}
            if bl_sum is not None:
                row["baseline_ens_mean"] = M.pair_scores(bl_sum / len(use_members), truth_mm)
            rows.append(row)
            LOG.info("scored %s step %03d (%d members): model rmse=%.3f mm",
                     date, step, len(use_members),
                     M.nanmean([m["model"].get("rmse_mm") for m in mem_rows]))
        truth_src.release()
        if baseline_src is not None:
            baseline_src.release()

    if not rows:
        raise RuntimeError("nothing scored — no (date, step) had both model and truth fields")

    per_step = aggregate_rows(rows)
    summary = summarize(per_step)
    payload = {
        "meta": {
            "route": "grib (prepml/FDB retrieve-only)",
            "model_source": model_grib_tpl,
            "truth_source": f"grib:{truth_grib_tpl}",
            "baseline_source": (f"lres-nn:{baseline_grib_tpl}"
                                if baseline_src is not None else "none"),
            "var": var, "unit": "mm / 6h window",
            "n_slices": len(rows),
            "wet_threshold_mm": wet_threshold_mm,
            "checkpoint_id": "",
            "n_members": max(len(r["members"]) for r in rows),
            "negative_handling": "raw values in rmse/bias/corr; negatives "
                                 "clipped to 0 only inside quantile histograms",
        },
        "per_step": per_step,
        "summary": summary,
        "rows": rows,
    }
    (out_dir / "scores.json").write_text(json.dumps(payload, indent=2))
    _write_csv(out_dir / "scores_rows.csv", rows)
    _render_pdf(out_dir / "plots" / "precip_scores.pdf", payload,
                run_label=run_label or "grib-route")
    LOG.info("score_gribs: %d slices -> %s", len(rows), out_dir)
    return out_dir


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-grib-tpl", required=True)
    ap.add_argument("--truth-grib-tpl", required=True)
    ap.add_argument("--baseline-grib-tpl", default="")
    ap.add_argument("--interp-index-cache", default="")
    ap.add_argument("--dates", required=True,
                    help="Comma-separated YYYYMMDD list.")
    ap.add_argument("--steps", default="",
                    help="Comma-separated step filter (default: all with truth).")
    ap.add_argument("--members", default="",
                    help="Comma-separated member filter (default: all present).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--var", default="tp")
    ap.add_argument("--run-label", default="")
    args = ap.parse_args()
    score_gribs(
        model_grib_tpl=args.model_grib_tpl,
        truth_grib_tpl=args.truth_grib_tpl,
        baseline_grib_tpl=args.baseline_grib_tpl,
        interp_index_cache=args.interp_index_cache,
        dates=[d.strip() for d in args.dates.split(",") if d.strip()],
        steps=[int(s) for s in args.steps.split(",") if s.strip()] or None,
        members=[int(m) for m in args.members.split(",") if m.strip()] or None,
        out_dir=Path(args.out_dir),
        var=args.var,
        run_label=args.run_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
