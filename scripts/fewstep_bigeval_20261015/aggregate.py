#!/usr/bin/env python3
"""Aggregate the few-step big eval (o96->o320) over all rows, in the run-judging protocol's order.

    python scripts/fewstep_bigeval_20261015/aggregate.py \
        --eval-root <EVAL_ROOT> --rows scripts/fewstep_bigeval_20261015/rows.yaml \
        --reference edm_pw30c0 --pred-root <PRED_ROOT> [--pred-root B=<PRED_ROOT_B>] \
        --out <OUT_DIR>

Reads ``<EVAL_ROOT>/<set>/<row_label>/evaluators/<evaluator>/`` (the layout of
``python -m eval.cli evaluate --predictions-dir X --output-dir <EVAL_ROOT>/<set>/<row_label>``;
a second view of one evaluator may sit in ``evaluators/<evaluator>__<view>`` or
``<EVAL_ROOT>/<set>/<row_label>/<view>/evaluators/<evaluator>``) and, for costs and the fallbacks,
the prediction tree ``<PRED_ROOT>[/<set>]/<arm>/step_NNNNNN/<sampler>/predictions/``.

Writes into ``--out``: 1_structure.md, 2_spectra.md, 3_extremes.md, 4_scores.md, 5_costs.md,
summary.md, and every table as CSV under csv/. A missing input never stops the run: the cell or
the section says "Not available: <reason>".

Rows file (YAML): a list (or {"rows": [...]}) of {label, arm, sampler, calls, sets[, step,
predictions: {set: dir}]}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from aggregate_lib import costs, extremes, scores, spectra, structure, summary  # noqa: E402
from aggregate_lib.common import Context, load_rows, run_section  # noqa: E402

SECTIONS = [("1. Structure", "1_structure.md", structure.section),
            ("2. Spectra", "2_spectra.md", spectra.section),
            ("3. Extremes", "3_extremes.md", extremes.section),
            ("4. Scores", "4_scores.md", scores.section),
            ("5. Costs", "5_costs.md", costs.section)]


def _csv_list(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--eval-root", required=True, type=Path)
    p.add_argument("--rows", required=True, type=Path, help="rows YAML")
    p.add_argument("--reference", default="edm_pw30c0", help="label of the reference row")
    p.add_argument("--pred-root", action="append", default=[],
                   help="prediction root; 'SET=PATH' for one set (repeatable). A plain PATH is "
                        "searched as PATH/<set>/<arm>/... and PATH/<arm>/...")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--only", default=None, help="comma list of parts to run: structure,spectra,"
                                                "extremes,scores,costs")
    p.add_argument("--t-crit", type=float, default=2.03)
    p.add_argument("--spectra-tol", type=float, default=0.03, help="|row - ref| per bin")
    p.add_argument("--truth-band", type=float, default=0.03, help="listed when |ratio - 1| exceeds it")
    p.add_argument("--min-km", type=float, default=0.0,
                   help="bins finer than this are tabulated but not listed or gated")
    p.add_argument("--global-lmin", type=int, default=20)
    p.add_argument("--global-nbins", type=int, default=30)
    p.add_argument("--shape-band", default="mid")
    p.add_argument("--score-vars", default=None, help="default 10u,10v,ws,2t,msl,t_850,z_500,tp")
    p.add_argument("--score-domains", default=None, help="default global,n.hem,tropics,s.hem,europe")
    p.add_argument("--test-domains", default=None, help="paired-test domains, default global,europe")
    p.add_argument("--recompute-vars", default="tp",
                   help="paired-test variables always recomputed from the prediction files (tp is "
                        "clipped at 0 there; probabilistic does not clip). '' for none")
    p.add_argument("--precip-sets", default=None, help="sets for the precipitation fallback")
    p.add_argument("--tp-scale", type=float, default=1000.0, help="tp unit to mm (m -> mm)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    pred_roots: dict[str, Path] = {}
    for item in a.pred_root:
        if "=" in item and not Path(item).exists():
            k, v = item.split("=", 1)
            pred_roots[k] = Path(v)
        else:
            pred_roots[""] = Path(item)
    options = {"t_crit": a.t_crit, "spectra_tol": a.spectra_tol, "truth_band": a.truth_band,
               "min_km": a.min_km, "global_lmin": a.global_lmin, "global_nbins": a.global_nbins,
               "shape_band": a.shape_band, "score_vars": _csv_list(a.score_vars),
               "score_domains": _csv_list(a.score_domains), "test_domains": _csv_list(a.test_domains),
               "precip_sets": _csv_list(a.precip_sets), "tp_scale": a.tp_scale,
               "recompute_vars": _csv_list(a.recompute_vars) or []}
    a.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(a.rows)
    ctx = Context(eval_root=a.eval_root, rows=rows, reference=a.reference, out_dir=a.out,
                  pred_roots=pred_roots, options=options)
    only = set(_csv_list(a.only) or [])
    results = {}
    for title, fname, fn in SECTIONS:
        if only and not any(o in title.lower() for o in only):
            continue
        sec = run_section(title, fn, ctx)
        results[title] = sec
        (a.out / fname).write_text(sec.md if sec.md.endswith("\n") else sec.md + "\n")
        print(f"wrote {a.out / fname}")
    (a.out / "summary.md").write_text(summary.build(ctx, results))
    print(f"wrote {a.out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
