"""``eval.cli final_battery``: the realism battery for a COMPLETELY TRAINED model only.

Owner, 18 Nov 2026: "make sure this battery is here in eval.cli as a battery of diagnostics that we apply only when
models are completely trained ... it is something we should evaluate again when the model is completely ready".
So it is not part of ``judge`` or any development read, and it refuses to start without ``--final-model``, the name
of the finished checkpoint it is run on (written into the summary page).

Three parts, each the reference kit of a 17 Nov 2026 realism investigation of GL512 450k (docs master,
epics/fine-scale-o320-o1280/in-progress/):

* terrain: temperature, humidity and pressure over terrain and coasts (T1-T8, Z1-Z6, 2d/skt set D, zoom sheets);
  note 20261117_realism_t2m_terrain.md s6, support/20261117_realism_t2m/BATTERY.md.
* consistency: ensemble and physical consistency (C1-C14, member maps); note
  20261117_realism_consistency_ensemble_physics.md s6.
* wind: 10 m wind and cyclones (W1-W9 = D1-D9 of the note); note 20261117_realism_wind_cyclones.md s6.

Actions: ``run`` (the battery, then the summary page), ``plan`` (print the environment and every command, run
nothing: the hand-off to a cluster shell), ``summary`` (re-render ``summary.md`` from an existing output folder).
CPU only. Input: member-mode box predictions (one draw per input member) on the 30 dates of 2023 at leads 12-72 h,
with all model outputs saved (2d, skt, tcw switch on the 2d/skt part).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eval.cli._common import Command

SUMMARY = "Final-model realism battery (terrain, consistency, wind): finished models only, needs --final-model."
DESCRIPTION = __doc__.split("\n\n", 1)[1] if __doc__ else SUMMARY


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.replace(",", " ").split())


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("final_battery", help=SUMMARY, description=DESCRIPTION,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("run", "plan", "summary"), help="run the battery, print the plan, or re-render the summary")
    p.add_argument("--final-model", default=None,
                   help="REQUIRED for run/plan: the finished checkpoint this is run on, e.g. 'FS-245 100k EMA-2000'. "
                        "The battery is for completely trained models only (owner, 18 Nov 2026).")
    p.add_argument("--row", default=None, help="Prediction folder stem: <pred-root>/pred/<row>_L<LLL>/predictions_*.nc")
    p.add_argument("--pred-root", type=Path, default=None, help="Root holding pred/<row>_L<LLL>/ (box member-mode layout).")
    p.add_argument("--out", type=Path, required=True, help="Output folder of this row's battery.")
    p.add_argument("--parts", nargs="+", choices=("terrain", "consistency", "wind"), default=["terrain", "consistency", "wind"])
    p.add_argument("--steps", nargs="+", default=None, help="Run only these steps (prefix match, e.g. terrain.z wind).")
    p.add_argument("--leads", default="12 24 36 48 60 72", help="Leads in hours (default: 12 24 36 48 60 72).")
    p.add_argument("--wind-leads", default="24 72", help="Leads of the wind part (default: 24 72, as on 17 Nov).")
    p.add_argument("--workers", type=int, default=6, help="Worker processes per kit (6 on a Jupiter login node).")
    p.add_argument("--consistency-workers", type=int, default=3, help="Workers of the consistency kit (about 5 GB each; default 3).")
    p.add_argument("--label", default=None, help="Model name on figures (default: --final-model).")
    p.add_argument("--user-root", default=None, help="Jupiter user root (default /e/data1/jureap-data/ecmwf/users/jdumont).")
    p.add_argument("--member-stores", default=None, help="Folder with stores/ and stores_leads/ (truth and input pf zarrs).")
    p.add_argument("--datasets", default=None, help="Datasets root (O1280 forcings, v4 control stores, matrix, grids).")
    p.add_argument("--orog320", default=None, help="O320 input orography grib.")
    p.add_argument("--no-edge5", action="store_true", help="Skip the 5-degree-mask consistency rerun.")
    p.add_argument("--no-maps", action="store_true", help="Skip the zoom sheets and member maps (no cartopy needed).")
    p.add_argument("--xset-rows", default="", help="XSET-1 row root (Atos) for W9 hurricane structure; empty = skipped.")
    p.add_argument("--xset-cases", default="", help="XSET-1 tc_cases.csv for W9.")
    p.add_argument("--force", action="store_true", help="Rerun steps that have a done marker.")
    return p


def run(args: argparse.Namespace) -> None:
    from eval.final_battery import runner, summary

    if args.action == "summary":
        print(summary.write(args.out))
        return
    if not args.final_model:
        raise SystemExit(
            "eval.cli final_battery: --final-model is required. This battery runs only on a completely trained model "
            "(owner, 18 Nov 2026); name the finished checkpoint, e.g. --final-model 'FS-245 100k EMA-2000'.")
    if not args.row or not args.pred_root:
        raise SystemExit("eval.cli final_battery: --row and --pred-root are required for run and plan.")
    cfg = runner.Config(
        final_model=args.final_model, row=args.row, pred_root=args.pred_root, out=args.out, label=args.label or "",
        leads=_ints(args.leads), wind_leads=_ints(args.wind_leads), parts=tuple(args.parts), workers=args.workers,
        consistency_workers=args.consistency_workers,
        edge5=not args.no_edge5, maps=not args.no_maps, xset_rows=args.xset_rows, xset_cases=args.xset_cases,
        only_steps=tuple(args.steps or ()), force=args.force,
        **{k: v for k, v in (("user_root", args.user_root), ("member_stores", args.member_stores),
                             ("datasets", args.datasets), ("orog320", args.orog320)) if v},
    )
    raise SystemExit(runner.run(cfg, dry_run=args.action == "plan"))


COMMANDS = (Command("final_battery", "final", SUMMARY, register, run),)
