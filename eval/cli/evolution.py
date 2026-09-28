"""``eval.cli evolution``: plot how an experiment evolves against a reference run.

Reads ladder cards (``ladder.json``) and needs no lane or host configuration. The
plotting itself lives in ``eval.jobs.evolution``.
"""
from __future__ import annotations

import argparse

from eval.cli._common import Command

SUMMARY = "Plot how an experiment evolves against a reference run, the input and the target."
DESCRIPTION = (
    "Plot how an experiment is evolving against a reference run, the EEFO input and the ENFO "
    "target: one row per weather state, one column per metric family."
)


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("evolution", help=SUMMARY, description=DESCRIPTION)
    p.add_argument("--exp", action="append", required=True,
                   help="LABEL=/path/to/ladder.json (repeatable). `baseline:<lane>` resolves "
                        "to the lane baseline's archived ladder card.")
    p.add_argument("--ref", default=None,
                   help="LABEL=/path/to/ladder.json for the reference RUN, drawn as its own curve. "
                        "`baseline:<lane>` (e.g. baseline:o96_o320) resolves to the archived ladder "
                        "card of the lane BASELINE, the standard during-run and end-of-run comparison.")
    p.add_argument("--input", dest="input_ref", default=None,
                   help="LABEL=/path/to/flat.json for the INPUT anchor (required).")
    p.add_argument("--target", dest="target_ref", default=None,
                   help="LABEL=/path/to/flat.json for the TARGET anchor (required).")
    p.add_argument("--hline", action="append", default=[],
                   help="LABEL=/path/to/flat.json for any further flat anchor (repeatable).")
    p.add_argument("--allow-missing-references", action="store_true",
                   help="Bootstrap a lane that has no reference yet; the gap is stamped on the figure.")
    p.add_argument("--rows", default=None, help="Comma-separated weather states.")
    p.add_argument("--columns", default=None, help="Comma-separated metric families.")
    p.add_argument("--region", default="n.hem", help="Region to plot (default: n.hem).")
    p.add_argument("--title", default=None, help="Figure title (default: none).")
    p.add_argument("--out", required=True, help="Output figure path.")
    p.add_argument("--allow-mixed-support", action="store_true",
                   help="Allow curves that were scored on different supports.")
    return p


def _resolve_card_spec(spec: str) -> str:
    """`baseline:<lane>` or `LABEL=baseline:<lane>` -> the lane baseline's archived
    ladder card (LABEL defaults to baseline-<ckpt8>)."""
    label, _, path = spec.rpartition("=")
    if path.startswith("baseline:"):
        from eval.baseline import baseline_ladder_card
        auto_label, card = baseline_ladder_card(path.split(":", 1)[1])
        return f"{label or auto_label}={card}"
    return spec


def run(args: argparse.Namespace) -> None:
    from eval.jobs.evolution import main as evolution_main

    forwarded: list[str] = ["--out", str(args.out), "--region", args.region]
    for e in args.exp:
        forwarded += ["--exp", _resolve_card_spec(e)]
    if args.ref:
        forwarded += ["--ref", _resolve_card_spec(args.ref)]
    if args.input_ref:
        forwarded += ["--input", args.input_ref]
    if args.target_ref:
        forwarded += ["--target", args.target_ref]
    for h in args.hline:
        forwarded += ["--hline", h]
    if args.allow_missing_references:
        forwarded.append("--allow-missing-references")
    if args.rows:
        forwarded += ["--rows", args.rows]
    if args.columns:
        forwarded += ["--columns", args.columns]
    if args.title:
        forwarded += ["--title", args.title]
    if args.allow_mixed_support:
        forwarded.append("--allow-mixed-support")
    evolution_main(forwarded)


COMMANDS = (Command("evolution", "comparison", SUMMARY, register, run),)
