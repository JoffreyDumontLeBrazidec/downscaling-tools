"""``eval.cli prepml-cleanup``: list and clean tracked prepml experiments.

Bypasses lane and host configuration: it works purely on the local ledger of
expvers and on ecFlow, where it force-runs the ``run/delete/*`` tasks.
"""
from __future__ import annotations

import argparse

from eval.cli._common import Command

SUMMARY = "List and clean tracked prepml experiments through ecFlow."


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("prepml-cleanup", help=SUMMARY, description=SUMMARY)
    p.add_argument(
        "--list", action="store_true",
        help="Print the ledger with the ecFlow state and exit.",
    )
    p.add_argument(
        "--expver", action="append", default=[],
        help="Expver to clean. Repeat the flag for several.",
    )
    p.add_argument(
        "--scope", choices=("fdb", "all"), default="fdb",
        help="Which run/delete tasks to force-run. fdb (default) matches the announcement; "
             "all is fdb + mars + s3 + quaver + workdir (the catalogue is always kept).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the ecflow_client commands that would run, without running them.",
    )
    p.add_argument(
        "--yes", action="store_true",
        help="Skip the final confirmation prompt.",
    )
    p.add_argument(
        "--no-ecflow", action="store_true",
        help="Skip the ecFlow state check when listing (faster, works offline).",
    )
    return p


def run(args: argparse.Namespace) -> None:
    from eval.predict.prepml_cleanup import main as prepml_cleanup_main
    forwarded: list[str] = []
    if args.list:
        forwarded.append("--list")
    for ev in args.expver:
        forwarded += ["--expver", ev]
    forwarded += ["--scope", args.scope]
    if args.dry_run:
        forwarded.append("--dry-run")
    if args.yes:
        forwarded.append("--yes")
    if args.no_ecflow:
        forwarded.append("--no-ecflow")
    raise SystemExit(prepml_cleanup_main(forwarded))


COMMANDS = (Command("prepml-cleanup", "maintenance", SUMMARY, register, run),)
