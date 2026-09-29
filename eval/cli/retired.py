"""Tombstones of retired or renamed ``eval.cli`` subcommands.

Each name below still parses, so an old command line or script fails loudly
instead of with argparse's "invalid choice": it prints where the tool went and
exits with status 1. Every argument after the name is accepted and ignored.
``eval.cli.main`` also checks for these names before parsing, so that flags the
tombstone does not declare cannot turn the message into a usage error.
"""
from __future__ import annotations

import argparse
import sys

from eval.cli._common import Command

# name -> the one message printed before exiting with status 1
TOMBSTONES = {
    "membermaps": (
        "eval.cli membermaps was renamed to zoom_maps on 2026-09-29. "
        "Run `python -m eval.cli zoom_maps` with the same flags."
    ),
    "report": "eval.cli report was retired on 2026-09-29, no replacement.",
    "videogen": (
        "eval.cli videogen was retired on 2026-09-29; "
        "the code is in eval/_quarantine/20260929/videogen/."
    ),
}


def exit_with_tombstone(name: str) -> None:
    print(f"ERROR: {TOMBSTONES[name]}", file=sys.stderr)
    raise SystemExit(1)


def _register_for(name: str):
    def register(subparsers) -> argparse.ArgumentParser:
        p = subparsers.add_parser(name, help=TOMBSTONES[name], description=TOMBSTONES[name])
        p.add_argument("ignored", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
        return p
    return register


def _run_for(name: str):
    def run(args: argparse.Namespace) -> None:
        exit_with_tombstone(name)
    return run


COMMANDS = tuple(
    Command(name, "retired", text, _register_for(name), _run_for(name))
    for name, text in TOMBSTONES.items()
)
