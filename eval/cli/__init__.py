"""Unified command line of the evaluation framework: ``python -m eval.cli <command>``.

Each command lives in its own module of this package and publishes a ``Command``
record in its ``COMMANDS`` tuple; ``build_parser`` collects them, and ``main``
dispatches. Commands that need a lane and a host go through ``_session`` first.

Groups (see ``python -m eval.cli --help``):

    discovery    list, describe
    pipeline     run, predict, prepare, evaluate, scoreboard, report
    comparison   evolution
    tc_tracks    tctracker, tccompare
    figures      membermaps, videogen
    maintenance  prepml-cleanup, config

The shared argument groups are in ``_common``; evaluator selection (``--only``)
is in ``_selection``.
"""
from __future__ import annotations

import argparse
import importlib
import logging

from eval.cli._common import ALL_EVALUATORS, DEFAULT_HOST, GROUP_TITLES, Command

# Modules that publish commands, in the order they appear in the help.
_COMMAND_MODULES = (
    "run", "predict", "prepare", "evaluate", "scoreboard", "report",
    "evolution", "tctracker", "tccompare", "membermaps", "videogen",
    "prepml_cleanup", "lane_config",
)

__all__ = ["ALL_EVALUATORS", "DEFAULT_HOST", "build_parser", "commands", "main"]


def commands() -> dict[str, Command]:
    """Every command by name (imports each command module)."""
    found: dict[str, Command] = {}
    for module_name in _COMMAND_MODULES:
        module = importlib.import_module(f"eval.cli.{module_name}")
        for command in module.COMMANDS:
            found[command.name] = command
    return found


def _grouped_help(found: dict[str, Command]) -> str:
    lines = ["commands:"]
    for group, title in GROUP_TITLES.items():
        members = [c for c in found.values() if c.group == group]
        if not members:
            continue
        lines.append(f"\n  {title}")
        for c in members:
            lines.append(f"    {c.name:<15} {c.summary}")
    lines.append("\nRun `python -m eval.cli <command> --help` for the flags of one command.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with one subparser per command."""
    found = commands()
    parser = argparse.ArgumentParser(
        prog="python -m eval.cli",
        description="Unified command line of the evaluation framework.",
        epilog=_grouped_help(found),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # help=SUPPRESS keeps argparse from listing the commands a second time,
    # ungrouped, above the grouped list in the epilog.
    subparsers = parser.add_subparsers(
        dest="subcommand", required=True, metavar="<command>", help=argparse.SUPPRESS,
    )
    # Set after add_subparsers: argparse builds each subcommand's prog from the
    # parent's usage at that moment, and it must stay "python -m eval.cli <name>".
    parser.usage = "python -m eval.cli <command> [options]"
    for command in found.values():
        parser_for_command = command.register(subparsers)
        parser_for_command.set_defaults(_command=command)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse args, resolve config when the command needs a lane, and dispatch."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    command: Command = args._command
    if command.needs_lane:
        from eval.cli._session import run_lane_command
        run_lane_command(args, command)
    else:
        command.run(args)
