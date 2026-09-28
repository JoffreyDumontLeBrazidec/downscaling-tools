"""``eval.cli membermaps``: single-member 10 m wind-speed cutout maps.

The flags come from the backend's own parser
(``eval.evaluators.membermaps.core.plot_member_wind_maps``), which this command
inherits, so the two cannot drift apart. It reads no lane or host configuration.
"""
from __future__ import annotations

import argparse
import re

from eval.cli._common import Command

SUMMARY = "Render single-member 10 m wind maps of the input, the truth and the predictions."
DESCRIPTION = (
    "Render the single-member 10 m wind-speed map set used for member-level case inspection: "
    "EEFO input, operational ENFO truth and one prediction panel per --run, all with a shared "
    "colour scale, projection and title style. Sources are the retrieved predictions_*.nc files "
    "(which embed x, y and y_pred); --grib panels cover steps absent from the predictions (for "
    "example step 0 read from FDB or MARS). Diagnostic maps only, no scoring."
)


def _escape_bare_percent(parser: argparse.ArgumentParser) -> None:
    """Make every help string safe for argparse's %-formatting.

    argparse formats each help text with ``%``, so a literal percent sign has to be
    written ``%%``. One help text of the backend's parser says "99% of ..." with a single
    percent sign, which made ``eval.cli membermaps --help`` crash with a TypeError. The
    backend file is being restyled on another branch, so the escape is done here. A help
    text that already formats correctly is not touched, so the fix stays harmless once the
    backend is corrected.
    """
    for action in parser._actions:
        text = action.help
        if not isinstance(text, str) or text is argparse.SUPPRESS:
            continue
        try:
            text % dict(vars(action), prog=parser.prog)
        except (TypeError, ValueError, KeyError):
            action.help = re.sub(r"%(?!\()", "%%", text.replace("%%", "%"))


def register(subparsers) -> argparse.ArgumentParser:
    from eval.evaluators.membermaps.core.plot_member_wind_maps import build_arg_parser as _membermaps_parser
    parser = subparsers.add_parser(
        "membermaps",
        parents=[_membermaps_parser(add_help=False)],
        help=SUMMARY,
        description=DESCRIPTION,
    )
    _escape_bare_percent(parser)
    return parser


def run(args: argparse.Namespace) -> None:
    from eval.evaluators.membermaps.core.plot_member_wind_maps import run as membermaps_run
    raise SystemExit(membermaps_run(args))


COMMANDS = (Command("membermaps", "figures", SUMMARY, register, run),)
