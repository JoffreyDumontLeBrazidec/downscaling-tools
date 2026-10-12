"""``eval.cli zoom_maps``: zoomed single-member maps of the input, the truth and the model.

Called ``membermaps`` until 2026-09-29; that name is now a tombstone (``eval/cli/retired.py``).

The flags come from the parser of
``eval.evaluators.zoom_maps.core.plot_member_wind_maps``, which this command
inherits, so the two cannot drift apart. It reads no lane or host configuration.
"""
from __future__ import annotations

import argparse
import re

from eval.cli._common import Command
from eval.shared.regions import PROTOCOL_REGIONS

SUMMARY = "Render zoomed single-member maps of the input, the truth and the predictions."
DESCRIPTION = (
    "Render the single-member map set used for member-level case inspection (10 m wind speed by "
    "default, or msl, 2t, t_850, z_500; the field itself or its high-pass fine-scale part): "
    "EEFO input, operational ENFO truth and one prediction panel per --run, all with a shared "
    "colour scale, projection and title style. Sources are the retrieved predictions_*.nc files "
    "(which embed x, y and y_pred); --grib panels cover steps absent from the predictions (for "
    "example step 0 read from FDB or MARS). Diagnostic maps only, no scoring."
)


def _escape_bare_percent(parser: argparse.ArgumentParser) -> None:
    """Make every help string safe for argparse's %-formatting.

    argparse formats each help text with ``%``, so a literal percent sign has to be
    written ``%%``. One help text of that parser says "99% of ..." with a single
    percent sign, which made the command's ``--help`` crash with a TypeError. The file of
    the parser is being restyled on another branch, so the escape is done here. A help
    text that already formats correctly is not touched, so the fix stays harmless once the
    parser is corrected.
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
    from eval.evaluators.zoom_maps.core.plot_member_wind_maps import build_arg_parser as _core_parser
    parser = subparsers.add_parser(
        "zoom_maps",
        parents=[_core_parser(add_help=False)],
        help=SUMMARY,
        description=DESCRIPTION,
    )
    _escape_bare_percent(parser)
    parser.add_argument(
        "--region", choices=sorted(PROTOCOL_REGIONS), default=None,
        help=("A fixed box of the run-judging protocol (eval/shared/regions.py): sets --extent "
              "and, unless given, --region-tag. alps = 43-49 N, 4-16 E (drawn widened to the "
              "12-degree minimum span). Use it instead of typing the extent, so every read "
              "draws the same box."),
    )
    return parser


def apply_region(args: argparse.Namespace) -> argparse.Namespace:
    """Turn ``--region`` into ``--extent`` (lon_min lon_max lat_min lat_max) and the tag."""
    name = getattr(args, "region", None)
    if not name:
        return args
    lat_min, lat_max, lon_min, lon_max = PROTOCOL_REGIONS[name]
    args.extent = [lon_min, lon_max, lat_min, lat_max]
    if args.region_tag == "europe-cutout":
        args.region_tag = name
    return args


def run(args: argparse.Namespace) -> None:
    from eval.evaluators.zoom_maps.core.plot_member_wind_maps import run as zoom_maps_run
    raise SystemExit(zoom_maps_run(apply_region(args)))


COMMANDS = (Command("zoom_maps", "figures", SUMMARY, register, run),)
