"""``eval.cli membermaps``: single-member 10 m wind-speed cutout maps.

The flags come from the backend's own parser
(``eval._backends.region_plotting.plot_member_wind_maps``), which this command
inherits, so the two cannot drift apart. It reads no lane or host configuration.
"""
from __future__ import annotations

import argparse

from eval.cli._common import Command

SUMMARY = "Render single-member 10 m wind maps of the input, the truth and the predictions."
DESCRIPTION = (
    "Render the single-member 10 m wind-speed map set used for member-level case inspection: "
    "EEFO input, operational ENFO truth and one prediction panel per --run, all with a shared "
    "colour scale, projection and title style. Sources are the retrieved predictions_*.nc files "
    "(which embed x, y and y_pred); --grib panels cover steps absent from the predictions (for "
    "example step 0 read from FDB or MARS). Diagnostic maps only, no scoring."
)


def register(subparsers) -> argparse.ArgumentParser:
    from eval._backends.region_plotting.plot_member_wind_maps import build_arg_parser as _membermaps_parser
    return subparsers.add_parser(
        "membermaps",
        parents=[_membermaps_parser(add_help=False)],
        help=SUMMARY,
        description=DESCRIPTION,
    )


def run(args: argparse.Namespace) -> None:
    from eval._backends.region_plotting.plot_member_wind_maps import run as membermaps_run
    raise SystemExit(membermaps_run(args))


COMMANDS = (Command("membermaps", "figures", SUMMARY, register, run),)
