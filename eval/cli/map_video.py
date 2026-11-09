"""``eval.cli map_video``: global or box map videos and stills of prediction files.

The flags come from ``eval.tools.map_video.build_arg_parser``, which this command inherits, so
the two cannot drift apart. It reads no lane or host configuration. The same renderer runs inside
``eval.cli evaluate`` as the ``tp_maps`` evaluator.
"""
from __future__ import annotations

import argparse

from eval.cli._common import Command

SUMMARY = "Render pixel-exact map videos and stills (tp by default) of prediction files."
DESCRIPTION = (
    "Interpolate a field from its native octahedral grid (whole or a crop such as the E2 box) to a "
    "regular latitude-longitude grid with ring-wise bilinear weights, draw one grid cell per pixel "
    "(0.1 degrees global = a 4K frame) with the precipitation classes 0.1-50 mm (white below), thin "
    "coastlines and a one-line title, and write an H.264 MP4 over valid time, a poster PNG and/or "
    "one PNG per file. Panels: pred, input, truth, or auto. Diagnostic figures only, no scoring."
)


def register(subparsers) -> argparse.ArgumentParser:
    from eval.tools.map_video.__main__ import build_arg_parser

    return subparsers.add_parser(
        "map_video", parents=[build_arg_parser(add_help=False)], help=SUMMARY, description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def run(args: argparse.Namespace) -> None:
    from eval.tools.map_video.__main__ import render

    render(args)


COMMANDS = (Command("map_video", "figures", SUMMARY, register, run),)
