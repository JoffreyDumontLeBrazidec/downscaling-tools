"""CLI entry — ``python -m eval._backends.videogen``.

Also called from ``eval.cli videogen`` (see ``eval/cli.py``).

Examples
--------
Preview the franklin_dual scene at a specific valid time::

    python -m eval._backends.videogen --scene franklin_dual --mode preview \
        --preview-valid 2023-08-29

Render the full video::

    python -m eval._backends.videogen --scene franklin_dual --mode all

Override scene defaults at the CLI::

    python -m eval._backends.videogen --scene franklin --mode all \
        --predictions-dir /path/to/other/predictions \
        --output-dir /tmp/my_videos
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .pipeline import make_video, render_preview
from .scenes import SCENES


def _parse_valid_time(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval._backends.videogen")
    parser.add_argument("--scene", required=True, choices=sorted(SCENES))
    parser.add_argument("--mode", choices=("preview", "all"), default="preview")
    parser.add_argument(
        "--preview-valid", type=_parse_valid_time, default=None,
        help="Valid time for preview frame (YYYY-MM-DD).",
    )
    parser.add_argument("--predictions-dir", type=Path, default=None,
                        help="Override scene's predictions_dir.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override scene's output_dir.")
    parser.add_argument("--ckpt-label", default=None,
                        help="Override scene's ckpt_label (cosmetic, used in the suptitle).")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    scene = SCENES[args.scene]
    overrides: dict = {}
    if args.predictions_dir is not None:
        overrides["predictions_dir"] = args.predictions_dir
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.ckpt_label is not None:
        overrides["ckpt_label"] = args.ckpt_label
    if overrides:
        scene = replace(scene, **overrides)

    if args.mode == "preview":
        out = render_preview(scene, valid_time=args.preview_valid)
        print(f"Preview: {out}")
    else:
        out = make_video(scene)
        print(f"Video:   {out}")


if __name__ == "__main__":
    main()
