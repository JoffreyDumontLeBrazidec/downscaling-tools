"""``eval.cli videogen``: render MP4 videos of downscaling predictions.

The backend is ``eval.tools.videogen``. Scenes are not listed here so that
this module stays cheap to import; the backend validates ``--scene`` against its
own ``SCENES`` registry.
"""
from __future__ import annotations

import argparse

from eval.cli._common import Command

SUMMARY = "Render MP4 videos of downscaling predictions."


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("videogen", help=SUMMARY, description=SUMMARY)
    p.add_argument("--scene", required=True,
                   help="Scene name (see eval.tools.videogen.scenes.SCENES).")
    p.add_argument("--mode", choices=("preview", "all"), default="preview",
                   help="preview renders one frame, all renders the whole video (default: preview).")
    p.add_argument("--preview-valid", default=None,
                   help="Valid time YYYY-MM-DD for preview mode.")
    p.add_argument("--predictions-dir", default=None,
                   help="Override the scene's predictions_dir.")
    p.add_argument("--output-dir", default=None,
                   help="Override the scene's output_dir.")
    p.add_argument("--ckpt-label", default=None,
                   help="Override the scene's ckpt_label (cosmetic).")
    return p


def run(args: argparse.Namespace) -> None:
    from eval.tools.videogen.__main__ import main as videogen_main
    forwarded = ["--scene", args.scene, "--mode", args.mode]
    if args.preview_valid:
        forwarded += ["--preview-valid", args.preview_valid]
    if args.predictions_dir:
        forwarded += ["--predictions-dir", str(args.predictions_dir)]
    if args.output_dir:
        forwarded += ["--output-dir", str(args.output_dir)]
    if args.ckpt_label:
        forwarded += ["--ckpt-label", args.ckpt_label]
    videogen_main(forwarded)


COMMANDS = (Command("videogen", "figures", SUMMARY, register, run),)
