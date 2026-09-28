"""``eval.cli prepare``: build truth-aware bundles without predicting."""
from __future__ import annotations

import argparse
from pathlib import Path

from eval.cli._bundles import _assert_serial_prepare_context
from eval.cli._common import LOG, Command, add_common_args, add_lane_override_args

SUMMARY = "Build truth-aware input bundles only, without predicting."


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("prepare", help=SUMMARY, description=SUMMARY)
    add_common_args(p)
    add_lane_override_args(p)
    p.add_argument(
        "--source-grib-root", required=True,
        help="Root directory of the source GRIB files.",
    )
    p.add_argument(
        "--bundle-dir", default=None,
        help="Output directory for the bundles (default: <output-dir>/bundles).",
    )
    return p


def cmd_prepare(args: argparse.Namespace, lane_config: dict, host_config: dict, output_dir: Path) -> None:
    """Build truth-aware bundles only (no prediction)."""
    _assert_serial_prepare_context()
    from eval.prepare.builder import build_bundles

    prepare_cfg = lane_config.get("prepare")
    if not prepare_cfg:
        raise SystemExit(f"Lane '{args.lane}' has no 'prepare:' section in its config.")

    source_grib_root = args.source_grib_root
    bundle_dir_arg = getattr(args, "bundle_dir", None)
    bundle_dir = Path(bundle_dir_arg) if bundle_dir_arg else output_dir / "bundles"

    predict_cfg = lane_config.get("predict", {})
    bundle_pairs_raw = predict_cfg.get("bundle_pairs", [])
    if isinstance(bundle_pairs_raw, str):
        bundle_pairs_raw = [bp.strip() for bp in bundle_pairs_raw.split(",") if bp.strip()]

    build_bundles(
        lane_config=lane_config,
        bundle_dir=bundle_dir,
        source_grib_root=source_grib_root,
        dates=list(predict_cfg.get("dates", [])),
        steps=[int(s) for s in predict_cfg.get("steps", [])],
        members=[int(m) for m in predict_cfg.get("members", [])],
        bundle_pairs=list(bundle_pairs_raw),
        verification_path=bundle_dir.parent / "bundle_build_verification.json",
    )
    LOG.info("Bundle preparation complete. Bundles in: %s", bundle_dir)


def run(args: argparse.Namespace, session) -> None:
    cmd_prepare(args, session.lane_config, session.host_config, session.output_dir)


COMMANDS = (Command("prepare", "pipeline", SUMMARY, register, run, needs_lane=True),)
