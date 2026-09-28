"""``eval.cli tctracker``: produce ECMWF tctracker basin-track archives.

The pair ``tctracker`` + ``tccompare`` is the month-scale, track-based tropical
cyclone diagnostic for prepml campaigns. It never feeds a scoreboard: TC verdicts
stay with the box-based raw-extremes ``tc`` evaluator on the canonical support.
Runbook: docs/epics/completed_epics/tc_track/TCTRACKER_EVAL_CLI.md (month-scale
section) in the project docs on hpc-login.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.cli._common import LOG, Command, add_common_args, add_lane_override_args

SUMMARY = "Produce ECMWF tctracker basin-track archives for an expver and its references."
DESCRIPTION = (
    "Produce, verify and parse ECMWF tctracker basin-track archives. By default it "
    "tracks one rd expver. With --track-sources the same tracker settings also run "
    "over the ctrl, target and input references, so every track set shares one "
    "support; operational references are cached under <scratch>/eval/tcrefs/. "
    "Compare the results with `eval.cli tccompare`."
)


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("tctracker", help=SUMMARY, description=DESCRIPTION)
    add_common_args(p)
    add_lane_override_args(p)
    p.add_argument("--expver", required=True, help="PrepML/FDB expver to track, e.g. j761.")
    p.add_argument("--output-dir", default=None, help="Tracker run root (default: <scratch>/eval/<lane>/tctracker/<expver>).")
    p.add_argument("--time", default=None, help="Forecast cycle hour, e.g. 00.")
    p.add_argument("--start-step", type=int, default=None, help="First forecast step (tctracker -s).")
    p.add_argument("--end-step", type=int, default=None, help="Last forecast step (tctracker -f).")
    p.add_argument("--step-interval", type=int, default=None, help="Forecast step interval (tctracker -i).")
    p.add_argument("--grid", type=int, default=None, help="Output grid resolution (tctracker -r).")
    p.add_argument("--class", dest="fdb_class", default=None, help="FDB class (tctracker -C).")
    p.add_argument("--type", dest="fdb_type", default=None, help="FDB type (tctracker -T).")
    p.add_argument("--stream", default=None, help="FDB stream (tctracker -S).")
    p.add_argument("--vorticity", choices=("true", "false"), default=None, help="Whether tctracker reads vorticity (tctracker -v).")
    p.add_argument("--model-keyword", default=None, help="Value exported as model_keyword before tctracker runs.")
    p.add_argument("--module", default=None, help="Environment module to load before tctracker runs (default: tctracker).")
    p.add_argument("--overwrite", action="store_true", default=False, help="Re-run targets even if their tar already exists.")
    p.add_argument("--verify-only", action="store_true", default=False, help="Only verify existing tars and manifests; do not run tctracker.")
    p.add_argument("--parse-only", action="store_true", default=False, help="Only parse existing tracks; do not run tctracker.")
    p.add_argument("--slurm-script", default=None, help="Write a resumable sbatch script per source to this path (suffixed by role when there are several) and exit.")
    p.add_argument("--role", default="model", help="Role label of this expver's tracks (default: model).")
    p.add_argument(
        "--track-sources", default=None,
        help=(
            "Comma-separated roles to track in one call, e.g. 'model,ctrl=j95z,target,input'. "
            "A bare 'target' or 'input' resolves from the lane's tctracker.sources or prepml "
            "blocks; reference (non-rd) sources are cached under <scratch>/eval/tcrefs/."
        ),
    )
    p.add_argument("--months", default=None, help="Comma-separated YYYYMM months, expanded to daily dates (alternative to --dates).")
    p.add_argument("--no-check-fdb", action="store_true", default=False, help="Skip the FDB completeness check for rd expvers.")
    p.add_argument("--track-incomplete", action="store_true", default=False, help="Also track partial or empty FDB dates (default: skip them with a warning).")
    return p


def cmd_tctracker(args: argparse.Namespace, lane_config: dict, host_config: dict, output_dir: Path) -> None:
    """Run, verify, or parse ECMWF tctracker archives for one or more sources.

    Default = the single --expver under --role (back-compatible). With
    --track-sources, the same tracker settings run over every requested role
    (model expver + ctrl/target/input references) so all tracks share ONE
    support; reference tars land in the shared tcrefs cache.
    """
    import dataclasses

    from eval.evaluators.tctracks.core import (
        build_config, completeness_report, expand_months, parse_atlantic_tracks,
        parse_sources_arg, render_slurm_script, resolve_source_configs,
        run_batch, verify_outputs, write_atlantic_summary,
        write_verification_summary,
    )
    from eval.evaluators.tctracks.core.tables import parse_run_root

    if getattr(args, "months", None) and not getattr(args, "dates", None):
        args.dates = ",".join(expand_months(args.months))

    base_config = build_config(args, lane_config, host_config, output_dir)
    roles = parse_sources_arg(getattr(args, "track_sources", None))
    if roles:
        model_override = roles.get("model")
        if model_override and model_override != base_config.expver:
            raise SystemExit("--track-sources model=<expver> must match --expver")
        sources = resolve_source_configs(base_config, roles, lane_config, host_config)
    else:
        sources = [(getattr(args, "role", "model") or "model", base_config.expver, base_config)]

    if getattr(args, "slurm_script", None):
        base_path = Path(args.slurm_script)
        for role, source_id, config in sources:
            script = render_slurm_script(
                config,
                code_root=host_config["code_root"],
                venv_activate=host_config["environment_setup"]["venv_activate"],
            )
            script_path = base_path if len(sources) == 1 else base_path.with_name(
                f"{base_path.stem}_{role}_{source_id}{base_path.suffix or '.sbatch'}"
            )
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(script, encoding="utf-8")
            script_path.chmod(script_path.stat().st_mode | 0o755)
            LOG.info("tctracker sbatch script (%s=%s) written to %s", role, source_id, script_path)
        return

    failures: list[str] = []
    for role, source_id, config in sources:
        LOG.info("=== tctracker source %s=%s (%s/%s/%s) -> %s",
                 role, source_id, config.fdb_class, config.stream, config.expver,
                 config.output_dir)
        if not getattr(args, "verify_only", False) and not getattr(args, "parse_only", False):
            # Warn-only FDB completeness preflight for rd expvers: partial or
            # empty dates are skipped by default (a tracker run on a half-
            # written date would silently produce truncated tracks).
            if config.fdb_class == "rd" and not getattr(args, "no_check_fdb", False):
                report = completeness_report(config)
                config.manifests_dir.mkdir(parents=True, exist_ok=True)
                (config.manifests_dir / "fdb_completeness.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8",
                )
                if report["checked"] and not getattr(args, "track_incomplete", False):
                    keep = tuple(d for d in config.dates if d in set(report["complete"]))
                    if keep != config.dates:
                        LOG.warning("%s=%s: tracking %d/%d complete dates",
                                    role, source_id, len(keep), len(config.dates))
                        config = dataclasses.replace(config, dates=keep)
            if not config.dates:
                LOG.warning("%s=%s: no complete dates to track; skipping source", role, source_id)
                continue
            try:
                run_batch(config)
            except RuntimeError as exc:
                failures.append(f"{role}={source_id}: {exc}")

        verification = verify_outputs(config)
        md_path, json_path = write_verification_summary(config, verification)
        LOG.info("verification (%s=%s) written to %s", role, source_id, md_path)
        if verification["issues"] and not getattr(args, "parse_only", False):
            failures.append(f"{role}={source_id}: {len(verification['issues'])} verification issue(s); see {json_path}")

        # Parse the WHOLE run root, not just this invocation's targets:
        # member-sliced production jobs run concurrently against one run root,
        # and a per-config parse would leave whichever member finished last.
        parsed_dir = parse_run_root(config.output_dir, role=role, source_id=source_id)
        LOG.info("parsed tables (%s=%s) written to %s", role, source_id, parsed_dir)
        if role == "model":  # keep the historical Atlantic summary artifacts
            tracks = parse_atlantic_tracks(config)
            write_atlantic_summary(config, tracks)

    if failures:
        raise RuntimeError("tctracker source failures:\n" + "\n".join(failures))


def run(args: argparse.Namespace, session) -> None:
    cmd_tctracker(args, session.lane_config, session.host_config, session.output_dir)


COMMANDS = (Command("tctracker", "tc_tracks", SUMMARY, register, run, needs_lane=True),)
