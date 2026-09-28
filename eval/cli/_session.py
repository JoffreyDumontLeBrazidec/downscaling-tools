"""What every lane-based command does before and after its own work.

Before: load the lane and host configuration, apply the host's environment,
choose the evaluators and the output directory, and write ``effective_config.json``
(or print it and stop for ``--dry-run``). After: write the ``--vs-baseline`` diff.
The commands that use this are the ones registered with ``needs_lane=True``:
run, predict, prepare, evaluate, scoreboard, tctracker and tccompare.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from eval.cli._bundles import _resolve_predict_input_root
from eval.cli._common import (
    ALL_EVALUATORS, DEFAULT_HOST, LOG, Command, _build_effective_config, _build_lane_overrides,
    _parse_int_csv, _resolve_output_dir, _write_effective_config,
)
from eval.cli._environment import _apply_host_module_loads, _assert_metview_for_regridded_tc
from eval.cli._selection import _resolve_evaluators
from eval.config.loader import (
    default_host_for_stage,
    load_host,
    load_lane,
    validate_lane_host_compatible,
)
from eval.evaluators.tc.comparison_contract import require_lane_analysis_reference


@dataclass
class Session:
    """The resolved inputs a lane command works from."""

    lane_name: str
    host_name: str
    lane_config: dict
    host_config: dict
    output_dir: Path
    evaluators: list[str] = field(default_factory=list)
    lane_overrides: dict = field(default_factory=dict)


def _output_dir_for(subcommand: str, args: argparse.Namespace, host_config: dict,
                    lane_name: str, lane_config: dict) -> Path:
    """Where this command writes: an explicit flag when given, else a timestamped run directory."""
    if subcommand == "scoreboard" and hasattr(args, "eval_dir") and args.eval_dir:
        return Path(args.eval_dir)
    if subcommand == "evaluate" and hasattr(args, "predictions_dir") and args.predictions_dir:
        # Place evaluator outputs alongside predictions, unless --output-dir overrides
        explicit_out = getattr(args, "output_dir", None)
        return Path(explicit_out) if explicit_out else Path(args.predictions_dir).parent
    if subcommand in ("run", "predict") and getattr(args, "output_dir", None):
        return Path(args.output_dir)
    if subcommand == "prepare":
        bundle_dir_arg = getattr(args, "bundle_dir", None)
        return Path(bundle_dir_arg).parent if bundle_dir_arg else _resolve_output_dir(host_config, lane_name)
    if subcommand == "tctracker":
        from eval._backends.tctracker.pipeline import default_output_dir
        explicit_out = getattr(args, "output_dir", None)
        return Path(explicit_out) if explicit_out else default_output_dir(
            host_config, lane_name, lane_config, getattr(args, "expver")
        )
    if subcommand == "tccompare":
        from eval._backends.tctracker.pipeline import _lane_short_name
        label = getattr(args, "label", None) or "_".join(
            m.strip() for m in str(args.months).split(",") if m.strip()
        )
        explicit_out = getattr(args, "out", None)
        return Path(explicit_out) if explicit_out else (
            Path(host_config["scratch_root"]) / "eval"
            / _lane_short_name(lane_name, lane_config) / "tctracks" / label
        )
    return _resolve_output_dir(host_config, lane_name)


def run_lane_command(args: argparse.Namespace, command: Command) -> None:
    """Resolve configuration, run ``command``, then write the vs-baseline diff if asked."""
    subcommand = command.name

    # --- Resolve config ---
    lane_name = args.lane

    lane_overrides = _build_lane_overrides(args)

    try:
        lane_config = load_lane(lane_name, overrides=lane_overrides or None)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Lane config not found: '{lane_name}'. "
            f"Available lanes are YAML files in eval/config/lanes/. Error: {exc}"
        ) from exc
    except Exception as exc:
        raise SystemExit(f"Failed to load lane config '{lane_name}': {exc}") from exc

    host_name = args.host or default_host_for_stage(lane_config, subcommand) or DEFAULT_HOST
    try:
        validate_lane_host_compatible(lane_name, lane_config, host_name, stage=subcommand)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    try:
        host_config = load_host(host_name)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Host config not found: '{host_name}'. "
            f"Available hosts are YAML files in eval/config/hosts/. Error: {exc}"
        ) from exc
    except Exception as exc:
        raise SystemExit(f"Failed to load host config '{host_name}': {exc}") from exc

    # --- Surface host-declared stage caveats (WARN ONLY, never a gate) ---
    # Hosts may declare `stage_warnings: {<stage-name>: "text"}`. A host can be perfectly
    # valid for one checkpoint class and degrading for another, so this warns and
    # continues rather than refusing (defaults-not-validators doctrine).
    _stage_warning = (host_config.get("stage_warnings") or {}).get(subcommand)
    if _stage_warning:
        LOG.warning(
            "host %r on stage %r: %s", host_name, subcommand, " ".join(str(_stage_warning).split())
        )

    # --- Export host-declared env vars so subprocesses (predict.main, evaluators) see them ---
    # The host YAML's environment_setup.exports lists vars like DATA_DIR, GRID_DIR,
    # RESIDUAL_STATISTICS_DIR that OmegaConf interpolations and model loaders depend on.
    # Existing values in os.environ take precedence (so user overrides still work).
    host_exports = host_config.get("environment_setup", {}).get("exports", {}) or {}
    for key, value in host_exports.items():
        if key not in os.environ:
            os.environ[key] = str(value)

    # --- C5 (a): apply host module_loads so the inline eval subprocess gets the
    # same modules the sbatch would (e.g. ecmwf-toolbox -> metview for TC). ---
    _apply_host_module_loads(host_config)

    # --- Export lane-declared inference env vars (e.g. ANEMOI_INFERENCE_NUM_CHUNKS) ---
    # Apply the CLI --num-chunks override on top of lane predict.env so the
    # dry-run output and downstream subprocesses see the same merged value.
    predict_section = lane_config.get("predict")
    if isinstance(predict_section, dict):
        predict_env = dict(predict_section.get("env") or {})
        if getattr(args, "num_chunks", None) is not None:
            chunk_value = str(int(args.num_chunks))
            predict_env["ANEMOI_INFERENCE_NUM_CHUNKS"] = chunk_value
            predict_env["ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR"] = chunk_value
            predict_env["ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER"] = chunk_value
            predict_section["env"] = predict_env
        for key, value in predict_env.items():
            os.environ[key] = str(value)

    # --- Propagate --steps to evaluator sections ---
    # When --steps is passed, override not just predict.steps but also any
    # evaluator-specific steps (e.g. spectra_ecmwf_v2.steps) so evaluators don't
    # request forecast steps that don't exist in predictions.
    if getattr(args, "steps", None) is not None and subcommand in ("evaluate", "run"):
        cli_steps = _parse_int_csv(args.steps)
        for section_name, section_val in lane_config.items():
            if section_name != "predict" and isinstance(section_val, dict) and "steps" in section_val:
                section_val["steps"] = cli_steps
                lane_overrides.setdefault(section_name, {})["steps"] = cli_steps

    # --- Resolve evaluators (for subcommands that need them) ---
    evaluators: list[str] = []
    if subcommand in ("run", "evaluate", "scoreboard"):
        evaluators = _resolve_evaluators(args, lane_config)

    # --- Resolve output dir ---
    output_dir = _output_dir_for(subcommand, args, host_config, lane_name, lane_config)

    if subcommand in ("run", "predict") and "predict" in lane_config:
        input_root = _resolve_predict_input_root(
            args, lane_config, host_config, output_dir,
            prepare_bundles=False,
            allow_host_fallback=getattr(args, "mode", "manual") != "prepml",
        )
        if input_root:
            lane_config.setdefault("predict", {})["input_root"] = input_root

    if subcommand in ("run", "evaluate") and "tc" in lane_config:
        require_lane_analysis_reference(
            lane_name, (lane_config.get("tc") or {}).get("analysis_expid"),
        )

    # --- Build effective config ---
    effective = _build_effective_config(
        args, lane_config, host_config,
        lane_name, host_name, lane_overrides,
        evaluators, output_dir,
    )

    # --- Dry run ---
    if args.dry_run:
        if subcommand == "tctracker":
            from eval._backends.tctracker import build_config, dry_run_payload
            effective["tctracker"] = dry_run_payload(
                build_config(args, lane_config, host_config, output_dir)
            )
        print(json.dumps(effective, indent=2, default=str))
        if getattr(args, "mode", "manual") == "prepml" and subcommand in ("run", "predict"):
            from eval.predict.prepml_config import generate_prepml_config
            from eval.predict.prepml import resolve_expver
            try:
                resolved_expver = resolve_expver(getattr(args, "expver", None), lane_config)
                prepml_cfg = generate_prepml_config(
                    lane_config=lane_config,
                    checkpoint_path=getattr(args, "checkpoint", ""),
                    runner_override=getattr(args, "prepml_runner", None),
                )
                import yaml
                print("\n--- PrepML Config Preview ---")
                print(yaml.dump(prepml_cfg, default_flow_style=False, sort_keys=False))
                print(f"Expver: {resolved_expver}")
            except Exception as exc:
                print(f"\n--- PrepML Config Preview (error) ---\n{exc}")
        return

    # --- Preflight: write effective config ---
    config_path = _write_effective_config(effective, output_dir)
    LOG.info("Effective config written to %s", config_path)

    # --- Validate evaluator names ---
    unknown_evals = [e for e in evaluators if e not in ALL_EVALUATORS]
    if unknown_evals:
        raise SystemExit(
            f"Unknown evaluator(s) in lane config evaluator_groups: {unknown_evals}. "
            f"Valid evaluators: {ALL_EVALUATORS}"
        )

    # --- C5 (b): for run/evaluate with regridded/both TC, require metview now
    # so we fail loudly instead of silently degrading to native support. ---
    if subcommand in ("run", "evaluate"):
        _assert_metview_for_regridded_tc(lane_config, evaluators)

    # --- Dispatch ---
    command.run(args, Session(
        lane_name=lane_name, host_name=host_name, lane_config=lane_config,
        host_config=host_config, output_dir=output_dir, evaluators=evaluators,
        lane_overrides=lane_overrides,
    ))

    # --- vs-baseline: every score is read relative to the lane BASELINE (top of the
    # lane scoreboard). Written AFTER the scoreboard step so scores.csv exists.
    # Warn-only: a missing baseline/scores must not fail an otherwise-good eval. ---
    if getattr(args, "vs_baseline", False):
        from eval.baseline import write_vs_baseline
        try:
            write_vs_baseline(output_dir, lane_name, run_label=getattr(args, "run_label", ""))
        except SystemExit as exc:
            LOG.warning("--vs-baseline skipped: %s", exc)
