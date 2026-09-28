"""Pieces shared by every ``eval.cli`` command.

Holds the ``Command`` record that each command module publishes, the argument
groups several commands have in common, the small comma-separated-list parsers,
and the effective-config helpers that write ``effective_config.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from eval.evaluators import registry as evaluator_registry

# One logger name for the whole package, so log lines read "eval.cli: ..." as
# they did when the CLI was a single module.
LOG = logging.getLogger("eval.cli")

# Every evaluator this CLI can run. Derived from the one registry
# (eval/evaluators/registry.py); retired evaluators are not in it.
ALL_EVALUATORS = evaluator_registry.runnable_names()

DEFAULT_HOST = "atos_ac"

# Order and titles of the command groups in the top-level ``--help``.
GROUP_TITLES = {
    "discovery": "Discovery (what can be evaluated, and how)",
    "pipeline": "Pipeline (predict, evaluate, rank)",
    "comparison": "Comparison across runs",
    "tc_tracks": "Tropical cyclone tracks",
    "figures": "Figures and videos",
    "maintenance": "Maintenance",
}


@dataclass(frozen=True)
class Command:
    """One ``python -m eval.cli <name>`` subcommand.

    ``register`` adds the subparser and returns it. ``run`` does the work: for a
    command with ``needs_lane`` it is called as ``run(args, session)`` after the
    lane and host configuration were resolved (see ``eval.cli._session``);
    otherwise it is called as ``run(args)`` and reads no lane or host file.
    """

    name: str
    group: str
    summary: str
    register: Callable[[Any], argparse.ArgumentParser]
    run: Callable[..., Any]
    needs_lane: bool = False


# ---------------------------------------------------------------------------
# Argument groups shared by several commands
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the arguments every lane-based command takes: --lane, --host, --dry-run."""
    parser.add_argument(
        "--lane", required=True,
        help="Lane name, the stem of a YAML file in eval/config/lanes/.",
    )
    parser.add_argument(
        "--host", default=None,
        help=f"Host config name in eval/config/hosts/ (default: the lane's host for this command, else {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print the resolved configuration as JSON and exit without running anything.",
    )


def add_lane_override_args(parser: argparse.ArgumentParser) -> None:
    """Add lane-overridable args.  All use default=None for precedence detection."""
    parser.add_argument(
        "--members", default=None,
        help="Comma-separated member indices, e.g. 1,2,3. Overrides predict.members of the lane.",
    )
    parser.add_argument(
        "--steps", default=None,
        help="Comma-separated forecast steps, e.g. 24,48. Overrides predict.steps of the lane.",
    )
    parser.add_argument(
        "--dates", default=None,
        help="Comma-separated dates as YYYYMMDD, e.g. 20230826,20230827. Overrides predict.dates of the lane.",
    )
    parser.add_argument(
        "--weather-states", default=None,
        help=(
            "Comma-separated weather_state names, e.g. 10u,2t,z_500. Overrides predict.weather_states "
            "of the lane. With --mode prepml it is the highest-priority source; manual mode still "
            "resolves the states from the checkpoint output."
        ),
    )


def add_prepare_args(parser: argparse.ArgumentParser) -> None:
    """Add truth-aware bundle-building args."""
    parser.add_argument(
        "--source-grib-root", default=None,
        help="Root directory of the source GRIB files, used to build truth-aware bundles.",
    )
    parser.add_argument(
        "--bundle-dir", default=None,
        help=(
            "Bundle directory. With --source-grib-root, where new bundles are written. "
            "Without it, an existing bundle directory used as input_root (no rebuild). "
            "Default: <output-dir>/bundles."
        ),
    )
    parser.add_argument(
        "--num-gpus-per-model", type=int, default=None,
        help="GPUs per model replica. Overrides predict.num_gpus_per_model of the lane.",
    )
    parser.add_argument(
        "--num-chunks", type=int, default=None,
        help=(
            "Set ANEMOI_INFERENCE_NUM_CHUNKS and its _PROCESSOR and _MAPPER variants in the "
            "inference environment, which chunks attention to fit on fewer GPUs. "
            "Falls back to predict.env of the lane."
        ),
    )


def add_prepml_args(parser: argparse.ArgumentParser) -> None:
    """Add PrepML-specific args to run and predict subcommands."""
    parser.add_argument(
        "--mode", choices=["manual", "prepml"], default="manual",
        help="Prediction backend: manual (bundle-based) or prepml (MARS/FDB). Default: manual.",
    )
    parser.add_argument(
        "--expver", default=None,
        help="PrepML experiment version. In prepml mode, defaults to the lane's debug expver.",
    )
    parser.add_argument(
        "--prepml-runner", default=None,
        help="PrepML runner or venv path, overriding the lane config.",
    )


# ---------------------------------------------------------------------------
# Small parsers and lane overrides
# ---------------------------------------------------------------------------

def _parse_csv_or_none(raw: str | None) -> list[str] | None:
    """Split a comma-separated option into a list, or None when it was not given."""
    if raw is None:
        return None
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    """Parse comma-separated integers, sorted ascending."""
    return sorted(int(tok.strip()) for tok in raw.split(",") if tok.strip())


def _parse_str_csv(raw: str) -> list[str]:
    """Parse comma-separated strings, preserving order."""
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _build_lane_overrides(args: argparse.Namespace) -> dict:
    """Build overrides dict from CLI args that are not None."""
    predict_overrides: dict = {}
    if getattr(args, "members", None) is not None:
        predict_overrides["members"] = _parse_int_csv(args.members)
    if getattr(args, "steps", None) is not None:
        predict_overrides["steps"] = _parse_int_csv(args.steps)
    if getattr(args, "dates", None) is not None:
        predict_overrides["dates"] = _parse_str_csv(args.dates)
    if getattr(args, "weather_states", None) is not None:
        predict_overrides["weather_states"] = _parse_str_csv(args.weather_states)
    if getattr(args, "num_gpus_per_model", None) is not None:
        predict_overrides["num_gpus_per_model"] = int(args.num_gpus_per_model)
    # Note: --num-chunks is NOT propagated here. The loader's _deep_merge is only
    # shallow at the second level, so injecting {"env": {...}} here would clobber
    # the lane YAML's full predict.env block. Instead, --num-chunks is applied
    # directly to lane_config["predict"]["env"] after load_lane returns.
    if predict_overrides:
        return {"predict": predict_overrides}
    return {}


# ---------------------------------------------------------------------------
# Effective config (what a run actually used)
# ---------------------------------------------------------------------------

def _get_git_commit() -> str:
    """Return current git commit hash, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _resolve_output_dir(host_config: dict, lane_name: str) -> Path:
    """Build output directory: <scratch_root>/eval/<lane>/run_<YYYYMMDDTHHMMSS>/"""
    scratch_root = Path(host_config["scratch_root"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return scratch_root / "eval" / lane_name / f"run_{timestamp}"


def _config_file_paths(lane_name: str, host_name: str) -> dict:
    """Return paths to the YAML config files that were loaded."""
    config_dir = Path(__file__).resolve().parents[1] / "config"
    return {
        "lane": str(config_dir / "lanes" / f"{lane_name}.yaml"),
        "host": str(config_dir / "hosts" / f"{host_name}.yaml"),
    }


def _build_effective_config(
    args: argparse.Namespace,
    lane_config: dict,
    host_config: dict,
    lane_name: str,
    host_name: str,
    overrides: dict,
    evaluators: list[str],
    output_dir: Path,
) -> dict:
    """Build the effective config dict for emission."""
    code_root = host_config.get("code_root", "unknown")
    return {
        "lane": lane_name,
        "host": host_name,
        "checkpoint": getattr(args, "checkpoint", None),
        "predictions_dir": getattr(args, "predictions_dir", None),
        "eval_dir": getattr(args, "eval_dir", None),
        "resolved": lane_config,
        "overrides": overrides,
        "cli_args": sys.argv[1:],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "code_root": code_root,
        "config_file_paths": _config_file_paths(lane_name, host_name),
        "output_dir": str(output_dir),
        "evaluators": evaluators,
        "evaluators_run": [],
        "mode": getattr(args, "mode", "manual"),
        "expver": getattr(args, "expver", None),
    }


def _write_effective_config(config: dict, output_dir: Path) -> Path:
    """Write effective_config.json to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "effective_config.json"
    path.write_text(json.dumps(config, indent=2, default=str) + "\n")
    return path


def _update_effective_config_completion(
    output_dir: Path, evaluators_run: list[str],
) -> None:
    """Update effective_config.json with completion info."""
    path = output_dir / "effective_config.json"
    if path.exists():
        config = json.loads(path.read_text())
    else:
        config = {}
    config["evaluators_run"] = evaluators_run
    config["completion_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(config, indent=2, default=str) + "\n")
