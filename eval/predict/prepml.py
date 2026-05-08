"""PrepML prediction backend for eval.cli.

Orchestrates: checkpoint metadata loading, expver resolution, PrepML config
generation, sbatch launch, MARS retrieval, and predictions_*.nc assembly.
"""
from __future__ import annotations

import csv
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

PREPML_BIN = "/usr/local/apps/prepml/0.99/bin/prepml"


def _to_plain_container(value: Any) -> Any:
    """Convert common config container objects to builtin Python containers."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass

    if hasattr(value, "model_dump"):
        try:
            return _to_plain_container(value.model_dump())
        except Exception:
            pass

    if hasattr(value, "dict"):
        try:
            return _to_plain_container(value.dict())
        except Exception:
            pass

    if not isinstance(value, dict):
        try:
            return vars(value)
        except TypeError:
            return value

    return value


def resolve_expver(expver: str | None, lane_config: dict) -> str:
    """Resolve the expver to use for PrepML inference.

    If expver is provided, use it directly. Otherwise pick from the
    debug_expvers pool in the lane config.
    """
    if expver is not None:
        return expver

    prepml = lane_config.get("prepml")
    if not prepml:
        raise ValueError(
            "No 'prepml' section in lane config. "
            "Cannot resolve default expver without prepml.debug_expvers."
        )

    debug_pool = prepml.get("debug_expvers", [])
    if not debug_pool:
        raise ValueError(
            "No debug expvers configured in prepml.debug_expvers. "
            "Either pass --expver explicitly or add debug_expvers to the lane config."
        )

    return debug_pool[0]


def _extract_weather_states_from_checkpoint(checkpoint_path: str) -> list[str]:
    """Extract output weather states from checkpoint metadata.

    Loads checkpoint hyper_parameters config (CPU-only, no GPU needed)
    and extracts the weather state names the model was trained to produce.

    Handles both base checkpoints (dict with hyper_parameters.config) and
    inference checkpoints (serialized AnemoiModelInterface objects). If the
    checkpoint is an inference variant, tries to find the companion base
    checkpoint for metadata extraction.
    """
    import torch
    from pathlib import Path

    ckpt_path = Path(checkpoint_path)

    # Inference checkpoints (inference-*.ckpt) are serialized model objects,
    # not dicts. Try the companion base checkpoint for metadata instead.
    if ckpt_path.name.startswith("inference-"):
        base_name = ckpt_path.name.replace("inference-", "", 1)
        base_path = ckpt_path.parent / base_name
        if base_path.exists():
            LOG.info("Using base checkpoint for metadata: %s", base_path)
            ckpt_path = base_path
        else:
            LOG.warning(
                "Inference checkpoint detected but no companion base checkpoint at %s. "
                "Cannot extract weather states from model object.",
                base_path,
            )
            return []

    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception:
        LOG.warning("Failed to load checkpoint for metadata: %s", ckpt_path, exc_info=True)
        return []

    if not isinstance(ckpt, dict):
        LOG.warning(
            "Checkpoint is not a dict (got %s). Cannot extract weather states.",
            type(ckpt).__name__,
        )
        return []

    config = ckpt.get("hyper_parameters", {}).get("config", {})

    config = _to_plain_container(config)
    if not isinstance(config, dict):
        LOG.warning("Cannot convert checkpoint config to dict (type=%s)", type(config).__name__)
        return []

    data_cfg = config.get("data", {})
    data_cfg = _to_plain_container(data_cfg)
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    output_names = data_cfg.get("forcing", [])
    diagnostic_names = data_cfg.get("diagnostic", []) or []

    if not output_names and not diagnostic_names:
        LOG.warning(
            "Could not extract weather states from checkpoint config. "
            "Will use lane config weather states."
        )
        return []

    return list(output_names) + list(diagnostic_names)


def _launch_prepml(
    prepml_config_path: Path,
    expver: str,
) -> None:
    """Launch PrepML inference by pushing config to ecFlow.

    PrepML is a job orchestrator: `prepml inference` pushes the config to an
    ecFlow server which schedules the actual GPU inference jobs. This returns
    quickly — use _wait_for_prepml() to block until inference completes.
    """
    # Set expver
    result = subprocess.run(
        [PREPML_BIN, "expver", "--set", expver],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"prepml expver --set {expver} failed: {result.stderr.strip()}")
    LOG.info("Set expver to %s", expver)

    # Push config to ecFlow
    result = subprocess.run(
        [PREPML_BIN, "inference", str(prepml_config_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"prepml inference failed: {result.stderr.strip()}"
        )
    LOG.info("PrepML config pushed to ecFlow. stdout:\n%s", result.stdout.strip())


def _wait_for_prepml(
    expver: str,
    poll_interval: int = 60,
    timeout: int = 43200,
) -> None:
    """Poll `prepml status --expver` until the ecFlow suite completes.

    Args:
        expver: experiment version to monitor
        poll_interval: seconds between status checks (default 60)
        timeout: max seconds to wait (default 12 hours)
    """
    import time

    elapsed = 0
    while elapsed < timeout:
        result = subprocess.run(
            [PREPML_BIN, "--quiet", "status", "--expver", expver],
            capture_output=True, text=True,
        )
        status_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        LOG.info("PrepML status (expver=%s, elapsed=%ds): %s", expver, elapsed, status_line)

        if status_line == "complete":
            LOG.info("PrepML suite completed for expver=%s", expver)
            return
        if status_line in ("aborted", "suspended"):
            raise RuntimeError(
                f"PrepML suite {status_line} for expver={expver}. "
                f"Check ecFlow logs at ~/prepml/{expver}/"
            )

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise RuntimeError(
        f"PrepML suite timed out after {timeout}s for expver={expver}. "
        f"Last status: {status_line}"
    )


def _write_provenance(
    output_dir: Path,
    expver: str,
    prepml_config_path: Path,
    checkpoint_path: str,
    weather_states: list[str],
    slurm_job_id: str | None = None,
) -> Path:
    """Write prepml_provenance.json to the output directory."""
    provenance = {
        "mode": "prepml",
        "expver": expver,
        "prepml_config": str(prepml_config_path),
        "checkpoint": checkpoint_path,
        "weather_states": weather_states,
        "slurm_job_id": slurm_job_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = output_dir / "prepml_provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    return path


def _write_manifest(
    manifest_path: Path,
    rows: list[tuple[str, int, int, str]],
) -> None:
    """Write predictions manifest CSV."""
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "step", "member", "predictions_path"])
        writer.writerows(rows)


def prepml_predict(
    *,
    checkpoint: str,
    lane_config: dict,
    host_config: dict,
    output_dir: Path,
    expver: str | None = None,
    runner_override: str | None = None,
) -> None:
    """Run PrepML prediction: generate config, launch, retrieve, assemble.

    This is the entry point called by eval.cli when --mode prepml.
    """
    from eval.predict.mars_retrieve import assemble_predictions_file
    from eval.predict.prepml_config import generate_prepml_config, write_prepml_config

    predict_cfg = lane_config["predict"]
    prepml_cfg = lane_config["prepml"]

    # 1. Resolve expver
    resolved_expver = resolve_expver(expver, lane_config)
    LOG.info("Using expver: %s", resolved_expver)

    # 2. Extract weather states from checkpoint
    weather_states = _extract_weather_states_from_checkpoint(checkpoint)
    if weather_states:
        from eval.predict.mars_retrieve import weather_state_to_mars

        invalid_states: list[str] = []
        for state in weather_states:
            try:
                weather_state_to_mars(state)
            except ValueError:
                invalid_states.append(state)
        if invalid_states:
            LOG.warning(
                "Checkpoint metadata yielded non-MARS weather states %s. "
                "Will use lane config weather states.",
                invalid_states,
            )
            weather_states = []
    if not weather_states:
        weather_states = lane_config.get("spectra", {}).get("fields", [])
        if not weather_states:
            raise ValueError(
                "Cannot determine output weather states. "
                "Neither checkpoint config nor lane spectra.fields provided them."
            )
        LOG.info("Using weather states from lane spectra config: %s", weather_states)
    else:
        LOG.info("Extracted weather states from checkpoint: %s", weather_states)

    # 3. Generate PrepML config
    prepml_config = generate_prepml_config(
        lane_config=lane_config,
        checkpoint_path=checkpoint,
        runner_override=runner_override,
    )
    prepml_config_path = write_prepml_config(
        prepml_config, output_dir / "prepml_config.yaml"
    )
    LOG.info("PrepML config written to %s", prepml_config_path)

    # 4. Launch PrepML (pushes to ecFlow) and wait for completion
    _launch_prepml(prepml_config_path, resolved_expver)
    LOG.info("Waiting for PrepML ecFlow suite to complete...")
    _wait_for_prepml(resolved_expver)

    # 5. Retrieve from MARS and assemble predictions_*.nc
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    dates = predict_cfg["dates"]
    steps = predict_cfg["steps"]
    members = predict_cfg["members"]
    output_mars = prepml_cfg["output"]
    truth_root = prepml_cfg["truth"]["root"]

    manifest_rows: list[tuple[str, int, int, str]] = []
    total = len(dates) * len(steps)
    done = 0

    for date in dates:
        for step in steps:
            out_path = assemble_predictions_file(
                expver=resolved_expver,
                date=date,
                step=step,
                members=members,
                output_mars=output_mars,
                weather_states=weather_states,
                truth_root=truth_root,
                output_dir=predictions_dir,
            )
            done += 1
            LOG.info("[%d/%d] %s", done, total, out_path)
            for member in members:
                manifest_rows.append((date, step, member, str(out_path)))

    # 6. Write manifest and provenance
    manifest_path = predictions_dir / "predictions_manifest.csv"
    _write_manifest(manifest_path, manifest_rows)
    LOG.info("Manifest written to %s", manifest_path)

    _write_provenance(
        output_dir, resolved_expver, prepml_config_path,
        checkpoint, weather_states,
    )
    LOG.info("PrepML predict complete. Predictions in %s", predictions_dir)
