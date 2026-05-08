"""PrepML prediction backend for eval.cli.

Orchestrates: checkpoint metadata loading, expver resolution, PrepML config
generation, sbatch launch, MARS retrieval, and predictions_*.nc assembly.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

PREPML_BIN = "/usr/local/apps/prepml/0.99/bin/prepml"

# Regex to parse bundle filenames and extract the template pattern
_BUNDLE_RE = re.compile(
    r"^(.+)_date(\d{8})_time\d{4}_mem(\d{2,3})_step(\d{3})h_input_bundle\.nc$"
)


def _discover_bundle_template(bundle_dir: Path) -> str:
    """Auto-discover bundle filename template from first .nc file in directory."""
    for f in sorted(bundle_dir.glob("*_input_bundle.nc")):
        m = _BUNDLE_RE.match(f.name)
        if m:
            prefix = m.group(1)
            tpl = f"{prefix}_date{{date}}_time0000_mem{{member:02d}}_step{{step:03d}}h_input_bundle.nc"
            LOG.info("Auto-discovered bundle template: %s", tpl)
            return tpl
    raise FileNotFoundError(f"No input bundle files found in {bundle_dir}")


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

    # Push config to ecFlow. --force skips the interactive overwrite prompt
    # for debug expvers that already have data.
    result = subprocess.run(
        [PREPML_BIN, "--force", "inference", str(prepml_config_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"prepml inference failed: {result.stderr.strip()}"
        )
    LOG.info("PrepML config pushed to ecFlow. stdout:\n%s", result.stdout.strip())


def _wait_for_prepml(
    expver: str,
    timeout: int = 43200,
) -> None:
    """Wait for PrepML ecFlow suite to complete using the prepml Python API.

    Uses prepml.utils.ecflow_client.EcflowClient.wait() which polls the
    ecFlow server directly — no subprocess needed.

    Args:
        expver: experiment version to monitor
        timeout: max seconds to wait (default 12 hours)
    """
    import sys
    from getpass import getuser

    # Import from prepml's own Python environment
    prepml_python_path = "/usr/local/apps/prepml/0.99/env/lib/python3.11/site-packages"
    if prepml_python_path not in sys.path:
        sys.path.insert(0, prepml_python_path)

    from prepml.utils.ecflow_client import EcflowClient

    import time as _time

    owner = getuser()
    client = EcflowClient(owner, expver, verbose=False)

    # Wait for ecFlow node to appear (may not exist immediately after push)
    elapsed = 0
    while elapsed < timeout:
        state = client.state()
        LOG.info("PrepML ecFlow state for expver=%s: %s (elapsed=%ds)", expver, state, elapsed)

        if state == "complete":
            LOG.info("PrepML suite completed for expver=%s", expver)
            return
        if state == "aborted":
            raise RuntimeError(
                f"PrepML suite aborted for expver={expver}. "
                f"Check ecFlow logs at ~/prepml/{expver}/"
            )
        if state is not None and state not in ("unknown",):
            # Node exists and is running — switch to ecFlow client polling
            break

        _time.sleep(10)
        elapsed += 10

    LOG.info("Waiting for PrepML suite to complete (timeout=%ds)...", timeout - elapsed)
    remaining = timeout - elapsed
    poll_elapsed = 0
    while poll_elapsed < remaining:
        state = client.state()
        LOG.info("PrepML status: %s (elapsed=%ds)", state, elapsed + poll_elapsed)

        if state == "complete":
            LOG.info("PrepML suite completed for expver=%s", expver)
            return
        if state == "aborted":
            raise RuntimeError(
                f"PrepML suite aborted for expver={expver}. "
                f"Check ecFlow logs at ~/prepml/{expver}/"
            )

        _time.sleep(30)
        poll_elapsed += 30

    raise RuntimeError(
        f"PrepML suite timed out after {timeout}s for expver={expver}."
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


def _generate_retrieve_script(
    retrieve_config_path: Path,
    output_dir: Path,
    host_config: dict,
) -> Path:
    """Generate sbatch script for MARS retrieval + bundle assembly.

    The retrieval needs ~16-32G memory due to earthkit's grid expansion.
    """
    venv = host_config.get("environment_setup", {}).get("venv_activate", "")
    code_root = host_config.get("code_root", "/home/ecm5702/dev/downscaling-tools")
    script = output_dir / "retrieve_predictions.sh"
    script.write_text(f"""#!/bin/bash
#SBATCH --job-name=prepml_retrieve
#SBATCH --output={output_dir}/retrieve_%j.out
#SBATCH --error={output_dir}/retrieve_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --qos=nf

set -euo pipefail
source {venv}
cd {code_root}

python -m eval.predict.prepml --retrieve {retrieve_config_path}
""")
    script.chmod(0o755)
    return script


def run_retrieval(retrieve_config_path: str) -> None:
    """Standalone entry point for MARS retrieval, called from the sbatch job.

    Reads retrieve_config.json and runs assemble_predictions_file for each
    date/step pair.
    """
    from eval.predict.mars_retrieve import assemble_predictions_file

    config_path = Path(retrieve_config_path)
    config = json.loads(config_path.read_text())

    expver = config["expver"]
    dates = config["dates"]
    steps = config["steps"]
    members = config["members"]
    output_mars = config["output_mars"]
    weather_states = config["weather_states"]
    bundle_dir = config["bundle_dir"]
    bundle_filename_tpl = config["bundle_filename_tpl"]
    predictions_dir = Path(config["predictions_dir"])
    predictions_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[tuple[str, int, int, str]] = []
    total = len(dates) * len(steps)
    done = 0

    for date in dates:
        for step in steps:
            out_path = assemble_predictions_file(
                expver=expver,
                date=date,
                step=step,
                members=members,
                output_mars=output_mars,
                weather_states=weather_states,
                bundle_dir=bundle_dir,
                bundle_filename_tpl=bundle_filename_tpl,
                output_dir=predictions_dir,
            )
            done += 1
            LOG.info("[%d/%d] %s", done, total, out_path)
            for member in members:
                manifest_rows.append((date, step, member, str(out_path)))

    manifest_path = predictions_dir / "predictions_manifest.csv"
    _write_manifest(manifest_path, manifest_rows)
    LOG.info("Manifest written to %s", manifest_path)
    LOG.info("Retrieval complete: %d files in %s", done, predictions_dir)


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
    #    This runs as a batch job (needs ~16G for earthkit MARS retrieval).
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    dates = predict_cfg["dates"]
    steps = predict_cfg["steps"]
    members = predict_cfg["members"]
    output_mars = prepml_cfg["output"]
    bundle_dir = predict_cfg.get("input_root", "")
    bundle_filename_tpl = lane_config.get("prepare", {}).get("bundle_filename_tpl", "")
    if not bundle_filename_tpl and bundle_dir:
        bundle_filename_tpl = _discover_bundle_template(Path(bundle_dir))

    retrieve_config = {
        "expver": resolved_expver,
        "dates": dates,
        "steps": steps,
        "members": members,
        "output_mars": output_mars,
        "weather_states": weather_states,
        "bundle_dir": str(bundle_dir),
        "bundle_filename_tpl": bundle_filename_tpl,
        "predictions_dir": str(predictions_dir),
    }
    retrieve_config_path = output_dir / "retrieve_config.json"
    retrieve_config_path.write_text(json.dumps(retrieve_config, indent=2) + "\n")

    retrieve_script = _generate_retrieve_script(
        retrieve_config_path, output_dir, host_config,
    )
    LOG.info("Submitting MARS retrieval batch job: %s", retrieve_script)
    result = subprocess.run(
        ["sbatch", "--wait", str(retrieve_script)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MARS retrieval batch job failed (exit {result.returncode}). "
            f"Check logs at {output_dir}/retrieve_*.out"
        )
    LOG.info("MARS retrieval batch job completed")

    # 6. Write provenance
    _write_provenance(
        output_dir, resolved_expver, prepml_config_path,
        checkpoint, weather_states,
    )
    LOG.info("PrepML predict complete. Predictions in %s", predictions_dir)


if __name__ == "__main__":
    import argparse as _ap

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    _parser = _ap.ArgumentParser()
    _parser.add_argument("--retrieve", required=True, help="Path to retrieve_config.json")
    _args = _parser.parse_args()
    run_retrieval(_args.retrieve)
