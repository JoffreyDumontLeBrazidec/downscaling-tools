"""PrepML prediction backend for eval.cli.

Orchestrates: checkpoint metadata loading, expver resolution, PrepML config
generation, sbatch launch, MARS retrieval, and predictions_*.nc assembly.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

PREPML_BIN = os.environ.get("EVAL_PREPML_BIN", "/usr/local/apps/prepml/0.99/bin/prepml")  # override to test newer prepml, e.g. /usr/local/apps/prepml/0.134/bin/prepml
ECFLOW_BIN = "/usr/local/apps/ecflow/5.13.0/bin/ecflow_client"
ECFLOW_ENV = {"ECF_HOST": "ecflow-gen-mlx-001", "ECF_PORT": "3141"}
LEDGER_PATH = Path.home() / ".config" / "eval" / "prepml_consumed.jsonl"

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


def ecflow_client(args: list[str], *, check: bool = True, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run an ecflow_client invocation against the prepml ecFlow server."""
    return subprocess.run(
        [ECFLOW_BIN, *args],
        env={**os.environ, **ECFLOW_ENV},
        capture_output=True, text=True, check=check, timeout=timeout,
    )


def discover_owner(user: str | None = None) -> str:
    """Resolve the ecFlow owner string for this user (e.g. ecm5702_joffrey_dumont_le_brazidec).

    Uses `ecflow_client --suites` to list all top-level suite names and picks
    the one starting with `<user>_` (or equal to `<user>`). Falls back to the
    raw user id if discovery fails.
    """
    if user is None:
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
    try:
        result = ecflow_client(["--suites"], check=False)
    except (OSError, subprocess.TimeoutExpired):
        return user
    if result.returncode != 0:
        return user
    candidates = result.stdout.split()
    for name in candidates:
        if name.startswith(f"{user}_"):
            return name
    for name in candidates:
        if name == user:
            return name
    return user


_HEX_HASH_RE = re.compile(r"^[0-9a-f]{32}$")


def _short_checkpoint_id(checkpoint: str | Path) -> str:
    """Extract an 8-char id from an MLflow-style checkpoint path.

    For .../checkpoint/<32-hex>/<file>.ckpt return the first 8 hex chars.
    Falls back to the parent dir's first 8 chars, then the file stem's first 8.
    """
    p = Path(str(checkpoint))
    parent = p.parent.name
    if _HEX_HASH_RE.match(parent):
        return parent[:8]
    if parent:
        return parent[:8]
    return p.stem[:8]


def _ledger_path() -> Path:
    """Return the configured ledger path, creating its parent dir if needed."""
    path = LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_ledger(path: Path | None = None) -> list[dict[str, Any]]:
    """Return every record in the prepml-consumed JSONL ledger (oldest first)."""
    target = path or _ledger_path()
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            LOG.warning("Skipping malformed ledger line: %s", line[:120])
    return records


def record_consumed(
    *,
    expver: str,
    lane: str,
    checkpoint: str,
    run_dir: Path | str,
    dates: list[str],
    steps: list[int],
    members: list[int],
    owner: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Append a record to the prepml-consumed ledger and return it.

    Failures to write are logged at WARNING and swallowed — never abort the
    surrounding prepml run because the ledger could not be updated.
    """
    record = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "expver": expver,
        "owner": owner or discover_owner(),
        "lane": lane,
        "checkpoint": str(checkpoint),
        "checkpoint_short": _short_checkpoint_id(checkpoint),
        "run_dir": str(run_dir),
        "dates": list(dates),
        "steps": list(steps),
        "members": list(members),
        "cleaned_ts_utc": None,
    }
    target = path or _ledger_path()
    try:
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        LOG.info("Recorded prepml expver %s in ledger %s", expver, target)
    except OSError as exc:
        LOG.warning("Could not append to prepml ledger %s: %s", target, exc)
    return record


def mark_cleaned(
    expver: str,
    *,
    owner: str | None = None,
    ts_utc: str | None = None,
    path: Path | None = None,
) -> int:
    """Set cleaned_ts_utc on every ledger record matching (expver, owner).

    Returns the number of records updated. Rewrites the file in place.
    """
    target = path or _ledger_path()
    if not target.exists():
        return 0
    records = read_ledger(target)
    stamp = ts_utc or datetime.now(timezone.utc).isoformat()
    updated = 0
    for rec in records:
        if rec.get("expver") != expver:
            continue
        if owner is not None and rec.get("owner") != owner:
            continue
        if rec.get("cleaned_ts_utc"):
            continue
        rec["cleaned_ts_utc"] = stamp
        updated += 1
    if updated:
        tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        tmp.replace(target)
    return updated


def _resolve_prepml_weather_states(
    *,
    checkpoint: str,
    lane_config: dict,
) -> list[str]:
    """Resolve the canonical weather_states list for a PrepML run.

    Priority chain (first non-empty wins):
      1. explicit `predict.weather_states` in lane YAML
      2. bundle `target_hres_*` discovery — matches manual's evaluation surface
      3. checkpoint `hyper_parameters.config.data` (forcing + diagnostic)
      4. lane `spectra.fields` — legacy last-resort fallback

    Results from (2)+(3)+(4) are validated against `weather_state_to_mars` so
    PrepML never tries to retrieve params it doesn't understand.
    """
    from eval.predict.mars_retrieve import (
        discover_weather_states_from_bundle,
        weather_state_to_mars,
    )

    def _validate(states: list[str], source: str) -> list[str]:
        kept: list[str] = []
        invalid: list[str] = []
        for state in states:
            try:
                weather_state_to_mars(state)
            except ValueError:
                invalid.append(state)
                continue
            kept.append(state)
        if invalid:
            LOG.warning(
                "%s yielded non-MARS weather states %s; dropping them.",
                source, invalid,
            )
        return kept

    # (1) explicit lane override
    explicit = list(lane_config.get("predict", {}).get("weather_states") or [])
    if explicit:
        kept = _validate(explicit, "lane predict.weather_states override")
        if kept:
            LOG.info("Using explicit predict.weather_states from lane YAML: %s", kept)
            return kept

    # (2) bundle discovery, intersected with what the model actually emits.
    # Bundle target_hres_* lists the truth/analysis vars (which the model may not all
    # produce — e.g., the o48_o96 26d63c37 ckpt omits `sp`). Asking PrepML for a var
    # the model never produced makes the MARS retrieve fail with "Expected N, got N-k".
    bundle_dir = lane_config.get("predict", {}).get("input_root", "") or ""
    if bundle_dir:
        bundle_path = _first_bundle_in_dir(Path(bundle_dir))
        if bundle_path is not None:
            try:
                states = discover_weather_states_from_bundle(bundle_path)
            except Exception:
                LOG.warning("Bundle weather_states discovery failed for %s", bundle_path, exc_info=True)
                states = []
            kept = _validate(states, f"bundle {bundle_path.name}")
            if kept:
                model_outputs = _model_output_states_from_checkpoint(checkpoint)
                if model_outputs:
                    model_set = set(model_outputs)
                    intersected = [s for s in kept if s in model_set]
                    dropped = sorted(set(kept) - model_set)
                    if dropped:
                        LOG.info(
                            "Dropping bundle weather_states not produced by checkpoint: %s",
                            dropped,
                        )
                    kept = intersected
                if kept:
                    LOG.info("Using weather_states discovered from bundle %s: %s", bundle_path.name, kept)
                    return kept

    # (3) checkpoint metadata
    ckpt_states = _extract_weather_states_from_checkpoint(checkpoint)
    if ckpt_states:
        kept = _validate(ckpt_states, "checkpoint metadata")
        if kept:
            LOG.info("Using weather_states from checkpoint metadata: %s", kept)
            return kept

    # (4) spectra.fields legacy fallback
    spectra_fields = list(lane_config.get("spectra", {}).get("fields") or [])
    if spectra_fields:
        kept = _validate(spectra_fields, "lane spectra.fields fallback")
        if kept:
            LOG.warning(
                "Falling back to lane spectra.fields for weather_states (%s). "
                "Consider setting predict.weather_states explicitly or rebuilding "
                "bundles so target_hres_* coverage is discoverable.",
                kept,
            )
            return kept

    raise ValueError(
        "Cannot determine output weather states. "
        "None of predict.weather_states / bundle target_hres_* / checkpoint metadata / "
        "spectra.fields yielded a usable list."
    )


def _first_bundle_in_dir(bundle_dir: Path) -> Path | None:
    """Return the first `*_input_bundle.nc` file in `bundle_dir`, or None."""
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        return None
    for f in sorted(bundle_dir.glob("*_input_bundle.nc")):
        if f.is_file():
            return f
    return None


def _model_output_states_from_checkpoint(checkpoint_path: str) -> list[str]:
    """Read `data_indices.model.output.name_to_index` from the checkpoint.

    Returns the actual list of weather_state names the model emits, or an empty
    list if it can't be read (e.g. inference-* checkpoint with no companion base,
    older checkpoints without data_indices, etc.). Caller intersects this with
    bundle-discovery to drop truth-only vars the model never produces.

    CPU-only; loads weights_only=False to access pickled IndexCollection.
    """
    import torch

    ckpt_path = Path(checkpoint_path)
    if ckpt_path.name.startswith("inference-"):
        base_path = ckpt_path.parent / ckpt_path.name.replace("inference-", "", 1)
        if base_path.exists():
            ckpt_path = base_path
        else:
            return []

    if not ckpt_path.exists():
        return []

    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception:
        LOG.warning("Failed to load checkpoint for model-output index: %s", ckpt_path, exc_info=True)
        return []

    if not isinstance(ckpt, dict):
        return []
    di = ckpt.get("hyper_parameters", {}).get("data_indices")
    if di is None:
        return []
    try:
        return list(di.model.output.name_to_index.keys())
    except AttributeError:
        return []


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
    abort_grace_seconds: int = 90,
) -> None:
    """Wait for PrepML ecFlow suite to complete using the prepml Python API.

    Uses prepml.utils.ecflow_client.EcflowClient.wait() which polls the
    ecFlow server directly — no subprocess needed.

    Args:
        expver: experiment version to monitor
        timeout: max seconds to wait (default 12 hours)
    """
    import time as _time

    elapsed = 0
    while elapsed < timeout:
        # Use prepml CLI for status — the Python EcflowClient doesn't work
        # reliably from non-prepml venvs.
        result = subprocess.run(
            [PREPML_BIN, "--quiet", "status", "--expver", expver],
            capture_output=True, text=True,
        )
        # Extract the last word from output (e.g., "active", "complete", "aborted")
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        state = lines[-1] if lines else None
        # prepml status outputs "family j5d7 # state:active ..." in verbose
        # or just "active" in quiet mode
        if state and "state:" in state:
            import re as _re
            m = _re.search(r"state:(\w+)", state)
            state = m.group(1) if m else state

        LOG.info("PrepML status (expver=%s, elapsed=%ds): %s", expver, elapsed, state)

        if state == "complete":
            LOG.info("PrepML suite completed for expver=%s", expver)
            return
        if state == "aborted":
            if elapsed < abort_grace_seconds:
                LOG.warning(
                    "PrepML status is aborted for expver=%s at elapsed=%ds; "
                    "continuing briefly in case this is stale state from a reused debug expver.",
                    expver,
                    elapsed,
                )
                _time.sleep(30)
                elapsed += 30
                continue
            raise RuntimeError(
                f"PrepML suite aborted for expver={expver}. "
                f"Check ecFlow logs at ~/prepml/{expver}/"
            )

        _time.sleep(30)
        elapsed += 30

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
    checkpoint_id = config.get("checkpoint_id") or expver

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
                checkpoint_id=checkpoint_id,
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
    lane: str = "",
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

    # 2. Resolve weather states. Priority:
    #    (a) explicit predict.weather_states in lane YAML — operator override
    #    (b) bundle target_hres_* discovery — same evaluation surface manual sees
    #    (c) checkpoint metadata — only when bundles aren't reachable
    #    (d) lane spectra.fields — last-resort fallback for legacy lanes
    weather_states = _resolve_prepml_weather_states(
        checkpoint=checkpoint,
        lane_config=lane_config,
    )

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
    # Record the consumed expver in the local ledger so `eval.cli prepml-cleanup`
    # can find it later. Record before waiting so a hung/aborted suite is still
    # tracked and cleanable.
    try:
        record_consumed(
            expver=resolved_expver,
            lane=lane or lane_config.get("name") or lane_config.get("lane") or "",
            checkpoint=checkpoint,
            run_dir=output_dir,
            dates=list(predict_cfg.get("dates", [])),
            steps=list(predict_cfg.get("steps", [])),
            members=list(predict_cfg.get("members", [])),
        )
    except Exception:
        LOG.warning("Failed to record prepml expver in ledger", exc_info=True)
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
        "checkpoint_id": Path(checkpoint).stem,
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
