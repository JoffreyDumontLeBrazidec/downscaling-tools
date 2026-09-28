"""Intermediate evaluator — diffusion sampling intermediate states.

Two-stage execution:
  1. Compute (GPU host, default ag-login): generate intermediate-state netCDF
     via plot_intermediate.py 'checkpoint' subcommand with --output-nc-only.
  2. Plot (current host, typically hpc-login): render a consolidated multi-page
     PDF in-process — one page per (region, weather_state), mirroring the
     region_plot evaluator's all_regions_plots.pdf structure.

Both hosts share /scratch/ecm5702 over Lustre, so the netCDF is path-portable.
If no checkpoint is passed in, picks the best one for the lane from the
existing scoreboard JSONs. If a netCDF already exists at the expected path,
the compute stage is skipped (cheap re-plot).
"""
from __future__ import annotations

import json
import logging
import math
import re
import shlex
import socket
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)

DEFAULT_SCOREBOARD_ROOT = Path("/home/ecm5702/scratch/eval")
DEFAULT_CKPT_ROOT = Path("/home/ecm5702/scratch/aifs/checkpoint")
DEFAULT_COMPUTE_HOST = "ag-login"
DEFAULT_COMPUTE_VENV = "/home/ecm5702/dev/.ds-ag/bin/activate"  # ag-login is aarch64
DEFAULT_COMPUTE_PROJECT_ROOT = "/home/ecm5702/dev/downscaling-tools"
DEFAULT_RANKING_METRIC = "spectra_mean_relative_l2"  # lower = better
LANE_REGEX = re.compile(r"o\d+_o\d+")
SCOREBOARD_HASH_RE = re.compile(r"^manual_([0-9a-f]+)_", re.IGNORECASE)


def _resolve_lane(eval_config: dict, predictions_dir: Path) -> str:
    explicit = eval_config.get("lane")
    if explicit:
        return str(explicit)
    for part in (predictions_dir, *predictions_dir.parents):
        m = LANE_REGEX.search(part.name)
        if m:
            return m.group(0)
    raise RuntimeError(
        f"Cannot infer lane from {predictions_dir!s}. "
        "Set lane_config['intermediate']['lane'] explicitly."
    )


def _full_ckpt_dir(prefix: str, ckpt_root: Path) -> str:
    matches = [p.name for p in ckpt_root.iterdir() if p.name.lower().startswith(prefix.lower())]
    matches = [m for m in matches if "_before_ft" not in m]
    if not matches:
        raise RuntimeError(f"No checkpoint dir starting with {prefix!r} under {ckpt_root}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous checkpoint prefix {prefix!r}: {matches}")
    return matches[0]


def pick_best_checkpoint(
    lane: str,
    *,
    scratch_root: Path = DEFAULT_SCOREBOARD_ROOT,
    ckpt_root: Path = DEFAULT_CKPT_ROOT,
    metric: str = DEFAULT_RANKING_METRIC,
) -> str:
    """Return the full checkpoint dir name with best (lowest) `metric` for a lane."""
    candidates: list[tuple[float, str, Path]] = []
    for sb_path in scratch_root.glob("*/scoreboard_metrics.json"):
        try:
            data = json.loads(sb_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sb_lane = data.get("lane")
        if sb_lane:
            if sb_lane != lane:
                continue
        elif lane not in sb_path.parent.name:
            continue
        score = data.get("scalars", {}).get(metric)
        if score is None:
            score = data.get(metric)
        if score is None or (isinstance(score, float) and math.isnan(score)):
            continue
        run_id = data.get("run_id") or sb_path.parent.name
        m = SCOREBOARD_HASH_RE.search(run_id)
        if not m:
            continue
        candidates.append((float(score), m.group(1), sb_path))
    if not candidates:
        raise RuntimeError(
            f"No usable scoreboards for lane {lane!r} under {scratch_root} "
            f"(metric={metric!r}). Pass checkpoint explicitly or populate a scoreboard."
        )
    candidates.sort(key=lambda t: t[0])
    score, prefix, sb_path = candidates[0]
    full_name = _full_ckpt_dir(prefix, ckpt_root)
    LOG.info(
        "Lane %s: picked checkpoint %s (prefix %s, %s=%.6f from %s)",
        lane, full_name, prefix, metric, score, sb_path,
    )
    return full_name


def _build_remote_cmd(
    local_cmd: list[str],
    host: str,
    venv: str,
    project_root: str,
    env_exports: dict | None = None,
    module_loads: list | None = None,
) -> list[str]:
    parts: list[str] = []
    if module_loads:
        for m in module_loads:
            parts.append(f"module load {shlex.quote(m)}")
    if env_exports:
        for k, v in env_exports.items():
            parts.append(f"export {k}={shlex.quote(str(v))}")
    parts.append(f"source {shlex.quote(venv)}")
    parts.append(f"cd {shlex.quote(project_root)}")
    parts.append(" ".join(shlex.quote(c) for c in local_cmd))
    inner = " && ".join(parts)
    return ["ssh", host, "bash", "-l", "-c", inner]


def _load_compute_host_setup(host_name: str) -> tuple[str, str, dict, list]:
    """Read host YAML to get venv, project_root, env_exports, module_loads."""
    try:
        import yaml  # type: ignore
    except ImportError:
        return DEFAULT_COMPUTE_VENV, DEFAULT_COMPUTE_PROJECT_ROOT, {}, []
    candidates = [
        Path(f"/home/ecm5702/dev/downscaling-tools/eval/config/hosts/{host_name}.yaml"),
        Path(f"/home/ecm5702/dev/downscaling-tools/eval/config/hosts/atos_{host_name.split('-')[0]}.yaml"),
    ]
    for p in candidates:
        if p.exists():
            cfg = yaml.safe_load(p.read_text()) or {}
            venv = cfg.get("venv_activate") or (cfg.get("venv", "") + "/bin/activate")
            project_root = cfg.get("code_root", DEFAULT_COMPUTE_PROJECT_ROOT)
            env_setup = cfg.get("environment_setup", {}) or {}
            return venv, project_root, env_setup.get("exports", {}) or {}, env_setup.get("module_loads", []) or []
    return DEFAULT_COMPUTE_VENV, DEFAULT_COMPUTE_PROJECT_ROOT, {}, []


DEFAULT_BUNDLE_GLOBS = {
    "o48_o96": "/home/ecm5702/hpcperm/data/input_data/o48_o96/humberto_20250926_20250930/enfo_o48_0001_date20250928_time0000_mem01_step072h_input_bundle.nc",
    "o96_o320": "/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia/enfo_o96_0001_date20230828_time0000_mem01_step048h_input_bundle.nc",
}


def _resolve_bundle_path(eval_config: dict, lane: str, lane_config: dict) -> Path:
    explicit = eval_config.get("bundle_path") or (lane_config.get("intermediate") or {}).get("bundle_path")
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise RuntimeError(f"Configured intermediate.bundle_path does not exist: {p}")
    glob_pattern = eval_config.get("bundle_glob") or DEFAULT_BUNDLE_GLOBS.get(lane)
    if glob_pattern:
        matches = sorted(Path("/").glob(glob_pattern.lstrip("/"))) if "*" in glob_pattern else [Path(glob_pattern)]
        matches = [m for m in matches if m.exists()]
        if matches:
            return matches[0]
    raise RuntimeError(
        f"input_mode=bundle but no bundle resolvable for lane {lane!r}. "
        "Set lane_config.intermediate.bundle_path or .bundle_glob."
    )


def _existing_nc_path(output_dir: Path, ckpt_name: str) -> Path | None:
    """Return cached NC path if present, else None. Tolerant of CKPT_NAME variations."""
    candidate = output_dir / f"inter_states_{ckpt_name}.nc"
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate
    matches = sorted(output_dir.glob("inter_states_*.nc"))
    matches = [p for p in matches if p.stat().st_size > 0]
    return matches[0] if matches else None


def _resolve_regions(eval_config: dict, lane_config: dict) -> dict:
    """Region-name → [lat_min, lat_max, lon_min, lon_max] dict.

    Mirrors region_plot's behavior: pulls from lane_config['regions']['interesting'].
    """
    explicit = eval_config.get("regions")
    if explicit:
        if isinstance(explicit, dict):
            return explicit
        from eval._backends.region_plotting.plotting.config import KNOWN_REGION_BOXES
        return {name: KNOWN_REGION_BOXES[name] for name in explicit if name in KNOWN_REGION_BOXES}
    lane_regions = (lane_config.get("regions") or {}).get("interesting") or {}
    if lane_regions:
        return dict(lane_regions)
    return {"default": [40.0, 50.0, 0.0, 10.0]}


def _resolve_weather_states(eval_config: dict, available: list[str]) -> list[str]:
    """Filter the requested weather-state list to those present in the dataset."""
    requested = eval_config.get("weather_states")
    if requested is None:
        from eval._backends.region_plotting.plotting.config import DEFAULT_WEATHER_STATES
        requested = DEFAULT_WEATHER_STATES
    available_set = set(available)
    return [w for w in requested if w in available_set]


def _render_pdf_from_nc(
    nc_path: Path,
    pdf_path: Path,
    *,
    lane: str,
    eval_config: dict,
    lane_config: dict,
) -> Path:
    """Read cached NC and render one consolidated multi-page PDF.

    One page per region. Each page is a (n_weather_states × n_cols) grid where
    n_cols = (5 sampling steps from half-max to end) + 1 truth column. Matches
    the region_plot/all_regions_plots.pdf aesthetic.
    """
    import gc
    import xarray as xr
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from eval._backends.plot_intermediate.plot_intermediate import (
        plot_intermediate_region_grid,
        select_steps_from_half,
    )

    # Lazy-load: each region+state selection materializes only its slice
    # (full inter_state is 1.5GB+ for o96_o320 — would OOM on login nodes).
    ds = xr.open_dataset(nc_path, chunks={"weather_state": 1, "sampling_step": 1})
    try:
        regions_dict = _resolve_regions(eval_config, lane_config)
        weather_states = _resolve_weather_states(eval_config, list(ds.weather_state.values))
        if not weather_states:
            raise RuntimeError(
                f"No weather_states from default list present in {nc_path}. "
                f"Dataset has: {list(ds.weather_state.values)[:10]}..."
            )
        n_steps = int(eval_config.get("n_panels_steps", 5))
        sampling_steps = select_steps_from_half(
            list(ds.sampling_step.values), n=n_steps,
        )
        sample_idx = int(eval_config.get("sample", 0))
        member_idx = int(eval_config.get("member", 0))

        LOG.info(
            "intermediate render: %d regions, %d weather_states, steps=%s → %s",
            len(regions_dict), len(weather_states), sampling_steps, pdf_path,
        )
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(pdf_path) as pdf:
            for region_name, region_box in regions_dict.items():
                LOG.info("  page: region=%s box=%s", region_name, region_box)
                fig = plot_intermediate_region_grid(
                    ds=ds,
                    region_name=region_name,
                    region_box=list(region_box),
                    weather_states=weather_states,
                    sampling_steps=sampling_steps,
                    sample=sample_idx,
                    member=member_idx,
                    suptitle_extra=f"lane={lane}  ckpt={nc_path.stem.replace('inter_states_', '')}",
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                gc.collect()
            d = pdf.infodict()
            d["Title"] = f"Intermediate diffusion states — {lane}"
            d["Subject"] = f"Source NC: {nc_path.name}"
    finally:
        ds.close()
    LOG.info("intermediate PDF written: %s", pdf_path)
    return pdf_path


def _resolve_ckpt_name(checkpoint: str | None, lane: str, eval_config: dict, ckpt_root: Path) -> str:
    if checkpoint:
        return Path(str(checkpoint)).name if "/" in str(checkpoint) else str(checkpoint)
    return pick_best_checkpoint(
        lane,
        ckpt_root=ckpt_root,
        metric=eval_config.get("ranking_metric", DEFAULT_RANKING_METRIC),
    )


def _resolve_output_dir(predictions_dir: Path, output_dir) -> Path:
    return Path(output_dir) if output_dir else predictions_dir / "evaluators" / "intermediate"


def _pdf_filename(lane: str, ckpt_name: str) -> str:
    return f"intermediate_{lane}_{ckpt_name}_all_regions.pdf"


def run(
    predictions_dir,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir=None,
    overwrite: bool = False,
    checkpoint: str | None = None,
    run_label: str = "",
    **kwargs,
) -> Path:
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = _resolve_output_dir(predictions_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lane = _resolve_lane(eval_config, predictions_dir)
    ckpt_root = Path(eval_config.get("ckpt_root", DEFAULT_CKPT_ROOT))
    ckpt_name = _resolve_ckpt_name(checkpoint, lane, eval_config, ckpt_root)

    nc_path = output_dir / f"inter_states_{ckpt_name}.nc"
    pdf_path = output_dir / _pdf_filename(lane, ckpt_name)

    cached = _existing_nc_path(output_dir, ckpt_name)
    if cached is not None and not eval_config.get("force_recompute", False):
        LOG.info("intermediate: reusing cached NC %s (skipping ag-login compute)", cached)
        return _render_pdf_from_nc(cached, pdf_path, lane=lane, eval_config=eval_config, lane_config=lane_config).parent

    capture_max = int(eval_config.get("capture_max_steps", 8))
    weather_state_for_compute = eval_config.get("weather_state", "msl")  # arg required by CLI; doesn't affect NC content
    compute_host = str(eval_config.get("compute_host", DEFAULT_COMPUTE_HOST))
    input_mode = str(eval_config.get("input_mode", "checkpoint")).lower()

    if input_mode == "bundle":
        bundle_path = _resolve_bundle_path(eval_config, lane, lane_config)
        capture_steps = eval_config.get("capture_steps", "0,4,8,12,16,20,24,29")
        compute_cmd = [
            "python", "-m", "eval._backends.plot_intermediate.generate_from_bundle",
            "--bundle-nc", str(bundle_path),
            "--out-nc", str(nc_path),
            "--ckpt-ref", ckpt_name,
            "--ckpt-root", str(ckpt_root),
            "--capture-steps", str(capture_steps),
            "--include-init-state",
        ]
    else:
        compute_cmd = [
            "python", "-m", "eval._backends.plot_intermediate.plot_intermediate", "checkpoint",
            "--name-ckpt", ckpt_name,
            "--ckpt-root", str(ckpt_root),
            "--save-intermediate-nc", str(nc_path),
            "--output-nc-only",
            "--weather-state", weather_state_for_compute,
            "--capture-max-steps", str(capture_max),
        ]

    is_local = compute_host in ("local", "", socket.gethostname())
    if is_local:
        full_cmd: list[str] = [sys.executable] + compute_cmd[1:]
    else:
        host_alias = eval_config.get("compute_host_config", compute_host)
        venv, project_root, env_exports, module_loads = _load_compute_host_setup(host_alias)
        venv = eval_config.get("compute_venv", venv)
        project_root = eval_config.get("compute_project_root", project_root)
        env_exports = {**env_exports, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        env_exports.update(eval_config.get("compute_env", {}))
        full_cmd = _build_remote_cmd(
            compute_cmd, compute_host, venv, project_root,
            env_exports=env_exports, module_loads=module_loads,
        )

    LOG.info("intermediate compute (%s): %s", compute_host, " ".join(full_cmd))
    subprocess.run(full_cmd, check=True)

    if not nc_path.exists():
        raise RuntimeError(f"Compute stage finished but netCDF missing: {nc_path}")

    _render_pdf_from_nc(nc_path, pdf_path, lane=lane, eval_config=eval_config, lane_config=lane_config)
    return output_dir


def render_only(
    output_dir,
    lane_config: dict,
    eval_config: dict,
    *,
    checkpoint: str | None = None,
    predictions_dir=None,
) -> Path:
    """Plot-only entry point: read cached NC under output_dir and re-render the PDF.

    Used by plotter.plot() when --plot-only is invoked from eval.cli.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise RuntimeError(f"intermediate output_dir does not exist: {output_dir}")

    if predictions_dir is None:
        predictions_dir = output_dir
    lane = _resolve_lane(eval_config, Path(predictions_dir))
    ckpt_root = Path(eval_config.get("ckpt_root", DEFAULT_CKPT_ROOT))
    try:
        ckpt_name = _resolve_ckpt_name(checkpoint, lane, eval_config, ckpt_root)
    except RuntimeError:
        ckpt_name = "unknown"
    cached = _existing_nc_path(output_dir, ckpt_name)
    if cached is None:
        raise RuntimeError(f"No cached inter_states_*.nc found under {output_dir}")
    actual_ckpt = cached.stem.replace("inter_states_", "")
    pdf_path = output_dir / _pdf_filename(lane, actual_ckpt)
    return _render_pdf_from_nc(cached, pdf_path, lane=lane, eval_config=eval_config, lane_config=lane_config)
