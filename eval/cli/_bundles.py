"""Bundle helpers shared by ``predict`` and ``prepare``.

A bundle is the truth-aware input the manual prediction path reads. This module
resolves where the bundles are (``_resolve_predict_input_root``), checks them
before prediction, and refuses to build them inside a multi-task launch, because
bundle preparation is inherently serial.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from eval.cli._common import LOG


def _predict_bundle_pairs(predict_cfg: dict) -> list:
    """Return bundle_pairs from predict config as a list."""
    bundle_pairs_raw = predict_cfg.get("bundle_pairs", [])
    if isinstance(bundle_pairs_raw, str):
        return [bp.strip() for bp in bundle_pairs_raw.split(",") if bp.strip()]
    return list(bundle_pairs_raw)


SERIAL_PREPARE_RANK_ENV_VARS = (
    "SLURM_PROCID",
    "PMI_RANK",
    "PMIX_RANK",
    "OMPI_COMM_WORLD_RANK",
    "MV2_COMM_WORLD_RANK",
    "RANK",
    "LOCAL_RANK",
)


def _distributed_rank_context_vars() -> dict[str, str]:
    return {
        key: value
        for key in SERIAL_PREPARE_RANK_ENV_VARS
        if (value := os.environ.get(key)) not in (None, "")
    }


# C8(iii): world-size env vars. A rank env var being set is harmless when the
# world is a single task (ntasks==1). We only refuse when there is genuinely
# more than one parallel task AND this is not rank 0.
SERIAL_PREPARE_WORLD_SIZE_ENV_VARS = (
    "SLURM_NTASKS",
    "SLURM_STEP_NUM_TASKS",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "WORLD_SIZE",
)


def _max_declared_world_size() -> int:
    """Largest world-size hint across known launchers (0 if none declared)."""
    sizes = [0]
    for key in SERIAL_PREPARE_WORLD_SIZE_ENV_VARS:
        value = os.environ.get(key)
        if value:
            try:
                sizes.append(int(value))
            except ValueError:
                pass
    return max(sizes)


def _all_ranks_zero(rank_vars: dict[str, str]) -> bool:
    """True if every declared rank var is rank 0."""
    for value in rank_vars.values():
        try:
            if int(value) != 0:
                return False
        except ValueError:
            return False
    return True


def _assert_serial_prepare_context() -> None:
    rank_vars = _distributed_rank_context_vars()
    if not rank_vars:
        return
    # C8(iii): allow when the launcher reports a single task (ntasks==1), or when
    # every rank var is 0 and no launcher declares more than one task. Bundle
    # prepare is inherently serial; a 1-task srun/rank-0 context is fine.
    world_size = _max_declared_world_size()
    if world_size <= 1 and _all_ranks_zero(rank_vars):
        return
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(rank_vars.items()))
    raise SystemExit(
        "Refusing serial bundle preparation inside a multi-task distributed "
        f"context ({rendered}; world_size={world_size}). Run "
        "`python -m eval.cli prepare` once with a single task (ntasks==1) or "
        "outside `srun`, then run prediction with `--bundle-dir <prepared-bundles>`."
    )


def _verify_predict_input_bundles(lane_config: dict, input_root: str) -> None:
    """Fail before prediction when a prepare lane is pointed at bad bundles."""
    if not lane_config.get("prepare"):
        return
    from eval.prepare.builder import verify_bundles

    predict_cfg = lane_config["predict"]
    verify_bundles(
        lane_config,
        Path(input_root),
        dates=list(predict_cfg.get("dates", [])),
        steps=[int(s) for s in predict_cfg.get("steps", [])],
        members=[int(m) for m in predict_cfg.get("members", [])],
        bundle_pairs=_predict_bundle_pairs(predict_cfg),
    )


def _resolve_predict_input_root(
    args: argparse.Namespace,
    lane_config: dict,
    host_config: dict,
    output_dir: Path,
    *,
    prepare_bundles: bool,
    allow_host_fallback: bool = True,
) -> str:
    """Resolve the prediction input_root and optionally build truth-aware bundles."""
    predict_cfg = lane_config["predict"]
    source_grib_root = getattr(args, "source_grib_root", None) or ""
    bundle_dir_arg = getattr(args, "bundle_dir", None)

    if lane_config.get("prepare") and source_grib_root:
        bundle_dir = Path(bundle_dir_arg) if bundle_dir_arg else output_dir / "bundles"
        if prepare_bundles:
            _assert_serial_prepare_context()
            from eval.prepare.builder import build_bundles

            LOG.info("=== Phase 0: Bundle preparation ===")
            build_bundles(
                lane_config=lane_config,
                bundle_dir=bundle_dir,
                source_grib_root=source_grib_root,
                dates=list(predict_cfg.get("dates", [])),
                steps=[int(s) for s in predict_cfg.get("steps", [])],
                members=[int(m) for m in predict_cfg.get("members", [])],
                bundle_pairs=_predict_bundle_pairs(predict_cfg),
                verification_path=output_dir / "bundle_build_verification.json",
            )
        return str(bundle_dir)

    if bundle_dir_arg:
        # Use pre-built bundles as input_root; skip rebuild.
        return str(bundle_dir_arg)

    # Resolve input_root: lane config takes precedence over host DATA_DIR.
    input_root = predict_cfg.get("input_root", "")
    if input_root:
        return str(input_root)

    if not allow_host_fallback:
        return ""

    env_setup = host_config.get("environment_setup", {})
    exports = env_setup.get("exports", {})
    return str(exports.get("DATA_DIR", ""))
