from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from eval.jobs.checkpoint_profile import infer_lane_from_config

from manual_inference.checkpoints import (
    ObjectFromCheckpointLoader,
    adapt_config_hpc,
    get_checkpoint,
    instantiate_config,
)
from manual_inference.prediction.predict import (
    _get_parallel_info,
    _init_model_comm_group,
    _resolve_device,
    _rewrite_dataset_paths_in_place,
)

from .sigma_evaluator import SigmaEvaluator
from .sigmas import sigmas


O1280_FAMILY_LANES = {"o320_o1280", "o1280_o2560"}
O1280_FAMILY_NUM_GPUS_PER_MODEL = 4
RESIDUAL_STATISTICS_FALLBACK_BY_LANE = {
    "o1280_o2560": "o2560_dict_6_72.npy",
}


def _normalize_cfg_for_lane_inference(cfg):
    if hasattr(cfg, "model_dump"):
        cfg = cfg.model_dump()
    try:
        from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        pass
    return cfg


def _maybe_fix_missing_residual_statistics(cfg) -> Path | None:
    try:
        residual_dir = getattr(cfg.hardware.paths, "residual_statistics")
        residual_file = getattr(cfg.hardware.files, "residual_statistics")
    except AttributeError:
        return None

    if not residual_dir or not residual_file:
        return None

    current_path = Path(residual_dir) / residual_file
    if current_path.exists():
        return None

    try:
        lane = infer_lane_from_config(_normalize_cfg_for_lane_inference(cfg))
    except Exception:
        return None

    fallback_name = RESIDUAL_STATISTICS_FALLBACK_BY_LANE.get(lane)
    if not fallback_name or fallback_name == residual_file:
        return None

    fallback_path = Path(residual_dir) / fallback_name
    if not fallback_path.exists():
        return None

    cfg.hardware.files.residual_statistics = fallback_name
    return fallback_path


def _destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _distributed_barrier(*, device: str, local_rank: int) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    sync_device_ids = [local_rank] if str(device).startswith("cuda") and torch.cuda.is_available() else None
    dist.barrier(device_ids=sync_device_ids)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sigma evaluator.")
    parser.add_argument(
        "--name_exp", type=str, required=True, help="Name of the experiment."
    )
    parser.add_argument(
        "--name_ckpt", type=str, required=True, help="Name of the checkpoint file."
    )
    parser.add_argument(
        "--ckpt-root",
        type=str,
        default="/home/ecm5702/scratch/aifs/checkpoint",
        help="Checkpoint root directory.",
    )
    parser.add_argument("--out_file", type=str, default="sigma_eval_table.csv")
    parser.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Optional full output CSV path. If unset, uses <ckpt-root>/<name_exp>/<out_file>.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=0,
        help="Model-parallel width. Use 0 to infer the canonical value from the checkpoint lane.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of validation batches to evaluate.",
    )
    parser.add_argument(
        "--validation_frequency",
        type=str,
        default="50h",
        help="Validation dataloader frequency override.",
    )
    parser.add_argument(
        "--sigmas",
        type=str,
        default="",
        help="Comma-separated sigma list override, e.g. '0.02,0.5,2'.",
    )
    parser.add_argument(
        "--run_pure_noise",
        action="store_true",
        help="Also evaluate pure-noise target mode.",
    )
    parser.add_argument(
        "--run_noised",
        action="store_true",
        help="Evaluate noised-target mode. If neither mode flag is set, this is enabled by default.",
    )
    return parser


def _resolve_out_csv(args: argparse.Namespace) -> Path:
    if args.out_csv:
        return Path(args.out_csv)
    return Path(args.ckpt_root) / args.name_exp / args.out_file


def run_sigma_evaluator(args: argparse.Namespace) -> Path:
    out_csv = _resolve_out_csv(args)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("ckpt used is", os.path.join(args.ckpt_root, args.name_exp, args.name_ckpt))
    print(f"Output CSV will be saved to: {out_csv}")

    global_rank, local_rank, world_size = _get_parallel_info()

    object_loader = ObjectFromCheckpointLoader(args.ckpt_root, args.name_exp, args.name_ckpt)
    checkpoint, config_checkpoint = get_checkpoint(args.ckpt_root, args.name_exp, args.name_ckpt)
    config = instantiate_config()
    config_checkpoint = adapt_config_hpc(config_checkpoint, config)

    object_loader.config_checkpoint = config_checkpoint
    # Some checkpoints were produced on external paths (e.g. /leonardo_work/...).
    # Rewrite known dataset prefixes to local mirrors when present.
    object_loader.config_for_datamodule = _rewrite_dataset_paths_in_place(
        object_loader.config_for_datamodule
    )
    object_loader.config_for_datamodule.dataloader.validation.frequency = (
        args.validation_frequency
    )
    if hasattr(object_loader.config_for_datamodule.dataloader.validation, "num_workers"):
        object_loader.config_for_datamodule.dataloader.validation.num_workers = 0

    inferred_lane = infer_lane_from_config(_normalize_cfg_for_lane_inference(config_checkpoint))
    required_model_parallel_gpus = (
        O1280_FAMILY_NUM_GPUS_PER_MODEL if inferred_lane in O1280_FAMILY_LANES else 1
    )
    requested_model_parallel_gpus = (
        int(args.num_gpus_per_model)
        if int(args.num_gpus_per_model) > 0
        else required_model_parallel_gpus
    )
    if (
        inferred_lane in O1280_FAMILY_LANES
        and requested_model_parallel_gpus != required_model_parallel_gpus
    ):
        raise RuntimeError(
            f"Lane {inferred_lane} requires num_gpus_per_model={required_model_parallel_gpus} "
            "for sigma evaluation. Single-GPU AG launches are not reliable for this lane."
        )
    if requested_model_parallel_gpus > 1 and world_size != requested_model_parallel_gpus:
        raise RuntimeError(
            f"Expected world_size={requested_model_parallel_gpus} for sigma evaluation on lane "
            f"{inferred_lane}, got {world_size}. Launch with srun/torchrun across "
            f"{requested_model_parallel_gpus} tasks."
        )

    if hasattr(object_loader.config_checkpoint, "hardware"):
        object_loader.config_checkpoint.hardware.num_gpus_per_model = requested_model_parallel_gpus
    if hasattr(object_loader.config_checkpoint.dataloader, "read_group_size"):
        object_loader.config_checkpoint.dataloader.read_group_size = requested_model_parallel_gpus
    if hasattr(object_loader.config_for_datamodule, "hardware"):
        object_loader.config_for_datamodule.hardware.num_gpus_per_model = requested_model_parallel_gpus
    if hasattr(object_loader.config_for_datamodule.dataloader, "read_group_size"):
        object_loader.config_for_datamodule.dataloader.read_group_size = requested_model_parallel_gpus

    fallback_residuals = None
    for cfg_candidate in (
        object_loader.config_checkpoint,
        object_loader.config_for_datamodule,
    ):
        repaired = _maybe_fix_missing_residual_statistics(cfg_candidate)
        if repaired is not None and fallback_residuals is None:
            fallback_residuals = repaired
    if fallback_residuals is not None:
        print(f"Using fallback residual statistics file: {fallback_residuals}")

    if args.device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")

    device = _resolve_device(requested_device, local_rank)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(int(str(device).split(":")[1]))

    try:
        model_comm_group = _init_model_comm_group(device, global_rank, world_size)
        object_loader.load()

        datamodule = object_loader.datamodule
        interface = object_loader.interface
        downscaler = object_loader.downscaler
        downscaler.model_comm_group = model_comm_group
        _ = checkpoint  # keep behavior; checkpoint is loaded for config compatibility.

        interface = interface.to(device)
        downscaler = downscaler.to(device)
        print(
            f"Running sigma evaluator on device: {device} "
            f"(lane={inferred_lane}, num_gpus_per_model={requested_model_parallel_gpus})"
        )

        if args.sigmas.strip():
            sigma_values = [float(x.strip()) for x in args.sigmas.split(",") if x.strip()]
        else:
            sigma_values = sigmas

        run_noised = args.run_noised or (not args.run_noised and not args.run_pure_noise)
        run_pure_noise = args.run_pure_noise
        sigma_evaluator = SigmaEvaluator(downscaler, datamodule, args.n_samples)

        def _run_one(sigma: float, prediction_on_pure_noise: bool) -> dict:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            loss, metrics = sigma_evaluator.evaluate_sigma(sigma, prediction_on_pure_noise)
            row = {
                "sigma": float(sigma),
                "prediction_on_pure_noise": bool(prediction_on_pure_noise),
                "loss": float(loss),
                "diff_all_var_non_weighted": float(metrics["diff_all_var_non_weighted"]),
            }
            if torch.cuda.is_available():
                row["cuda_max_memory_allocated_GB"] = float(
                    torch.cuda.max_memory_allocated() / 1e9
                )
            for k, v in metrics.items():
                try:
                    row[f"metric__{k}"] = float(v)
                except Exception:
                    row[f"metric__{k}"] = v
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return row

        rows = []
        for sigma in sigma_values:
            if run_noised:
                print(f"Evaluating sigma {sigma} with noisy output")
                rows.append(_run_one(sigma, prediction_on_pure_noise=False))
            if run_pure_noise:
                print(f"Evaluating sigma {sigma} with pure noise")
                rows.append(_run_one(sigma, prediction_on_pure_noise=True))

        if global_rank == 0:
            df = pd.DataFrame(rows)
            base_cols = ["sigma", "prediction_on_pure_noise", "loss"]
            metric_cols = sorted([c for c in df.columns if c.startswith("metric__")])
            other_cols = [c for c in df.columns if c not in base_cols + metric_cols]
            df = df[[c for c in base_cols if c in df.columns] + other_cols + metric_cols]
            if out_csv.exists():
                out_csv.unlink()
            df.to_csv(out_csv, index=False)

        _distributed_barrier(device=device, local_rank=local_rank)
        return out_csv
    finally:
        _destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    out_csv = run_sigma_evaluator(args)
    print(f"Saved sigma table: {out_csv}")


if __name__ == "__main__":
    main()
