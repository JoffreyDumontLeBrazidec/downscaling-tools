from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import pandas as pd
import torch

from manual_inference.checkpoints import (
    ObjectFromCheckpointLoader,
    adapt_config_hpc,
    get_checkpoint,
    instantiate_config,
)
from manual_inference.prediction.predict import _rewrite_dataset_paths_in_place

from .sigma_evaluator import SigmaEvaluator
from .sigmas import sigmas

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
    object_loader.load()

    datamodule = object_loader.datamodule
    interface = object_loader.interface
    downscaler = object_loader.downscaler
    _ = checkpoint  # keep behavior; checkpoint is loaded for config compatibility.

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")

    interface = interface.to(device)
    downscaler = downscaler.to(device)
    print(f"Running sigma evaluator on device: {device}")

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

    df = pd.DataFrame(rows)
    base_cols = ["sigma", "prediction_on_pure_noise", "loss"]
    metric_cols = sorted([c for c in df.columns if c.startswith("metric__")])
    other_cols = [c for c in df.columns if c not in base_cols + metric_cols]
    df = df[[c for c in base_cols if c in df.columns] + other_cols + metric_cols]
    if out_csv.exists():
        out_csv.unlink()
    df.to_csv(out_csv, index=False)
    return out_csv


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    out_csv = run_sigma_evaluator(args)
    print(f"Saved sigma table: {out_csv}")


if __name__ == "__main__":
    main()
