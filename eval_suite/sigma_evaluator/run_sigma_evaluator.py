import os
import sys

sys.path.append("/home/ecm5702/dev/downscaling-tools")

from get_objects_from_ckpt import *
import gc
import torch
import pandas as pd

from eval_suite.sigma_evaluator.sigma_evaluator import SigmaEvaluator
import argparse

parser = argparse.ArgumentParser(description="Run sigma evaluator with sbatch.")
parser.add_argument(
    "--name_exp", type=str, required=True, help="Name of the experiment."
)
parser.add_argument(
    "--name_ckpt", type=str, required=True, help="Name of the checkpoint file."
)
parser.add_argument("--out_file", type=str, default="sigma_eval_table.csv")

args = parser.parse_args()


dir_exp = "/home/ecm5702/scratch/aifs/checkpoint"
name_exp = args.name_exp
name_ckpt = args.name_ckpt
out_csv = os.path.join(dir_exp, name_exp, args.out_file)
print("ckpt used is", os.path.join(dir_exp, name_exp, name_ckpt))
print(f"Output CSV will be saved to: {out_csv}")


# Prepare object_loader
object_loader = ObjectFromCheckpointLoader(dir_exp, name_exp, name_ckpt)

## Modify config of object_loader before loading objects
checkpoint, config_checkpoint = get_checkpoint(dir_exp, name_exp, name_ckpt)
config = instantiate_config()
config_checkpoint = adapt_config_hpc(config_checkpoint, config)

object_loader.config_checkpoint = config_checkpoint
object_loader.config_for_datamodule.dataloader.validation.frequency = "50h"

# Load objects
object_loader.load()

datamodule = object_loader.datamodule
interface = object_loader.interface
downscaler = object_loader.downscaler
device = "cuda"
interface = interface.to(device)
downscaler = downscaler.to(device)

from sigmas import sigmas
import argparse

N_samples = 10
sigma_evaluator = SigmaEvaluator(downscaler, datamodule, N_samples)


def _run_one(sigma: float, prediction_on_pure_noise: bool):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    loss, metrics = sigma_evaluator.evaluate_sigma(sigma, prediction_on_pure_noise)

    row = {
        "sigma": float(sigma),
        "prediction_on_pure_noise": bool(prediction_on_pure_noise),
        "loss": float(loss),
    }

    # record GPU peak memory during this sigma run (optional but useful)
    if torch.cuda.is_available():
        row["cuda_max_memory_allocated_GB"] = float(
            torch.cuda.max_memory_allocated() / 1e9
        )

    # flatten metrics dict into columns
    for k, v in metrics.items():
        try:
            row[f"metric__{k}"] = float(v)
        except Exception:
            row[f"metric__{k}"] = v

    # best-effort cleanup between runs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


rows = []
for sigma in sigmas:
    print(f"Evaluating sigma {sigma} with noisy output")
    rows.append(_run_one(sigma, prediction_on_pure_noise=False))
for sigma in sigmas:
    print(f"Evaluating sigma {sigma} with pure noise")
    rows.append(_run_one(sigma, prediction_on_pure_noise=True))

df = pd.DataFrame(rows)

base_cols = [
    "sigma",
    "prediction_on_pure_noise",
    "loss",
]


metric_cols = sorted([c for c in df.columns if c.startswith("metric__")])
other_cols = [c for c in df.columns if c not in base_cols + metric_cols]
df = df[[c for c in base_cols if c in df.columns] + other_cols + metric_cols]
# df = df[[c for c in base_cols if c in df.columns] ]


df.to_csv(out_csv, index=False)
df
