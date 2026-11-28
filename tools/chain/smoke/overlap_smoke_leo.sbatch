#!/bin/bash
#SBATCH --job-name=ovl_smoke
#SBATCH --account=DestE_340_26
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Run from the smoke folder: cd <smoke dir> && sbatch overlap_smoke_leo.sbatch
bash "$SLURM_SUBMIT_DIR/overlap_smoke_body.sh" "$SLURM_SUBMIT_DIR/root" 1500
