#!/bin/bash
 
#SBATCH --nodes=1
#SBATCH --qos=ng
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=05:00:00
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/2026-01-26/sigma_evaluator-%j.out

set -eux


source /home/ecm5702/dev/.ds-dyn/bin/activate
export ANEMOI_BASE_SEED=756
export HYDRA_FULLL_ERROR=1
export NCCL_IB_TIMEOUT=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEV=/home/ecm5702/dev/
export DATA_DIR=/home/mlx/ai-ml/datasets/
export DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/

WORKDIR=/home/ecm5702/dev
cd $WORKDIR

name_exp="c0367b4ebe8849029985f521d1196426"
name_ckpt="anemoi-by_epoch-epoch_000-step_000001.ckpt" 
out_file="sigma_eval_table_000001.csv"

srun python -u /home/ecm5702/dev/downscaling-tools/eval_suite/sigma_evaluator/run_sigma_evaluator.py --name_exp $name_exp --name_ckpt $name_ckpt --out_file $out_file