#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=494000
#SBATCH --output=/leonardo/home/userexternal/jdumontl/dev/jobscripts/inference/outputs/%j.out
#SBATCH --partition=boost_usr_prod
#SBATCH --account=DestE_340_25


 
set -eux
source /leonardo/home/userexternal/jdumontl/dev/aifs/bin/activate

export ANEMOI_BASE_SEED=756
export HYDRA_FULLL_ERROR=1
export NCCL_IB_TIMEOUT=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEV=/leonardo/home/userexternal/jdumontl/dev/
export DATA_DIR=/leonardo_work/DestE_340_25/ai-ml/datasets/
export DATA_STABLE_DIR=/leonardo_work/DestE_340_25/ai-ml/datasets/
export OUTPUT=/leonardo_work/DestE_340_25/output/jdumontl/downscaling/
export GRID_DIR=/leonardo_work/DestE_340_25/AIFS_grids
export INTER_MAT_DIR=/leonardo/home/userexternal/jdumontl/inter_mat/
export RESIDUAL_STATISTICS_DIR=/leonardo/home/userexternal/jdumontl/residuals_statistics/
export PYTHONPATH="/leonardo/home/userexternal/jdumontl/dev/downscaling-tools:${PYTHONPATH:-}"
export HPC="leo"

inference="save_sampling.py"

name_exp="4d3399b2ce754d269b339d79a024d806"
name_ckpt="anemoi-by_epoch-epoch_020-step_388752.ckpt"
N_members=2
N_samples=1
num_steps=50

srun --export=ALL,HPC python $inference --name_exp $name_exp --name_ckpt $name_ckpt --N_members $N_members --num_steps=$num_steps --N_samples $N_samples
