#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=/home/ecm5702/dev/jobscripts/manual_inference_job_outputs/%j.out
#SBATCH --qos=dg

set -eux
cd /home/ecm5702/dev/downscaling-tools/manual_inference

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
export PYTHONPATH="/home/ecm5702/dev/downscaling-tools:${PYTHONPATH:-}"
export HPC="atos"

inference="save_sampling.py"

dir_exp="/home/ecm5702/perm/leo_checkpoint_save"
name_exp="20c6993866294adc8ebcaec21fd6d329"
name_ckpt="anemoi-by_time-epoch_460-step_1066516.ckpt"
N_members=2
N_samples=1
num_steps=50
sigma_max=10000
S_max=800
S_noise=1.05
S_churn=2.5
sigma_min=0.03
S_min=0.75

srun --export=ALL,HPC python $inference --dir_exp $dir_exp --name_exp $name_exp --name_ckpt $name_ckpt --N_members $N_members --num_steps=$num_steps --N_samples $N_samples --sigma_max $sigma_max --S_max $S_max --S_noise $S_noise --S_churn $S_churn --sigma_min $sigma_min --S_min $S_min