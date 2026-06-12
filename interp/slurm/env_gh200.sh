# Interp runtime for GH200 nodes (aarch64) — .ds-ag venv (torch 2.7 / cu128).
# Select with:  sbatch --gres=gpu:gh200:1 --export=ALL,INTERP_ENV=env_gh200 run.sbatch ...
source /home/ecm5702/dev/.ds-ag/bin/activate
export ANEMOI_BASE_SEED=756 TORCH_COMPILE_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR="" DEV=/home/ecm5702/dev/ OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/ INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
