# Interp runtime for UNIFIED (multi-ds-unified) checkpoints — ds-260612.
# Mirrors eval host atos_ac_unified: .ds-260612 venv + multi-ds-unified anemoi-core
# (deserializes unified ckpts) + real DATA_DIR (forcings) + grids/residual stats.
module load ecmwf-toolbox 2>/dev/null || true
source /home/ecm5702/dev/runtimes/ds-260612/activate.sh
export DATA_DIR=/home/mlx/ai-ml/datasets/ DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export ANEMOI_BASE_SEED=756 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1 TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_COMPILE_DISABLE=1
