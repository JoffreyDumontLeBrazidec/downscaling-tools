# Shared interp runtime for DS (single-dataset) checkpoints.
# 2026-06-12 runtime cleanup: venv + PYTHONPATH (incl. the earthkit-utils 0.1.2
# overlay and the live DS sources in anemoi-core-ds, which carry
# data_indices/ds_tensor.py needed to deserialize DS checkpoints) now come from
# the runtime layer, which also asserts import resolution and branch/sha.
source /home/ecm5702/dev/runtimes/ds/activate.sh
export ANEMOI_BASE_SEED=756 TORCH_COMPILE_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR="" DEV=/home/ecm5702/dev/ OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/ INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
