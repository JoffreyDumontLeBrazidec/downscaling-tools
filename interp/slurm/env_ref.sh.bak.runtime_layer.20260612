# Shared interp runtime for DS (single-dataset) checkpoints, post env-reorg 2026-06-11.
# .ds-dyn was deleted; .ds-dyn-dev exists but its anemoi editables point at the
# deleted anemoi-core-dev. The live DS sources (with data_indices/ds_tensor.py,
# needed to deserialize DS checkpoints like eecdb127) are in anemoi-core-ref.
source /home/ecm5702/dev/.ds-dyn-dev/bin/activate
# earthkit-utils 0.1.2 overlay: .ds-dyn-dev ships utils 0.3.0 which dropped
# array_to_numpy that earthkit-data 0.18.3 still calls when loading .nc bundles.
export PYTHONPATH=/home/ecm5702/dev/interp_overlay:/home/ecm5702/dev/anemoi-core-ds/models/src:/home/ecm5702/dev/anemoi-core-ds/training/src:/home/ecm5702/dev/anemoi-core-ds/graphs/src
export ANEMOI_BASE_SEED=756 TORCH_COMPILE_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR="" DEV=/home/ecm5702/dev/ OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/ INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
