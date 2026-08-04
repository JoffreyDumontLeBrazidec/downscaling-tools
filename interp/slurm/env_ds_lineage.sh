# Interp runtime for ds-API checkpoints (eecdb127/cfec83a3/181be03e), post the
# 2026-06-26 runtime consolidation. Sources the guard-verified `ds` replay
# lineage (anemoi-core @ a7d6e8ae9) and re-adds the earthkit-utils 0.1.2 overlay
# (interp_overlay) — the lineage venv ships a newer earthkit-utils that dropped
# array_to_numpy, which earthkit-data still calls when loading .nc bundles.
source /home/ecm5702/hpcperm/lineages/ds/activate.sh
export PYTHONPATH="/home/ecm5702/dev/interp_overlay:${PYTHONPATH:-}"
export ANEMOI_BASE_SEED=756 TORCH_COMPILE_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR="" DEV=/home/ecm5702/dev/ OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/ INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
