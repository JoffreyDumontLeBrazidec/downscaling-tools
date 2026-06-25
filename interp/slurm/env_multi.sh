# Interp runtime for b785bf12 (multi-ds unified) sharded o320->o1280.
# dev/multi/anemoi-core has BOTH the diffusiondownscaler class (deserializes the
# pickle) AND apply_shard_shapes/get_shard_shapes (the sharded forward needs them);
# dev/unified lacks the latter. Explicit PYTHONPATH (mirrors env_260612's rationale).
source /home/ecm5702/dev/.ds-260612/bin/activate
export PYTHONPATH=/home/ecm5702/dev/interp_overlay:/home/ecm5702/dev/multi/anemoi-core/models/src:/home/ecm5702/dev/multi/anemoi-core/training/src:/home/ecm5702/dev/multi/anemoi-core/graphs/src
export ANEMOI_BASE_SEED=756 TORCH_COMPILE_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR="" DEV=/home/ecm5702/dev/ OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/ INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
