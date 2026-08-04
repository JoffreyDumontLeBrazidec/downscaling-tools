# Interp runtime for the tc_o320_o1280 REGIONAL lane checkpoints (dict-API,
# plain pristine, num_gpus_per_model=1 -- single-GPU-able because the local
# hidden mesh is only 28,226 nodes (23x smaller than global o1280 655,362), so
# none of the b785 sharding fixes (deadlock/shard_strategy/reconstruct) apply.
#
# Regional ckpts are pickled against the training runtime that produced them:
# ~/dev/pristine/anemoi-core (cert-20260707-02, c286c1a33), venv .ds-260612,
# NO PYTHONPATH overlay -- .ds-260612 anemoi.{training,models,graphs} editable
# installs already resolve straight to ~/dev/pristine/anemoi-core (verified
# 2026-07-10: python -c import anemoi.training as t; print(t.__file__)).
# This matches fast_tc_check.sh comment: no PYTHONPATH overlay, plain
# pristine .ds-260612 code path.
source /home/ecm5702/dev/.ds-260612/bin/activate
export ANEMOI_BASE_SEED=756 TORCH_COMPILE_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR=/home/mlx/ai-ml/datasets/ DEV=/home/ecm5702/dev/ OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/ INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
