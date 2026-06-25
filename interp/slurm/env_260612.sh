# Interp runtime for UNIFIED (multi-ds-unified, dict-API) checkpoints — ds-260612.
# Select with:  sbatch --export=ALL,INTERP_ENV=env_260612 ...
#
# Direct activation (mirrors env_ref.sh) instead of the canonical runtime activator:
# that activator has a hard-fail assert layer that is flaky/not set -e safe under sbatch
# (`set -euxo pipefail`) and intermittently left anemoi only partially on sys.path
# (ModuleNotFoundError on the diffusiondownscaler submodule during unpickle). The explicit
# PYTHONPATH below pins anemoi.{models,training,graphs} to the unified checkout deterministically.
source /home/ecm5702/dev/.ds-260612/bin/activate
export PYTHONPATH=/home/ecm5702/dev/interp_overlay:/home/ecm5702/dev/unified/anemoi-core/models/src:/home/ecm5702/dev/unified/anemoi-core/training/src:/home/ecm5702/dev/unified/anemoi-core/graphs/src
export ANEMOI_BASE_SEED=756 TORCH_COMPILE_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR="" DEV=/home/ecm5702/dev/ OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/ INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
