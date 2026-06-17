#!/bin/bash --login
# ============================================================================
# Multi-GPU GRID-SHARDED launcher for the interp `trajectory` tool — for lanes
# too large for one GPU (O1280). srun spawns N ranks that grid-shard ONE model
# across the node's GPUs (manual_inference._init_model_comm_group + the DS
# model's native grid sharding). The trajectory tool auto-detects world_size>1.
#
# USAGE:
#   sbatch interp/slurm/run_trajectory_sharded.sbatch <CKPT_PATH> <CKPT_ID> [extra...]
#   INTERP_EVENT=franklin_o320_o1280 (default), TRAJ_NTASKS=4 (default), INTERP_ENV=env_ref.
#   Extras are dash-less key[=value] (ecsbatch swallows --flags); list-valued keys
#   (seeds, ceiling-sigmas, sigmas, dates, members, steps) accept comma lists.
#     sbatch ... <CKPT> 181b_traj num-steps=12 n-seeds=3 ceiling-sigmas=5,10,20,40,80
# ============================================================================
#SBATCH --job-name=traj_shard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --gres=gpu:ga100:4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --mem=320G
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/interp_%j.out
#SBATCH --error=/home/ecm5702/dev/jobscripts/outputs/interp_%j.err
set -euxo pipefail

CKPT_PATH="${1:?Usage: run_trajectory_sharded.sbatch <CKPT_PATH> <CKPT_ID> [extra...]}"
CKPT_ID="${2:?CKPT_ID}"
shift 2
# Convert dash-less key[=value] extras into --key value tool flags (mirrors run.sbatch).
NARGS_KEYS=" sigmas seeds ceiling-sigmas seed-sigmas dates members steps "
EXTRA=()
for kv in "$@"; do
  if [[ "$kv" == *=* ]]; then
    k="${kv%%=*}"; v="${kv#*=}"
    if [[ "$NARGS_KEYS" == *" $k "* ]]; then
      EXTRA+=("--$k"); IFS=, read -ra parts <<< "$v"; EXTRA+=("${parts[@]}")
    else
      EXTRA+=("--$k=$v")
    fi
  else
    EXTRA+=("--$kv")
  fi
done
set -- ${EXTRA[@]+"${EXTRA[@]}"}

OUT_ROOT="/home/ecm5702/perm/interp/${CKPT_ID}"
EVENT="${INTERP_EVENT:-franklin_o320_o1280}"
NTASKS="${TRAJ_NTASKS:-4}"

source "/home/ecm5702/dev/downscaling-tools/interp/slurm/${INTERP_ENV:-env_ref}.sh"
cd /home/ecm5702/dev/downscaling-tools

# srun spawns NTASKS ranks that grid-shard ONE model. NO --gpus-per-task: each
# rank must see all GPUs so the loader can torch.cuda.set_device(local_rank).
# MASTER_ADDR/PORT are left unset on purpose -> manual_inference._init_model_comm_group
# falls back to distributed.utils.init_parallel (tcp:// derived from SLURM_NODELIST/JOBID).
srun --ntasks="${NTASKS}" --ntasks-per-node="${NTASKS}" \
  python -m interp trajectory --checkpoint "${CKPT_PATH}" \
    --output-dir "${OUT_ROOT}/trajectory" --event "${EVENT}" "$@"

echo "trajectory (sharded x${NTASKS}) complete for ${CKPT_ID} -> ${OUT_ROOT}"
