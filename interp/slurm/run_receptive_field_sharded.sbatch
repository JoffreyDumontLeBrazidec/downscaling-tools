#!/bin/bash --login
# ============================================================================
# Grid-sharded launcher for the interp `receptive_field` tool.
#
# The rank count and the node count are POSITIONAL parameters, because the two
# high-resolution lanes need different shapes: the o320->o1280 lane fits 4 ranks
# on one AC node, while the o1280->o2560 lane needs 8 ranks across 2 nodes.
# They are positional rather than environment variables on purpose: ECMWF's
# sbatch wrapper rewrites --export, so nothing exported at submit time survives
# into the job.  Everything the job needs therefore arrives on its command line.
#
# USAGE
#   sbatch --nodes=<N> --ntasks-per-node=<R/N> [--time=...] \
#     interp/slurm/run_receptive_field_sharded.sbatch \
#       <CKPT_PATH> <CKPT_ID> <NODES> <NTASKS> <INTERP_ENV> <CHUNKS> <EVENT> [extra...]
#
#   The --nodes / --ntasks-per-node given to sbatch must agree with NODES and
#   NTASKS/NODES: sbatch allocates, NTASKS is what srun then spawns.
#   INTERP_ENV names a file in interp/slurm/ without the .sh suffix.
#   CHUNKS sets all three ANEMOI_INFERENCE_NUM_CHUNKS* latches (they are read at
#   import time, so they are exported before python starts).
#
#   Extras follow the dash-less key[=value] convention of run.sbatch (ECMWF's
#   sbatch wrapper swallows --flags after the script name).  List-valued keys
#   (sigmas, state-sigmas, dump-sigmas, centres, dates, members, steps) take
#   comma lists:
#     sbatch ... <CKPT> <ID> 1 4 env_pristine_regional 1 franklin_o320_o1280 \
#         sigmas=80,20,5,1,0.2 theoretical-reach-km=510 r0-km=50
#
# Results land in /home/ecm5702/perm/interp/<CKPT_ID>/receptive_field/ unless an
# 8th positional OUT_ROOT is given via out-root=<dir> among the extras.
# ============================================================================
#SBATCH --job-name=interp_rf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --gres=gpu:ga100:4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/interp_rf_%j.out
#SBATCH --error=/home/ecm5702/dev/jobscripts/outputs/interp_rf_%j.err
set -euxo pipefail

usage() {
  echo "usage: run_receptive_field_sharded.sbatch <CKPT_PATH> <CKPT_ID> <NODES>" \
       "<NTASKS> <INTERP_ENV> <CHUNKS> <EVENT> [extra...]" >&2
  exit 2
}

CKPT_PATH="${1:-}"; CKPT_ID="${2:-}"; NODES="${3:-}"; NTASKS="${4:-}"
ENV_NAME="${5:-}"; CHUNKS="${6:-}"; EVENT="${7:-}"
[[ -n "$CKPT_PATH" && -n "$CKPT_ID" && -n "$NODES" && -n "$NTASKS" \
   && -n "$ENV_NAME" && -n "$CHUNKS" && -n "$EVENT" ]] || usage
shift 7

# Chunking knobs are module-level latches read at import time -> export BEFORE python.
export ANEMOI_INFERENCE_NUM_CHUNKS="${CHUNKS}"
export ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR="${CHUNKS}"
export ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER="${CHUNKS}"

REPO="/home/ecm5702/dev/downscaling-tools-interp-rf"
OUT_ROOT="/home/ecm5702/perm/interp/${CKPT_ID}"

# Convert dash-less key[=value] extras into --key value tool flags.
NARGS_KEYS=" sigmas state-sigmas dump-sigmas centres dates members steps "
EXTRA=()
for kv in "$@"; do
  if [[ "$kv" == out-root=* ]]; then
    OUT_ROOT="${kv#out-root=}"
    continue
  fi
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

PER_NODE=$(( NTASKS / NODES ))

source "${REPO}/interp/slurm/${ENV_NAME}.sh"
cd "${REPO}"

# NO --gpus-per-task: every rank must see all GPUs so the loader can call
# torch.cuda.set_device(local_rank).  MASTER_ADDR/PORT are left unset on purpose
# -> manual_inference._init_model_comm_group falls back to init_parallel, which
# derives the rendezvous from SLURM_NODELIST / SLURM_JOBID.
srun --nodes="${NODES}" --ntasks="${NTASKS}" --ntasks-per-node="${PER_NODE}" \
  python -m interp receptive_field --checkpoint "${CKPT_PATH}" \
    --output-dir "${OUT_ROOT}/receptive_field" --event "${EVENT}" "$@"

echo "receptive_field (sharded x${NTASKS} over ${NODES} node(s)) complete for ${CKPT_ID} -> ${OUT_ROOT}"
