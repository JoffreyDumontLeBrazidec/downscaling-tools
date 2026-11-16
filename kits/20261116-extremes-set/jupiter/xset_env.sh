# Extremes set XSET-1 (16 Nov 2026): environment of the global GL512 read on Jupiter. Source it after xset_runs.sh.
# Copy of agent-work/20261104-ens-member-gl512/kit/env.sh as it stood on 16 Nov 2026 09:53 UTC (md5 74ae329b: modules, the
# .ds venv, code and datasets sha checks, chunks 1/1/1 = bit-identical and 34.8 s per draw, speed-up note s3b); only OUTPUT differs.
set +e +u
source /e/software/default/configs/modules.sh 2>/dev/null
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3 git/2.45.1 >/dev/null 2>&1
set -eo pipefail
source /e/data1/jureap-data/ecmwf/users/jdumont/venvs/.ds/bin/activate
DEV_ROOT=/e/home/jusers/dumontlebrazidec1/jupiter/dev
: "${CODE_ROOT:?source xset_runs.sh first}" "${CODE_SHA_FILE:?}" "${DATASETS_SHA_FILE:?}"
DATASETS_REPO=$DEV_ROOT/anemoi-datasets-hres-precip-ru
[[ "$(git -C $CODE_ROOT rev-parse HEAD)" == "$(tr -d '[:space:]' < $CODE_SHA_FILE)" ]] || { echo "FATAL code sha"; return 1; }
[[ -z "$(git -C $CODE_ROOT status --porcelain)" ]] || { echo "FATAL dirty code tree"; return 1; }
[[ "$(git -C $DATASETS_REPO rev-parse HEAD)" == "$(tr -d '[:space:]' < $DATASETS_SHA_FILE)" ]] || { echo "FATAL datasets sha"; return 1; }
echo "[env] code $CODE_ROOT at $(git -C $CODE_ROOT rev-parse HEAD) = CODE_PIN, clean; datasets $(git -C $DATASETS_REPO rev-parse --short HEAD) = DATASETS_PIN"
export DEV="$DEV_ROOT" WORKDIR="$DEV_ROOT"
export OUTPUT=/e/data1/jureap-data/ecmwf/users/jdumont/outputs/xset1_20261116/_anemoi_output
export RESIDUAL_STATISTICS_DIR=/e/data1/jureap-data/ecmwf/users/jdumont/anemoi/residuals_statistics
export HYDRA_FULL_ERROR=1 PYTHONFAULTHANDLER=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ANEMOI_GRAPH_TRANSFORMER_IMPL=triton ANEMOI_BASE_SEED=756
export NCCL_IB_TIMEOUT=30 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR=1 ANEMOI_INFERENCE_NUM_CHUNKS=1 ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=1
export SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4} SLURM_NNODES=${SLURM_NNODES:-1} TMPDIR=/tmp
export TRITON_CACHE_DIR=/e/data1/jureap-data/ecmwf/users/jdumont/.cache/triton
export CUDA_CACHE_PATH=/e/data1/jureap-data/ecmwf/users/jdumont/.cache/nv_compute
export PYTHONPATH="$DATASETS_REPO/src:$CODE_ROOT/training/src:$CODE_ROOT/models/src:$CODE_ROOT/graphs/src"
mkdir -p "$OUTPUT" "/tmp/$USER/hydra"
