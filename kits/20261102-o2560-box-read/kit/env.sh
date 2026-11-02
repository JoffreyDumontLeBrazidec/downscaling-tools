# o2560 Europe-box read (2 Nov 2026): environment for the predictor and the dry runs. Source it after rows.sh (which sets
# CODE_ROOT and CODE_PIN_SHA for the row). Modules, venv and anemoi-datasets fork as the Jupiter chain db_o2560_e2_50k_v2.
set +e +u
source /e/software/default/configs/modules.sh 2>/dev/null
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3 git/2.45.1 >/dev/null 2>&1
set -eo pipefail
source /e/data1/jureap-data/ecmwf/users/jdumont/venvs/.ds/bin/activate
KIT=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read
DATASETS_REPO=/e/home/jusers/dumontlebrazidec1/jupiter/dev/anemoi-datasets-hres-precip-ru
DATASETS_PIN=870fb9bc36bd59a7b09420f80192645a12ebf7e2   # DATASETS_PIN of db_o2560_e2_50k_v2_20261030 (and the AG crop chain)
: "${CODE_ROOT:?rows.sh first}" "${CODE_PIN_SHA:?rows.sh first}"
[[ "$(git -C $CODE_ROOT rev-parse HEAD)" == "$CODE_PIN_SHA" ]] || { echo "FATAL code sha $(git -C $CODE_ROOT rev-parse HEAD) != $CODE_PIN_SHA"; return 1; }
[[ -z "$(git -C $CODE_ROOT status --porcelain --untracked-files=no)" ]] || { echo "FATAL dirty code tree $CODE_ROOT"; return 1; }
[[ "$(git -C $DATASETS_REPO rev-parse HEAD)" == "$DATASETS_PIN" ]] || { echo "FATAL datasets sha"; return 1; }
GT=$CODE_ROOT/models/src/anemoi/models/triton/gt.py   # the fixed Triton kernel guard of the training job (exit 47 there)
grep -q 'out_saved = torch.empty((N_dst, H, C), device=q.device, dtype=torch.float32)' $GT && grep -q 'D = torch.empty((N_dst, H), device=q.device, dtype=torch.float32)' $GT || { echo "FATAL $GT is not the fixed Triton kernel (bbca1748, fork f1659385f)"; return 1; }
export OUTPUT=/e/data1/jureap-data/ecmwf/users/jdumont/outputs/o2560_box_read_20261102/_anemoi_output
export HYDRA_FULL_ERROR=1 PYTHONFAULTHANDLER=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ANEMOI_GRAPH_TRANSFORMER_IMPL=triton ANEMOI_BASE_SEED=756
export ANEMOI_INFERENCE_NUM_CHUNKS=${ANEMOI_INFERENCE_NUM_CHUNKS:-16}
export SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1} SLURM_NNODES=${SLURM_NNODES:-1} TMPDIR=/tmp
export TRITON_CACHE_DIR=/e/data1/jureap-data/ecmwf/users/jdumont/.cache/triton
export CUDA_CACHE_PATH=/e/data1/jureap-data/ecmwf/users/jdumont/.cache/nv_compute
export PYTHONPATH="$DATASETS_REPO/src:$CODE_ROOT/training/src:$CODE_ROOT/models/src:$CODE_ROOT/graphs/src"
export CFG_DIR=$KIT/configs CFG_NAME=db_o2560_read
mkdir -p "$OUTPUT" "/tmp/$USER/hydra"
