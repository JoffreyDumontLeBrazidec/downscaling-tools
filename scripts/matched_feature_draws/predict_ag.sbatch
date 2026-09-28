#!/bin/bash --login
#SBATCH --job-name=matched-draws
#SBATCH --partition=gpu_debug
#SBATCH --qos=dg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/home/ecm5702/agent-work/20260915-matched-feature-draws/logs/%x_%j.out
set -euo pipefail
ARM="${1:-${ARM:-}}"; : "${ARM:?ARM positionally, e.g. R47k_full}"
CKPT="${2:-${CKPT:-}}"; : "${CKPT:?BASE checkpoint path positionally}"
D="${3:-${D:-}}"; : "${D:?date YYYYMMDD positionally}"
SB=/home/ecm5702/hpcperm/sandbox/20260906-hres-lead-branch/code
VENV=/home/ecm5702/dev/.ds-ag-260616
ROOT=/home/ecm5702/agent-work/20260915-matched-feature-draws
FIRST=${4:?first seed index}
LAST=${5:?last seed index}
OUT=$ROOT/outputs/$ARM
[[ -f "$CKPT" ]] || { echo "FATAL missing checkpoint: $CKPT"; exit 1; }
module load ecmwf-toolbox 2>/dev/null || true
source "$VENV/bin/activate" || { echo "venv activation FAILED: $VENV"; exit 3; }
export PYTHONPATH="$SB/anemoi-core/training/src:$SB/anemoi-core/models/src:$SB/anemoi-core/graphs/src:$SB/anemoi-inference/src"
python - "$SB" <<'PYGUARD' || { echo "GUARD FAILED: overlay did not win"; exit 4; }
import importlib, os, sys
root = os.path.realpath(sys.argv[1]); bad = []
for name in ("anemoi.models", "anemoi.training", "anemoi.graphs", "anemoi.inference"):
    p = os.path.realpath(importlib.import_module(name).__file__)
    (bad.append if not p.startswith(root + os.sep) else print)(f"  {name} -> {p}")
import torch; print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()}")
if bad: print("\n".join(bad)); sys.exit(1)
PYGUARD
export DATA_DIR=/home/mlx/ai-ml/datasets/ DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/ OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/ INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ANEMOI_INFERENCE_NUM_CHUNKS=1 ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR=1 ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" CODEX_UNIFIED_SHARD_STRATEGY=edges
export TORCH_COMPILE_DISABLE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ANEMOI_GRAPH_TRANSFORMER_IMPL=triton HYDRA_FULL_ERROR=1
export TMPDIR=/tmp; mkdir -p /tmp/$USER/hydra
cd $ROOT/tooling
echo "host=$(hostname) arch=$(uname -m) start=$(date -Is) dstools=$(git rev-parse --short HEAD) core=$(git -C $SB/anemoi-core rev-parse --short HEAD) arm=$ARM ckpt=$CKPT date=$D members=1 steps=24,120 precision=fp32"
mkdir -p "$OUT"

if [[ "$D" == 20250926 ]]; then
 INPUT_ROOT=/home/ecm5702/hpcperm/data/input_data/o320_o1280/humberto_20250926_20250930
else
 INPUT_ROOT=/home/ecm5702/scratch/eval/o320_o1280/manual_731d203a_pristine_20260818/bundles_with_y
fi
sha256sum "$CKPT" > "$ROOT/outputs/${ARM}_checkpoint_sha256.txt"
for ((i=FIRST; i<=LAST; i++)); do
 export ANEMOI_BASE_SEED=$((2026091600 + 100*i))
 DEST="$OUT/seed_${ANEMOI_BASE_SEED}/predictions"
 mkdir -p "$DEST"
 echo "DRAW_START arm=$ARM date=$D seed=$ANEMOI_BASE_SEED seed_index=$i"
 srun --ntasks 4 --ntasks-per-node 4 python -m eval.predict.main  --name-ckpt "$CKPT" --out-dir "$DEST" --members 1 --steps 24,120 --dates "$D"  --input-root "$INPUT_ROOT" --allow-existing-out-dir --num-gpus-per-model 4 --precision fp32  --output-weather-states 10u,10v  --local-scope-json '{"mode":"bbox","lat_min":10,"lat_max":40,"lon_min":-100,"lon_max":-58}'  --extra-args-json '{"schedule_type":"experimental_piecewise","num_steps":30,"sigma_max":100000.0,"sigma_transition":10.0,"sigma_min":0.03,"high_schedule_type":"exponential","low_schedule_type":"karras","num_steps_high":10,"num_steps_low":20,"rho":7.0,"sampler":"heun","S_churn":2.5,"S_min":0.75,"S_max":100000.0,"S_noise":1.05}'
 echo "DRAW_COMPLETE arm=$ARM date=$D seed=$ANEMOI_BASE_SEED"
done
