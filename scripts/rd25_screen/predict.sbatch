#!/bin/bash --login
#SBATCH --job-name=se_full_ag
#SBATCH --partition=gpu_debug
#SBATCH --qos=dg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=/home/ecm5702/agent-work/20260915-rd25-evaluation/logs/%x_%j.out
# Six-sample matched RD25k/R47k diagnostic. Global forward, GH200 fp32, seed 756.
# Sampler is inherited unchanged; spectra explicitly use member 1 only.
set -euo pipefail
ARM="${1:-${ARM:-}}"; : "${ARM:?ARM positionally, e.g. R47k_full}"
CKPT="${2:-${CKPT:-}}"; : "${CKPT:?BASE checkpoint path positionally}"
D="${3:-${D:-}}"; : "${D:?date YYYYMMDD positionally}"
SB=/home/ecm5702/hpcperm/sandbox/20260906-hres-lead-branch/code
VENV=/home/ecm5702/dev/.ds-ag-260616
OUT=/home/ecm5702/scratch/eval/o320_o1280/rd25_screen_${ARM}
[[ -f "$CKPT" ]] || { echo "FATAL missing checkpoint: $CKPT"; exit 1; }
module load ecmwf-toolbox 2>/dev/null || true
source "$VENV/bin/activate" || { echo "venv activation FAILED: $VENV"; exit 3; }
CORE=/home/ecm5702/agent-work/20260915-rd25-evaluation/core
export PYTHONPATH="$CORE/training/src:$CORE/models/src:$CORE/graphs/src:$SB/anemoi-inference/src"
export ANEMOI_BASE_SEED=756
python - "$SB" "$CORE" <<'PYGUARD' || { echo "GUARD FAILED: overlay did not win"; exit 4; }
import importlib, os, sys
root = os.path.realpath(sys.argv[1]); bad = []
for name in ("anemoi.models", "anemoi.training", "anemoi.graphs", "anemoi.inference"):
    p = os.path.realpath(importlib.import_module(name).__file__)
    expected = root if name == "anemoi.inference" else os.path.realpath(sys.argv[2])
    (bad.append if not p.startswith(expected + os.sep) else print)(f"  {name} -> {p}")
import torch; print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()}")
if bad or not torch.cuda.is_available(): print("\n".join(bad)); sys.exit(1)
print("GUARD OK: RD source and CUDA")
PYGUARD
export DATA_DIR=/home/mlx/ai-ml/datasets/ DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/ OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/ INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ANEMOI_INFERENCE_NUM_CHUNKS=1 ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR=1 ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" CODEX_UNIFIED_SHARD_STRATEGY=edges
export TORCH_COMPILE_DISABLE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ANEMOI_GRAPH_TRANSFORMER_IMPL=triton HYDRA_FULL_ERROR=1
export TMPDIR=/tmp; mkdir -p /tmp/$USER/hydra
cd /home/ecm5702/agent-work/20260915-rd25-evaluation/tools
echo "host=$(hostname) arch=$(uname -m) start=$(date -Is) dstools=$(git rev-parse --short HEAD) core=$(git -C $CORE rev-parse --short HEAD) arm=$ARM ckpt=$CKPT date=$D members=1-3 steps=24,120 precision=fp32"
mkdir -p "$OUT"
t0=$(date +%s)
python -m eval.cli predict --lane rd25_screen_20260915 --host atos_ag \
  --mode manual --checkpoint "$CKPT" --output-dir "$OUT" --members 1,2,3 --steps 24,120 --dates "$D" \
  --num-gpus-per-model 4 --bundle-dir /home/ecm5702/hpcperm/data/input_data/o320_o1280/humberto_20250926_20250930

rc=$?
echo "RD25_SCREEN arm=$ARM date=$D host=AG rc=$rc wall_s=$(( $(date +%s) - t0 ))"
exit $rc
