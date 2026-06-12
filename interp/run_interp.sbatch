#!/bin/bash --login
# ============================================================================
# Mechanistic interpretability experiment runner
#
# Runs all 4 Tier-1 tools on a single checkpoint:
#   1. Feature permutation importance
#   2. Conditioning ablation
#   3. Activation profiling
#   4. CKA layer similarity
#
# USAGE:
#   sbatch interp/run_interp.sbatch <CKPT_PATH> <CKPT_ID>
#
# Example:
#   sbatch interp/run_interp.sbatch \
#     /home/ecm5702/scratch/aifs/checkpoint/85884ee70d8749609087d205c0b15605/anemoi-by_epoch-epoch_144-step_189950.ckpt \
#     85884ee7_189k
# ============================================================================

#SBATCH --job-name=interp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --qos=ng
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/interp_%j.out
#SBATCH --error=/home/ecm5702/dev/jobscripts/outputs/interp_%j.err

set -euxo pipefail

# ─── Arguments ──────────────────────────────────────────────────────────────
CKPT_PATH="${1:?Usage: sbatch run_interp.sbatch <CKPT_PATH> <CKPT_ID>}"
CKPT_ID="${2:?Usage: sbatch run_interp.sbatch <CKPT_PATH> <CKPT_ID>}"
# Persistent perm storage: interp results live alongside plots,
# tied to a checkpoint id (e.g. 59e4_300k). Use --output-dir per-tool below.
OUT_ROOT="/home/ecm5702/perm/interp/${CKPT_ID}"

# ─── Environment ────────────────────────────────────────────────────────────
source /home/ecm5702/dev/.ds-dyn/bin/activate
export ANEMOI_BASE_SEED=756
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR=""
export DEV=/home/ecm5702/dev/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
cd /home/ecm5702/dev/downscaling-tools

echo "============================================================"
echo "Interpretability experiments for checkpoint: ${CKPT_ID}"
echo "Checkpoint path: ${CKPT_PATH}"
echo "Output: ${OUT_ROOT}"
echo "============================================================"

# ─── Tool 1: Feature Permutation ────────────────────────────────────────────
echo ""
echo ">>> [1/4] Feature permutation importance"
python -m interp.feature_permutation \
    --checkpoint "${CKPT_PATH}" \
    --output-dir "${OUT_ROOT}/feature_permutation" \
    --sigmas 0.1 1.0 5.0 20.0 80.0 \
    --n-repeats 5 \
    --n-batches 2 \
    --device cpu \
    --precision fp32

# ─── Tool 2: Conditioning Ablation ──────────────────────────────────────────
echo ""
echo ">>> [2/4] Conditioning ablation"
python -m interp.conditioning_ablation \
    --checkpoint "${CKPT_PATH}" \
    --output-dir "${OUT_ROOT}/conditioning_ablation" \
    --sigmas 0.1 0.5 1.0 5.0 20.0 80.0 \
    --n-batches 2 \
    --device cpu \
    --precision fp32

# ─── Tool 3: Activation Profiling ───────────────────────────────────────────
echo ""
echo ">>> [3/4] Activation profiling"
python -m interp.activation_profiling \
    --checkpoint "${CKPT_PATH}" \
    --output-dir "${OUT_ROOT}/activation_profiling" \
    --sigmas 0.01 0.1 0.5 1.0 2.0 5.0 10.0 20.0 40.0 80.0 \
    --n-batches 2 \
    --device cpu \
    --precision fp32

# ─── Tool 4: CKA Layer Similarity ───────────────────────────────────────────
echo ""
echo ">>> [4/4] CKA layer similarity"
python -m interp.cka_analysis \
    --checkpoint "${CKPT_PATH}" \
    --output-dir "${OUT_ROOT}/cka_analysis" \
    --sigmas 0.1 1.0 5.0 20.0 80.0 \
    --n-batches 2 \
    --device cpu \
    --precision fp32

echo ""
echo "============================================================"
echo "All interpretability tools completed for ${CKPT_ID}"
echo "Results in: ${OUT_ROOT}"
echo "============================================================"
