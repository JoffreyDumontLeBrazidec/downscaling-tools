#!/bin/bash --login
# ============================================================================
# Tier-2 Integrated Gradients runner (GPU, .ds-ag venv, fp32).
#
# Storm-centered (probe-box) + global-mean attribution, per surface target,
# computed on REAL event bundles (never the val dataloader).
#
# USAGE:
#   sbatch interp/run_ig.sbatch <CKPT_PATH> <CKPT_ID> [smoke|full]
#
# Example (dev o96->o320 checkpoint, Franklin TC probe):
#   sbatch interp/run_ig.sbatch \
#     /home/ecm5702/scratch/aifs/checkpoint/59e40596023d4f72a1c5e2027805d7df/anemoi-by_epoch-epoch_260-step_300000.ckpt \
#     59e4_300k full
# ============================================================================

#SBATCH --job-name=interp_ig
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/interp_ig_%j.out
#SBATCH --error=/home/ecm5702/dev/jobscripts/outputs/interp_ig_%j.err

set -euxo pipefail

CKPT_PATH="${1:?Usage: sbatch run_ig.sbatch <CKPT_PATH> <CKPT_ID> [smoke|full]}"
CKPT_ID="${2:?Usage: sbatch run_ig.sbatch <CKPT_PATH> <CKPT_ID> [smoke|full]}"
MODE="${3:-full}"
OUT_ROOT="/home/ecm5702/perm/interp/${CKPT_ID}"
PY=/home/ecm5702/dev/.ds-ag/bin/python
BUNDLE_DIR=/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia

# ─── Environment ────────────────────────────────────────────────────────────
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
echo "Integrated Gradients (Tier-2) — ${CKPT_ID} (mode=${MODE})"
echo "Checkpoint: ${CKPT_PATH}"
echo "Output:     ${OUT_ROOT}/integrated_gradients"
echo "============================================================"

if [ "${MODE}" = "smoke" ]; then
  SIGMAS="5.0"
  FUNCTIONALS="global_mean,box,eye"
  IG_STEPS=8
  BOXES="--boxes franklin:auto"
else
  SIGMAS="0.1 1.0 5.0 20.0 80.0"
  FUNCTIONALS="global_mean,box,eye"
  IG_STEPS=32
  BOXES="--boxes franklin:auto"
fi

# init 20230826 + step048h => valid 2023-08-28 = Franklin major-hurricane peak (NW Atlantic)
$PY -m interp.integrated_gradients \
    --checkpoint "${CKPT_PATH}" \
    --output-dir "${OUT_ROOT}/integrated_gradients" \
    --bundle-dir "${BUNDLE_DIR}" \
    --dates 20230826 --members 01 --steps 048 \
    --sigmas ${SIGMAS} \
    --functionals "${FUNCTIONALS}" \
    ${BOXES} \
    --ig-steps ${IG_STEPS} \
    --device cuda --precision fp32 \
    --map-sigma 5.0 --eye-radius-km 150 --zoom-deg 12 --topk-zoom 4

# NB: render the PDF on the login node (interp.viz) so cartopy can fetch coastlines.

echo "============================================================"
echo "IG done for ${CKPT_ID}: ${OUT_ROOT}/integrated_gradients/integrated_gradients.json"
echo "============================================================"
