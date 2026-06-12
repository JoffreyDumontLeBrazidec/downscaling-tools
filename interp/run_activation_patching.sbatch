#!/bin/bash --login
# ============================================================================
# Tier-3 causal interpretability: activation patching
#   modes: residual (per-block localizer) | grid_region (storm subset) | stage
#   Input: REAL TC event bundle (never val_dataloader).
#
# USAGE:
#   sbatch interp/run_activation_patching.sbatch <CKPT_PATH> <CKPT_ID> [BUNDLE_NC]
# ============================================================================
#SBATCH --job-name=act_patch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gh200:1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=ng
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/act_patch_%j.out
#SBATCH --error=/home/ecm5702/dev/jobscripts/outputs/act_patch_%j.err
set -euxo pipefail

CKPT_PATH="${1:?Usage: sbatch run_activation_patching.sbatch <CKPT_PATH> <CKPT_ID> [BUNDLE_NC]}"
CKPT_ID="${2:?Usage: sbatch run_activation_patching.sbatch <CKPT_PATH> <CKPT_ID> [BUNDLE_NC]}"
BUNDLE_NC="${3:-/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia/eefo_o96_0001_date20230826_time0000_mem01_step072h_input_bundle.nc}"
OUT_ROOT="/home/ecm5702/perm/interp/${CKPT_ID}"

# GH200 is aarch64 -> must use the .ds-ag venv (torch 2.7 / cu128)
source /home/ecm5702/dev/.ds-ag/bin/activate
export ANEMOI_BASE_SEED=756 TORCH_COMPILE_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR="" DEV=/home/ecm5702/dev/ OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/ INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
cd /home/ecm5702/dev/downscaling-tools

echo ">>> activation patching: ${CKPT_ID}  bundle=${BUNDLE_NC}"
python -m interp.activation_patching \
    --checkpoint "${CKPT_PATH}" \
    --output-dir "${OUT_ROOT}/activation_patching" \
    --sigmas 0.1 1.0 5.0 20.0 80.0 \
    --modes residual grid_region stage \
    --corruption zero_lres \
    --region-radius-deg 8.0 \
    --storm-label franklin \
    --bundle "${BUNDLE_NC}" \
    --device cuda --precision fp32

echo ">>> rendering PDF"
python -m interp.viz --run-dir "${OUT_ROOT}" --tools activation_patching

echo "Done: ${OUT_ROOT}/activation_patching/activation_patching.json"
echo "Plot: ${OUT_ROOT}/plots/activation_patching.pdf"
