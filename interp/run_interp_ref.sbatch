#!/bin/bash --login
#SBATCH --job-name=interp_ref
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --gres=gpu:ga100:1
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/interp_ref_%j.out
#SBATCH --error=/home/ecm5702/dev/jobscripts/outputs/interp_ref_%j.err
set -euxo pipefail
CKPT_PATH="${1:?CKPT}"; CKPT_ID="${2:?ID}"
OUT_ROOT="/home/ecm5702/perm/interp/${CKPT_ID}"
source /home/ecm5702/dev/downscaling-tools/interp/env_ref.sh
cd /home/ecm5702/dev/downscaling-tools

# Tier-1 reads REAL event bundles (no val_dataloader). 4 members @ step048 -> batch 4.
# denoise_at_sigma loops batch-1 internally (model mis-assembles input at batch>1).
BUNDLE_DIR=/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia
BUNDLE_ARGS="--bundle-dir ${BUNDLE_DIR} --dates 20230826 --members 01 02 03 04 --steps 048"
DEV="--device cuda --precision fp32"

echo ">>> [1/4] feature permutation"
python -m interp.feature_permutation --checkpoint "${CKPT_PATH}" \
    --output-dir "${OUT_ROOT}/feature_permutation" \
    --sigmas 0.1 1.0 5.0 20.0 80.0 --n-repeats 5 ${BUNDLE_ARGS} ${DEV}

echo ">>> [2/4] conditioning ablation"
python -m interp.conditioning_ablation --checkpoint "${CKPT_PATH}" \
    --output-dir "${OUT_ROOT}/conditioning_ablation" \
    --sigmas 0.1 0.5 1.0 5.0 20.0 80.0 ${BUNDLE_ARGS} ${DEV}

echo ">>> [3/4] activation profiling"
python -m interp.activation_profiling --checkpoint "${CKPT_PATH}" \
    --output-dir "${OUT_ROOT}/activation_profiling" \
    --sigmas 0.01 0.1 0.5 1.0 2.0 5.0 10.0 20.0 40.0 80.0 ${BUNDLE_ARGS} ${DEV}

echo ">>> [4/4] cka layer similarity"
python -m interp.cka_analysis --checkpoint "${CKPT_PATH}" \
    --output-dir "${OUT_ROOT}/cka" \
    --sigmas 0.1 1.0 5.0 20.0 80.0 ${BUNDLE_ARGS} ${DEV}

echo "tier-1 complete for ${CKPT_ID}"
