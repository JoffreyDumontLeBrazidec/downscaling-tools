#!/bin/bash --login
#SBATCH --job-name=ig_ref
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --gres=gpu:ga100:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/ig_ref_%j.out
#SBATCH --error=/home/ecm5702/dev/jobscripts/outputs/ig_ref_%j.err
set -euxo pipefail
CKPT_PATH="${1:?CKPT}"; CKPT_ID="${2:?ID}"; MODE="${3:-full}"
OUT_ROOT="/home/ecm5702/perm/interp/${CKPT_ID}"
BUNDLE_DIR=/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia
source /home/ecm5702/dev/downscaling-tools/interp/env_ref.sh
cd /home/ecm5702/dev/downscaling-tools
if [ "${MODE}" = "smoke" ]; then SIGMAS="5.0"; IG_STEPS=8; else SIGMAS="0.1 1.0 5.0 20.0 80.0"; IG_STEPS=32; fi
python -m interp.integrated_gradients \
    --checkpoint "${CKPT_PATH}" --output-dir "${OUT_ROOT}/integrated_gradients" \
    --bundle-dir "${BUNDLE_DIR}" --dates 20230826 --members 01 --steps 048 \
    --sigmas ${SIGMAS} --functionals "global_mean,box,eye" --boxes franklin:auto \
    --ig-steps ${IG_STEPS} --device cuda --precision fp32 \
    --map-sigma 5.0 --eye-radius-km 150 --zoom-deg 12 --topk-zoom 4
echo "IG ${MODE} done: ${OUT_ROOT}/integrated_gradients/integrated_gradients.json"
