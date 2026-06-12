#!/bin/bash --login
#SBATCH --job-name=patch_ref
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --gres=gpu:ga100:1
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/patch_ref_%j.out
#SBATCH --error=/home/ecm5702/dev/jobscripts/outputs/patch_ref_%j.err
set -euxo pipefail
CKPT_PATH="${1:?CKPT}"; CKPT_ID="${2:?ID}"
BUNDLE_NC="${3:-/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia/eefo_o96_0001_date20230826_time0000_mem01_step072h_input_bundle.nc}"
OUT_ROOT="/home/ecm5702/perm/interp/${CKPT_ID}"
source /home/ecm5702/dev/downscaling-tools/interp/env_ref.sh
cd /home/ecm5702/dev/downscaling-tools
python -m interp.activation_patching --checkpoint "${CKPT_PATH}" \
    --output-dir "${OUT_ROOT}/activation_patching" \
    --sigmas 0.1 1.0 5.0 20.0 80.0 --modes residual grid_region stage \
    --corruption zero_lres --region-radius-deg 8.0 --storm-label franklin \
    --bundle "${BUNDLE_NC}" --device cuda --precision fp32
echo "patching done: ${OUT_ROOT}/activation_patching/activation_patching.json"
