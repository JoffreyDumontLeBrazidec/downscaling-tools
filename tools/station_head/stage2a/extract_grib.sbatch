#!/bin/bash
#SBATCH --job-name=sh2a_extract
#SBATCH --qos=nf
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/ecm5702/agent-work/20260909-station-head-adapter/logs/%x_%A_%a.out
set -euo pipefail
module load python3 ecmwf-toolbox
NSLICES=${NSLICES:?}
python3 /home/ecm5702/dev/downscaling-tools-station-head/tools/station_head/stage2a/extract_grib.py \
  --slice "${SLURM_ARRAY_TASK_ID}" --nslices "${NSLICES}" --members "${MEMBERS:-10}"
