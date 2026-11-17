#!/bin/bash
#SBATCH --job-name=xset_bin
#SBATCH --qos=nf
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set/logs/%x-%j.out
# Input member stores of the given start dates. Usage: sbatch --job-name=xset_bin_<i> xset_build_in.sbatch <yyyymmdd> [...]
ulimit -c 0; module load python3 ecmwf-toolbox >/dev/null 2>&1
A=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set
args=""; for d in "$@"; do args="$args --date $d"; done
echo "[bin] $args $(date -u +%FT%TZ)"
python3 $A/downscaling-tools/kits/20261116-extremes-set/xset_build_in.py \
  --gin /ec/ai/project/ai-ml/datasets/downscaling-od-cf-eefh-0001-mars-o320-2003-2023-24h-12h-v4.zarr \
  --grib $A/grib --out $A/stores $args || { echo "[bin] BUILD_FAIL $(date -u +%FT%TZ)"; exit 1; }
echo "[bin] JOB_DONE $(date -u +%FT%TZ)"
