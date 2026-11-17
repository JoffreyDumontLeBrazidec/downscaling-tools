#!/bin/bash
#SBATCH --job-name=xset_bin
#SBATCH --qos=nf
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set/logs/%x-%j.out
# A set other than XSET-1 (S25, ...): Atos drops the caller's environment (SBATCH_EXPORT=NONE and the ECMWF sbatch
# wrapper), so exported XSET_* never arrive. Put them in xset_set.env in the directory you submit from (17 Nov 2026).
[[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/xset_set.env" ]] && source "$SLURM_SUBMIT_DIR/xset_set.env"
echo "[xset] set work ${XSET_WORK:-XSET-1 default} lists ${XSET_LISTS:-default} cache ${XSET_CACHE:-default}"
# Input member stores of the given start dates. Usage: sbatch --job-name=xset_bin_<i> xset_build_in.sbatch <yyyymmdd> [...]
ulimit -c 0; module load python3 ecmwf-toolbox >/dev/null 2>&1
A=${XSET_WORK:-/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set}
args=""; for d in "$@"; do args="$args --date $d"; done
echo "[bin] $args $(date -u +%FT%TZ)"
python3 $A/downscaling-tools/kits/20261116-extremes-set/xset_build_in.py \
  --gin /ec/ai/project/ai-ml/datasets/downscaling-od-cf-eefh-0001-mars-o320-2003-2023-24h-12h-v4.zarr \
  --grib $A/grib --out $A/stores $args || { echo "[bin] BUILD_FAIL $(date -u +%FT%TZ)"; exit 1; }
echo "[bin] JOB_DONE $(date -u +%FT%TZ)"
