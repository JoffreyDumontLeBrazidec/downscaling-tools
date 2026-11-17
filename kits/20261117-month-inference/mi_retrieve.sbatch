#!/bin/bash
#SBATCH --job-name=mi_ret
#SBATCH --qos=nf
#SBATCH --time=2-00:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261117-month-inference/logs/%x-%j.out
# Usage: sbatch --job-name=mi_ret_<i> mi_retrieve.sbatch <yyyymmdd> [<yyyymmdd> ...]   (dates in sequence; a DATE_DONE line each)
ulimit -c 0
module load ecmwf-toolbox >/dev/null 2>&1 || true
K=/ec/res4/scratch/ecm5702/agent-work/20261117-month-inference/downscaling-tools/kits/20261117-month-inference
rc=0
for d in "$@"; do bash $K/mi_retrieve_date.sh $d || { echo "[ret] FAIL $d"; rc=1; }; done
echo "[ret] JOB_DONE rc=$rc $(date -u +%FT%TZ)"; exit $rc
