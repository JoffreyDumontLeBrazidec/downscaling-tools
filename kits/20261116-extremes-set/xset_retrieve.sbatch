#!/bin/bash
#SBATCH --job-name=xset_ret
#SBATCH --qos=nf
#SBATCH --time=2-00:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set/logs/%x-%j.out
# A set other than XSET-1 (S25, ...): Atos drops the caller's environment (SBATCH_EXPORT=NONE and the ECMWF sbatch
# wrapper), so exported XSET_* never arrive. Put them in xset_set.env in the directory you submit from (17 Nov 2026).
[[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/xset_set.env" ]] && source "$SLURM_SUBMIT_DIR/xset_set.env"
echo "[xset] set work ${XSET_WORK:-XSET-1 default} lists ${XSET_LISTS:-default} cache ${XSET_CACHE:-default}"
# Usage: sbatch --job-name=xset_ret_<i> xset_retrieve.sbatch <yyyymmdd> [<yyyymmdd> ...]   (dates in sequence; a DATE_DONE line each)
ulimit -c 0
module load ecmwf-toolbox >/dev/null 2>&1 || true
K=${XSET_WORK:-/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set}/downscaling-tools/kits/20261116-extremes-set
rc=0
for d in "$@"; do bash $K/xset_retrieve_date.sh $d || { echo "[ret] FAIL $d"; rc=1; }; done
echo "[ret] JOB_DONE rc=$rc $(date -u +%FT%TZ)"; exit $rc
