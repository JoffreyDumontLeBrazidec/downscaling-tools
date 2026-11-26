#!/bin/bash
#SBATCH --job-name=s24_ret
#SBATCH --qos=nf
#SBATCH --time=2-00:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261125-s24-set/logs/%x-%j.out
# Usage (from $XSET_WORK, which holds xset_set.env): sbatch --job-name=s24_ret_<d> s24_retrieve_truth.sbatch <yyyymmdd> [...]
[[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/xset_set.env" ]] && source "$SLURM_SUBMIT_DIR/xset_set.env"
echo "[xset] set work ${XSET_WORK:-NONE} leads ${XSET_LEADS:-NONE}"
[[ -n "${XSET_WORK:-}" ]] || { echo "FATAL no xset_set.env in the submit dir"; exit 2; }
ulimit -c 0; module load ecmwf-toolbox >/dev/null 2>&1 || true
rc=0
for d in "$@"; do bash $XSET_WORK/downscaling-tools/kits/20261117-baselines/s24/s24_retrieve_truth.sh $d || { echo "[ret] FAIL $d"; rc=1; }; done
echo "[ret] JOB_DONE rc=$rc $(date -u +%FT%TZ)"; exit $rc
