#!/bin/bash
#SBATCH --job-name=mi4_ret
#SBATCH --qos=nf
#SBATCH --account=ecaifs
#SBATCH --time=2-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261119-mi4k-s26/logs/%x-%j.out
# MI4-S26 retrieval of one or more start dates in sequence. Usage: sbatch --job-name=mi4_ret_<d> mi4_retrieve.sbatch <yyyymmdd> [...]
ulimit -c 0
module load ecmwf-toolbox >/dev/null 2>&1 || true
K=/ec/res4/scratch/ecm5702/agent-work/20261119-mi4k-s26/dt/kits/20261119-month-inference-4km
source $K/mi4_set.sh
export TMPDIR=$MI4_A/tmp; mkdir -p $TMPDIR
echo "[ret] host $(hostname) kit $K git $(git -C $K rev-parse --short HEAD) A=$MI4_A PAR=$MI4_PAR CAP=$MI4_CAP dates $*"
rc=0
for d in "$@"; do bash $K/mi4_retrieve_date.sh $d || { echo "[ret] FAIL $d"; rc=1; }; done
echo "[ret] JOB_DONE rc=$rc $(date -u +%FT%TZ)"; exit $rc
