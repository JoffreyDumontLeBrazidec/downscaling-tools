#!/bin/bash
#SBATCH --job-name=qm_apply_s24
#SBATCH --qos=nf
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261125-s24-set/logs/%x-%j.out
# S24 (25 Nov 2026): qm/qm_apply.sbatch for the S24 set at leads 12..240. Same tables (B/qm, fitted on the 9 km training
# climatology, window w09 for September) and same qm_apply.py, but run from the S24 worktree so xset_common reads
# XSET_LEADS from xset_set.env (the baselines checkout has the six leads hard-coded and does not source the set file).
# Usage (from $XSET_WORK): sbatch --job-name=qm_apply_s24_<i> qm_apply_s24.sbatch <yyyymmdd> [...]
[[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/xset_set.env" ]] && source "$SLURM_SUBMIT_DIR/xset_set.env"
echo "[xset] set work ${XSET_WORK:-NONE} leads ${XSET_LEADS:-NONE}"
[[ -n "${XSET_WORK:-}" && -n "${XSET_LEADS:-}" ]] || { echo "FATAL no xset_set.env in the submit dir"; exit 2; }
set -o pipefail; ulimit -c 0; module load python3 >/dev/null 2>&1
B=/ec/res4/scratch/ecm5702/agent-work/20261117-baselines
K=$XSET_WORK/downscaling-tools/kits/20261117-baselines/qm
DS=/ec/ai/project/ai-ml/datasets
args=""; for d in "$@"; do args="$args --date $d"; done
python3 $K/qm_apply.py --work $XSET_WORK --qm $B/qm --gout $DS/downscaling-od-cf-enfh-0001-mars-o1280-2003-2023-24h-12h-v4.zarr $args \
  || { echo "[qm_apply] FAIL $(date -u +%FT%TZ)"; exit 1; }
echo "[qm_apply] JOB_DONE $* $(date -u +%FT%TZ)"
