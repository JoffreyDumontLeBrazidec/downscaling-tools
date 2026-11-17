#!/bin/bash
#SBATCH --job-name=qm_apply
#SBATCH --qos=nf
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261117-baselines/logs/%x-%j.out
# QM baseline phase 3: the QM row of one evaluation set for the given start dates.
# Usage: sbatch --job-name=qm_apply_<set>_<i> qm_apply.sbatch <SET_WORK_DIR> <yyyymmdd> [...]
#   SET_WORK_DIR = /ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set (XSET-1) or .../20261117-s25-set (S25)
set -o pipefail; ulimit -c 0; module load python3 >/dev/null 2>&1
B=/ec/res4/scratch/ecm5702/agent-work/20261117-baselines
K=$B/downscaling-tools/kits/20261117-baselines/qm
DS=/ec/ai/project/ai-ml/datasets
W=${1:?set work dir}; shift
args=""; for d in "$@"; do args="$args --date $d"; done
python3 $K/qm_apply.py --work $W --qm $B/qm --gout $DS/downscaling-od-cf-enfh-0001-mars-o1280-2003-2023-24h-12h-v4.zarr $args \
  || { echo "[qm_apply] FAIL $(date -u +%FT%TZ)"; exit 1; }
echo "[qm_apply] JOB_DONE $* $(date -u +%FT%TZ)"
