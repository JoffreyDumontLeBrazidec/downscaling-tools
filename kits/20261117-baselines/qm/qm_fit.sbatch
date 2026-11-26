#!/bin/bash
#SBATCH --job-name=qm_fit
#SBATCH --qos=nf
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-12
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261117-baselines/logs/%x-%A_%a.out
# QM baseline phase 2: quantile tables of the window centred on month $SLURM_ARRAY_TASK_ID (needs months m-1, m, m+1 extracted).
set -o pipefail; ulimit -c 0; module load python3 >/dev/null 2>&1
B=/ec/res4/scratch/ecm5702/agent-work/20261117-baselines
K=$B/downscaling-tools/kits/20261117-baselines/qm
python3 $K/qm_fit.py --out $B/qm --month $SLURM_ARRAY_TASK_ID || { echo "[qm_fit] FAIL $(date -u +%FT%TZ)"; exit 1; }
echo "[qm_fit] JOB_DONE $(date -u +%FT%TZ)"
