#!/bin/bash
#SBATCH --job-name=qm_ext
#SBATCH --qos=nf
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-12
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261117-baselines/logs/%x-%A_%a.out
# QM baseline phase 1: training pairs of calendar month $SLURM_ARRAY_TASK_ID -> sample matrices (qm_extract.py).
set -o pipefail; ulimit -c 0; module load python3 >/dev/null 2>&1
B=/ec/res4/scratch/ecm5702/agent-work/20261117-baselines
K=$B/downscaling-tools/kits/20261117-baselines/qm
DS=/ec/ai/project/ai-ml/datasets
export OMP_NUM_THREADS=8
echo "[qm_ext] month $SLURM_ARRAY_TASK_ID kit $(git -C $B/downscaling-tools rev-parse --short HEAD) $(date -u +%FT%TZ)"
python3 $K/qm_extract.py --gin $DS/downscaling-od-cf-eefh-0001-mars-o320-2003-2023-24h-12h-v4.zarr \
  --gout $DS/downscaling-od-cf-enfh-0001-mars-o1280-2003-2023-24h-12h-v4.zarr --out $B/qm --month $SLURM_ARRAY_TASK_ID \
  --n-base ${QM_NBASE:-40} || { echo "[qm_ext] FAIL $(date -u +%FT%TZ)"; exit 1; }
echo "[qm_ext] JOB_DONE $(date -u +%FT%TZ)"
