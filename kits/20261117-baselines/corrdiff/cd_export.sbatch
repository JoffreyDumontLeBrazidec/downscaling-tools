#!/bin/bash
# CorrDiff baseline: export fill, one array task per part (I/O bound; 8 parts per task). Submit after `cd_export.py init`.
#   sbatch --array=0-3 cd_export.sbatch            (4 tasks x 8 parts = 32 parts)
#SBATCH --job-name=cd_export
#SBATCH --account=DestE_340_26
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --output=/leonardo_work/DestE_340_26/users/jdumontl/sandbox/20261117-corrdiff/logs/%x_%A_%a.out
set -euo pipefail
source /leonardo_work/DestE_340_26/users/jdumontl/sandbox/20261117-corrdiff/kit/cd_env.sh
OUT=$(cat $CD/E2V4_OUT_STORE); IN=$(cat $CD/E2V4_IN_STORE)
NP=32; T=${SLURM_ARRAY_TASK_ID}

