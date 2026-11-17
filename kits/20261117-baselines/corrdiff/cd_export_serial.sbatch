#!/bin/bash
# CorrDiff baseline: export fill on the CPU serial partition, one part per array task (same 32 parts as cd_export.sbatch).
# Fill is resumable (rows with a finite output are skipped), so it can run beside a queued cd_export.sbatch array:
# whichever starts first writes a row, the other skips it (identical content if both write the same row).
#   sbatch --array=0-31 cd_export_serial.sbatch
#SBATCH --job-name=cd_export_serial
#SBATCH --account=DestE_340_26
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30000M
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/DestE_340_26/users/jdumontl/sandbox/20261117-corrdiff/logs/%x_%A_%a.out
set -euo pipefail
source /leonardo_work/DestE_340_26/users/jdumontl/sandbox/20261117-corrdiff/kit/cd_env.sh
OUT=$(cat $CD/E2V4_OUT_STORE); IN=$(cat $CD/E2V4_IN_STORE)
NP=32; P=${SLURM_ARRAY_TASK_ID}
OMP_NUM_THREADS=4 python3 $CD_KIT/cd_export.py fill --out-store $OUT --in-store $IN --dest $CD_DATA --part $P --nparts $NP
echo "EXPORT_PART_DONE $P"
