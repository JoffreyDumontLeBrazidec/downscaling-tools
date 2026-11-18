#!/bin/bash
# CorrDiff baseline: export fill on the CPU serial partition. QOS lrd_all_serial allows per user 10 submitted, 2 running,
# 8 CPUs and 30.8 GB in total, so one job runs one group of 8 parts (the same groups as cd_export.sbatch task T),
# one process per part. Fill is resumable (a row with a finite output is complete and skipped), so passes chain with
# singleton and can run beside the queued booster array:
#   for T in 0 1 2 3 0 1 2 3; do sbatch --dependency=singleton cd_export_serial.sbatch $T; done
#SBATCH --job-name=cd_export_ser
#SBATCH --account=DestE_340_26
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/DestE_340_26/users/jdumontl/sandbox/20261117-corrdiff/logs/%x_%j.out
set -euo pipefail
source /leonardo_work/DestE_340_26/users/jdumontl/sandbox/20261117-corrdiff/kit/cd_env.sh
OUT=$(cat $CD/E2V4_OUT_STORE); IN=$(cat $CD/E2V4_IN_STORE)
NP=32; T=${1:?group 0-3}
echo "[cd_export_ser] group $T start $(date -u +%FT%TZ)"
pids=()
for j in 0 1 2 3 4 5 6 7; do
  P=$((T * 8 + j))
  OMP_NUM_THREADS=1 python3 $CD_KIT/cd_export.py fill --out-store $OUT --in-store $IN --dest $CD_DATA --part $P --nparts $NP &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
(( fail == 0 )) || { echo "EXPORT_GROUP_FAIL $T: $fail of 8 parts failed or were cut"; exit 1; }
echo "EXPORT_GROUP_DONE $T $(date -u +%FT%TZ)"
