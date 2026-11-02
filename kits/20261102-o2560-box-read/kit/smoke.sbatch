#!/bin/bash
# o2560 Europe-box read (2 Nov 2026): GPU smoke, NOT the read. For each of A100, B24, B50 and B24X (B24 weights on A's code):
# one date (20241101), lead +24 h, 2 members, rule-31 seeds, the read's sampler and chunking. Checks: finite fields, per-variable
# ranges (RANGE / SANITY lines), seconds per draw, torch peak memory per draw, nvidia-smi peak, MaxRSS (sacct after the job).
# B24 and B24X write the same seeds through two code pins: smoke_compare.py prints their max |difference| per variable.
# Usage: sbatch smoke.sbatch. Log ends O2560_SMOKE_DONE (or O2560_SMOKE_FAIL with the failing rows).
#SBATCH --job-name=o25_smoke
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/logs/%x-%j.out
set -o pipefail
ulimit -c 0
KITK=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/kit
LOGD=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/logs
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 10 > $LOGD/gpumem_smoke_${SLURM_JOB_ID}.csv 2>/dev/null &
bad=""
for row in B24 A100 B50 B24X; do
  ( source $KITK/rows.sh $row && source $KITK/env.sh && export MODE=smoke MEMBERS=2 OMP_NUM_THREADS=8 \
      OUTDIR=$OUT/smoke/$row CASES=20241101:24 && cd /tmp && python $KITK/o2560_box_predict.py ) > $LOGD/smoke_${row}_${SLURM_JOB_ID}.log 2>&1
  rc=$?; echo "[smoke] $row rc=$rc $(grep -c '^.*DRAW ' $LOGD/smoke_${row}_${SLURM_JOB_ID}.log) draws; $(grep -h 'DRAW ' $LOGD/smoke_${row}_${SLURM_JOB_ID}.log | tail -1); sanity flags $(grep -c 'SANITY ' $LOGD/smoke_${row}_${SLURM_JOB_ID}.log) $(date -u +%FT%TZ)"
  grep -q O2560_PREDICT_DONE $LOGD/smoke_${row}_${SLURM_JOB_ID}.log && [[ $rc == 0 ]] || { bad="$bad $row"; tail -30 $LOGD/smoke_${row}_${SLURM_JOB_ID}.log; }
done
( source $KITK/rows.sh B24 && source $KITK/env.sh && python $KITK/smoke_compare.py $OUT/smoke/B24/predictions_20241101_step024.nc $OUT/smoke/B24X/predictions_20241101_step024.nc ) || bad="$bad compare"
echo "[smoke] nvidia-smi peak MiB: $(sort -t, -k2 -n $LOGD/gpumem_smoke_${SLURM_JOB_ID}.csv | tail -1)"
[[ -z "$bad" ]] && echo "[smoke] O2560_SMOKE_DONE $(date -u +%FT%TZ)" || { echo "[smoke] O2560_SMOKE_FAIL$bad $(date -u +%FT%TZ)"; exit 1; }
