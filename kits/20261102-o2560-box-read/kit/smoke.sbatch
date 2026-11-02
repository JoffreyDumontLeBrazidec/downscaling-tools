#!/bin/bash
# o2560 Europe-box read (2 Nov 2026): GPU smoke, NOT the read. One GH200, the read's sampler, rule-31 seeds, date 20241101.
#   B24     2 members at +24 h and +48 h            (seed checks; finite fields; RANGE and SANITY lines; s and GiB per draw)
#   B24_m1  1 member at +24 h, separate process     (must reproduce B24 member 1)
#   B24_c8  1 member at +24 h, ANEMOI_INFERENCE_NUM_CHUNKS=8 (the read uses 16; difference reported)
#   A100, B50, B24X (B24 weights on A's code, crop off): 2 members at +24 h
# Then seed_check.py (SEED_CHECK_PASS or SEED_CHECK_FAIL) and smoke_compare.py B24 vs B24X (information only), nvidia-smi
# peak; MaxRSS from sacct after the job. Usage: sbatch smoke.sbatch. Log ends O2560_SMOKE_DONE or O2560_SMOKE_FAIL.
#SBATCH --job-name=o25_smoke
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=02:30:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/logs/%x-%j.out
set -o pipefail
ulimit -c 0
KITK=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/kit
LOGD=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/logs
SM=/e/data1/jureap-data/ecmwf/users/jdumont/outputs/o2560_box_read_20261102/smoke_${SLURM_JOB_ID}
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 10 > $LOGD/gpumem_smoke_${SLURM_JOB_ID}.csv 2>/dev/null &
echo "[smoke] start $(date -u +%FT%TZ) node $(hostname) outputs $SM"
bad=""
run() {   # run <label> <row> <members> <cases> [chunks]
  local lab=$1 row=$2 mem=$3 cases=$4 ch=${5:-16} lg=$LOGD/smoke_${1}_${SLURM_JOB_ID}.log
  ( source $KITK/rows.sh $row && export ANEMOI_INFERENCE_NUM_CHUNKS=$ch && source $KITK/env.sh && export MODE=smoke MEMBERS=$mem \
      OMP_NUM_THREADS=8 OUTDIR=$SM/$lab CASES=$cases && cd /tmp && python $KITK/o2560_box_predict.py ) > $lg 2>&1
  local rc=$?
  echo "[smoke] $lab rc=$rc draws=$(grep -c 'DRAW case' $lg) chunks=$ch sanity_flags=$(grep -c 'SANITY ' $lg) $(date -u +%FT%TZ)"
  grep -h 'DRAW case' $lg | sed 's/^/[smoke]   /'
  grep -h 'SANITY ' $lg | sed 's/^/[smoke]   /'
  grep -q O2560_PREDICT_DONE $lg && [[ $rc == 0 ]] || { bad="$bad $lab"; tail -30 $lg; }
}
run B24 B24 2 20241101:24,20241101:48
run B24_m1 B24 1 20241101:24
run B24_c8 B24 1 20241101:24 8
run A100 A100 2 20241101:24
run B50 B50 2 20241101:24
run B24X B24X 2 20241101:24
sc=$( ( source $KITK/rows.sh B24 && source $KITK/env.sh && python $KITK/seed_check.py $SM/B24 $SM/B24_m1 $SM/B24_c8 ) 2>&1 )
echo "$sc" | sed 's/^/[smoke] /'
grep -q "^SEED_CHECK_PASS" <<< "$sc" || bad="$bad seed_check"
( source $KITK/rows.sh B24 && source $KITK/env.sh && python $KITK/smoke_compare.py $SM/B24/predictions_20241101_step024.nc $SM/B24X/predictions_20241101_step024.nc ) 2>&1 | sed 's/^/[smoke] /'
echo "[smoke] nvidia-smi peak MiB: $(sort -t, -k2 -n $LOGD/gpumem_smoke_${SLURM_JOB_ID}.csv | tail -1)"
[[ -z "$bad" ]] && echo "[smoke] O2560_SMOKE_DONE $(date -u +%FT%TZ)" || { echo "[smoke] O2560_SMOKE_FAIL$bad $(date -u +%FT%TZ)"; exit 1; }
