#!/bin/bash
# MI-S25: run mi_predict_global.py for one row inside a 1-node x 4 GH200 allocation (model sharded over 4 ranks).
# Environment = the XSET-1 environment file (modules, .ds venv, code and datasets sha checks, chunks 1/1/1); OUTPUT differs.
# Usage: bash mi_run.sh <ROW> <predict|xcheck|dry> <CASES "YYYYMMDD:lead,..."> [master port]
# dry (26 Nov 2026): login node, CPU, one process: everything but the draws (trainer, class, weights, anchor, inputs of every case).
# The kit is the checkout this file lives in (26 Nov 2026: was the live checkout $MK/downscaling-tools in every case; the same path
# for that checkout, so its rows are unchanged).
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/mi_runs.sh "${1:?row}"
KROOT=$(cd "$HERE/../../.." && pwd); MKIT=$KROOT/kits/20261117-month-inference
export MODE=${2:?mode} CASES=${3:?cases}
if [[ $MODE == dry ]]; then
  export OUTDIR=$MOUT/dry/$1 SEED_BASE=${SEED_BASE:-1234}
elif [[ $MODE == xcheck ]]; then
  export OUTDIR=$MOUT/xcheck/$1 SEED_BASE=1234
  export XCHECK_IN=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set/stores/in
  export XCHECK_ROWS=/e/data1/jureap-data/ecmwf/users/jdumont/outputs/xset1_20261116/rows/$1
else
  export SEED_BASE=${SEED_BASE:-1234}
fi
source $KROOT/kits/20261116-extremes-set/jupiter/xset_env.sh
export OUTPUT=$MOUT/_anemoi_output; mkdir -p "$OUTPUT"
export OMP_NUM_THREADS=8
S6=$(printf %06d "$CKPT_STEP")
ema=${EMA_FILE:-$(ls $RUN_DIR/*/inference-${EMA_TAG}-anemoi-by_step-epoch_*-step_$S6.ckpt 2>/dev/null | head -1 || true)}
raw=${RAW_FILE:-$(ls $RUN_DIR/*/anemoi-by_step-epoch_*-step_$S6.ckpt 2>/dev/null | head -1 || true)}
[[ -n "$ema" ]] || { echo "FATAL no $EMA_TAG file for step $CKPT_STEP under $RUN_DIR"; exit 7; }
[[ -n "$raw" ]] || { echo "FATAL no raw by-step file for step $CKPT_STEP under $RUN_DIR"; exit 7; }
export EMA_FILE=$ema RAW_FILE=$raw
if [[ ${SAMPLER_MODE:-} == meanflow ]]; then   # one EMA file only; the trainer warm start is the converted warm file (sha pinned)
  n_ema=$(ls $RUN_DIR/*/inference-${EMA_TAG}-anemoi-by_step-epoch_*-step_$S6.ckpt 2>/dev/null | wc -l)
  [[ $n_ema == 1 ]] || { echo "FATAL $n_ema $EMA_TAG files for step $CKPT_STEP under $RUN_DIR/*/"; exit 7; }
  wsha=$(sha256sum "$raw" | cut -d' ' -f1)
  [[ "$wsha" == "$WARM_SHA" ]] || { echo "FATAL warm file sha256 $wsha != $WARM_SHA ($raw)"; exit 7; }
fi
[[ "$OUTDIR" == $MOUT/* ]] || { echo "FATAL OUTDIR $OUTDIR"; exit 9; }
echo "[mi] row=$1 mode=$MODE step=$CKPT_STEP ema=$ema sha256=$(sha256sum $ema | cut -c1-16) raw=$raw seed_base=$SEED_BASE cases=$CASES members=$MEMBERS outdir=$OUTDIR kit=$KROOT@$(git -C $KROOT rev-parse --short HEAD) code=$CODE_ROOT@$(git -C $CODE_ROOT rev-parse --short HEAD) $(date -u +%FT%TZ)"
cd /tmp
t0=$(date +%s); rc=0
if [[ $MODE == dry ]]; then
  unset SLURM_NTASKS SLURM_PROCID SLURM_LOCALID
  CUDA_VISIBLE_DEVICES="" nice -n 10 timeout 3600 python $MKIT/jupiter/mi_predict_global.py || rc=$?
else
  MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1); export MASTER_ADDR MASTER_PORT=${4:-29591}
  mkdir -p "$OUTDIR"
  srun --kill-on-bad-exit=1 --label bash -c 'export CUDA_VISIBLE_DEVICES=0,1,2,3; exec python '"$MKIT"'/jupiter/mi_predict_global.py' || rc=$?
fi
echo "[mi] rc=$rc seconds=$(( $(date +%s) - t0 )) $(date -u +%FT%TZ)"
exit $rc
