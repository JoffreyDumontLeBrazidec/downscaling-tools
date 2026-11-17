#!/bin/bash
# Extremes set XSET-1: run xset_predict_global.py for one row inside a 1-node x 4 GH200 allocation (model sharded over 4 ranks).
# Usage: bash xset_run.sh <ROW> <predict|test> <CASES "YYYYMMDD:lead,..."> [master port]
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/xset_runs.sh "${1:?row}"
export MODE=${2:?mode} CASES=${3:?cases}
[[ $MODE == test ]] && export OUTDIR=$XOUT/test/$1
source $HERE/xset_env.sh
export OMP_NUM_THREADS=8
S6=$(printf %06d "$CKPT_STEP")
ema=$(ls $RUN_DIR/*/inference-${EMA_TAG}-anemoi-by_step-epoch_*-step_$S6.ckpt 2>/dev/null | head -1 || true)
raw=$(ls $RUN_DIR/*/anemoi-by_step-epoch_*-step_$S6.ckpt 2>/dev/null | head -1 || true)
[[ -n "$ema" ]] || { echo "FATAL no $EMA_TAG file for step $CKPT_STEP under $RUN_DIR"; exit 7; }
[[ -n "$raw" ]] || { echo "FATAL no raw by-step file for step $CKPT_STEP under $RUN_DIR"; exit 7; }
export EMA_FILE=$ema
[[ "$OUTDIR" == /e/data1/jureap-data/ecmwf/users/jdumont/outputs/xset1_20261116/* ]] || { echo "FATAL OUTDIR $OUTDIR"; exit 9; }
echo "[xset] row=$1 mode=$MODE step=$CKPT_STEP ema=$ema sha256=$(sha256sum $ema | cut -c1-16) raw=$raw cases=$CASES members=$MEMBERS outdir=$OUTDIR kit=$(git -C $XK/downscaling-tools rev-parse --short HEAD) code=$CODE_ROOT@$(git -C $CODE_ROOT rev-parse --short HEAD) $(date -u +%FT%TZ)"
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1); export MASTER_ADDR MASTER_PORT=${4:-29591}
mkdir -p "$OUTDIR"
cd /tmp
t0=$(date +%s); rc=0
srun --kill-on-bad-exit=1 --label bash -c 'export CUDA_VISIBLE_DEVICES=0,1,2,3; exec python '"$XKIT"'/xset_predict_global.py' || rc=$?
echo "[xset] rc=$rc seconds=$(( $(date +%s) - t0 )) $(date -u +%FT%TZ)"
exit $rc
