#!/bin/bash
# The body of a station-head run, shared by train_score.sbatch (one GPU) and
# train_score_cpu.sbatch (the nf queue). It trains the full head for one target,
# trains the control head that sees only the interpolated AIFS input, and then
# scores both of them against the nearest-point rule and against the analysis at
# the nearest point, on the validation cases and, for reference, on the training
# cases.
#
# It reads its instructions from the environment: TARGET (2t, 2d or 10ff), TAG
# (the assembled dataset directory), RUN_PREFIX (the name the run folders start
# with), VAL_FROM and VAL_TO (the split by initialisation date), EPOCHS, PATIENCE
# and PIPELINE_TEST=1 to label every output as a pipeline test and not a result.
set -uo pipefail
S=/home/ecm5702/dev/downscaling-tools-station-head/tools/station_head/stage2a
TARGET=${TARGET:?set TARGET to 2t, 2d or 10ff}
TAG=${TAG:-firstcut}
RUN_PREFIX=${RUN_PREFIX:-firstcut}
VAL_FROM=${VAL_FROM:-2026-08-18}
VAL_TO=${VAL_TO:-}
EPOCHS=${EPOCHS:-60}
CACHE_GB=${CACHE_GB:-0}
PATIENCE=${PATIENCE:-8}
DATA=/home/ecm5702/scratch/eval/station_head_2a/datasets/${TARGET}_${TAG}
RUNS=/home/ecm5702/perm/station-head-adapter/head

EXTRA=""
[ -n "$VAL_TO" ] && EXTRA="$EXTRA --val-init-to $VAL_TO"
[ "${PIPELINE_TEST:-0}" = "1" ] && EXTRA="$EXTRA --pipeline-test"

module load ecmwf-toolbox
source /home/ecm5702/hpcperm/sandbox/20260902-hres-local-branch/activate.sh || exit 3
PY="env -u PYTHONPATH python -u"

RUN_FULL=${RUN_PREFIX}_${TARGET}
RUN_INT=${RUN_PREFIX}_${TARGET}_xinterp

echo "=== training the full head ($RUN_FULL) on $DATA ==="
$PY "$S/train.py" --target "$TARGET" --data-dir "$DATA" --run-id "$RUN_FULL" \
    --features both --val-init-from "$VAL_FROM" --epochs "$EPOCHS" --patience "$PATIENCE" --cache-gb "$CACHE_GB" $EXTRA || exit 4

echo "=== training the interpolated-input control head ($RUN_INT) ==="
$PY "$S/train.py" --target "$TARGET" --data-dir "$DATA" --run-id "$RUN_INT" \
    --features xinterp --val-init-from "$VAL_FROM" --epochs "$EPOCHS" --patience "$PATIENCE" --cache-gb "$CACHE_GB" $EXTRA || exit 5

echo "=== scoring on the validation cases ==="
$PY "$S/score.py" --run-dir "$RUNS/$RUN_FULL" --xinterp-run-dir "$RUNS/$RUN_INT" --on validation || exit 6

echo "=== scoring on the training cases, for reference ==="
$PY "$S/score.py" --run-dir "$RUNS/$RUN_FULL" --xinterp-run-dir "$RUNS/$RUN_INT" --on training || exit 7

echo "HEAD_RC=0"
