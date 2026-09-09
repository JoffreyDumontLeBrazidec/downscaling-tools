#!/bin/bash --login
#SBATCH --job-name=sh2a_score
#SBATCH --qos=nf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ecm5702/agent-work/20260909-station-head-adapter/logs/%x_%j.out
# Score an already trained station head and its interpolated-input control head
# against the nearest-point rule and the analysis, on the validation and on the
# training cases. This is the same scoring train_score_body.sh runs at its end;
# it exists as a separate job because on 2026-09-09 the pipeline-test job
# 34925896 (nf queue) ended at exactly 00:20:00 with exit code 0 after its second
# training run and before its scoring step, for a reason that was never found (a
# 30-minute sleep probe, job 34933904, survived on the same queue, so it is not a
# queue-wide cap). run_stage2a_head.sh submits this job with afterany on each
# training job, so the scores are written whether or not the training job ran
# its own scoring step; score.py is idempotent and only reads checkpoints.
#
#   sbatch --export=ALL,TARGET=2t,RUN_PREFIX=firstcut_20260909 score_only.sbatch
set -uo pipefail
S=/home/ecm5702/dev/downscaling-tools-station-head/tools/station_head/stage2a
TARGET=${TARGET:?set TARGET to 2t, 2d or 10ff}
RUN_PREFIX=${RUN_PREFIX:-firstcut}
RUNS=/home/ecm5702/perm/station-head-adapter/head
RUN_FULL=${RUN_PREFIX}_${TARGET}
RUN_INT=${RUN_PREFIX}_${TARGET}_xinterp
for d in "$RUNS/$RUN_FULL" "$RUNS/$RUN_INT"; do
  [ -f "$d/head.pt" ] || { echo "missing checkpoint $d/head.pt; the training job did not finish"; exit 2; }
done
module load ecmwf-toolbox
source /home/ecm5702/hpcperm/sandbox/20260902-hres-local-branch/activate.sh || exit 3
PY="env -u PYTHONPATH python -u"
echo "=== scoring on the validation cases ==="
$PY "$S/score.py" --run-dir "$RUNS/$RUN_FULL" --xinterp-run-dir "$RUNS/$RUN_INT" --on validation || exit 6
echo "=== scoring on the training cases, for reference ==="
$PY "$S/score.py" --run-dir "$RUNS/$RUN_FULL" --xinterp-run-dir "$RUNS/$RUN_INT" --on training || exit 7
echo "SCORE_RC=0"
