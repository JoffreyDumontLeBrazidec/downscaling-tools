#!/bin/bash
#SBATCH --job-name=a2-forecast
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/logs/%x_%j.out
#
# Stage 3: the forecasts, on exactly one A100.
#
#   BLOCK=202601 sbatch run_forecasts.sbatch
#
# One GPU, never an array, because the project allows only one GPU in use at a
# time.  The job loads the checkpoint once and then walks every start and every
# member of the block, so the twenty-five second load is paid once rather than
# once per forecast.
#
# The job is resumable exactly as it stands: a member whose output file already
# passes validation is skipped, so resubmitting the same command after an
# interruption continues where it stopped.
#
# Sizing the time limit: a member costs roughly six seconds of forecast plus
# about twenty seconds of reading, validating and writing.  A monthly block of
# 62 starts is 620 members, so twelve hours leaves a wide margin.  Set
# SBATCH_TIMELIMIT or edit --time for a longer block.

source /home/ecm5702/dev/downscaling-tools-aifsens2-regen/tools/aifsens2_regen/env.sh
# The block name is taken from the first argument, or from BLOCK in the
# environment if no argument is given.  The argument form is the reliable one:
# the ECMWF sbatch wrapper does not always carry a caller's environment into the
# job, so "sbatch run_forecasts.sbatch pilot_20260101" is preferred over "BLOCK=... sbatch run_forecasts.sbatch".
BLOCK="${1:-${BLOCK:-}}"
: "${BLOCK:?give the block as the first argument, for example: sbatch run_forecasts.sbatch 202601}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m aifsens2_regen.run_forecasts --block "$BLOCK" \
  ${MEMBERS_ROOT:+--members-root "$MEMBERS_ROOT"} \
  ${NATIVE_ROOT:+--native-root "$NATIVE_ROOT"}
rc=$?
echo "FORECAST_TASK_RC=$rc block=$BLOCK"
exit $rc
