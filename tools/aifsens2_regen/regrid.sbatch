#!/bin/bash
#SBATCH --job-name=a2-regrid
#SBATCH --qos=nf
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/logs/%x_%j.out
#
# Stage 4: select the 68 lane variables and regrid them from N320 to O320.
#
#   BLOCK=202601 sbatch regrid.sbatch
#
# This is CPU work, so it runs on the nf quality of service and never touches a
# GPU.  It can therefore overlap with the next block's forecasts.

source /home/ecm5702/dev/downscaling-tools-aifsens2-regen/tools/aifsens2_regen/env.sh
# The block name is taken from the first argument, or from BLOCK in the
# environment if no argument is given.  The argument form is the reliable one:
# the ECMWF sbatch wrapper does not always carry a caller's environment into the
# job, so "sbatch regrid.sbatch pilot_20260101" is preferred over "BLOCK=... sbatch regrid.sbatch".
BLOCK="${1:-${BLOCK:-}}"
: "${BLOCK:?give the block as the first argument, for example: sbatch regrid.sbatch 202601}"

python -m aifsens2_regen.regrid --block "$BLOCK" \
  --workdir "${TMPDIR:-/tmp}/a2regrid_${BLOCK}_$$" \
  ${NATIVE_ROOT:+--native-root "$NATIVE_ROOT"} \
  ${DERIVED_ROOT:+--derived-root "$DERIVED_ROOT"}
rc=$?
echo "REGRID_TASK_RC=$rc block=$BLOCK"
exit $rc
