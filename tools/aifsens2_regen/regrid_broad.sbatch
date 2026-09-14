#!/bin/bash
#SBATCH --job-name=a2-regrid-broad
#SBATCH --qos=nf
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/ecm5702/agent-work/20260911-aifs-dataset-rebuild/outputs/%x_%j.out
#
# Stage 4, broad variant: select the 108 variables of the archive half of the
# campaign, regrid them from N320 to O320, re-encode the headers the way the
# archive encodes them, and turn the step-12 accumulations into six-hour
# windows.
#
#   sbatch regrid_broad.sbatch 202601
#
# This is CPU work, so it runs on the nf quality of service and never touches a
# GPU.  Every path can be overridden from the environment: NATIVE_ROOT says
# where the regenerated native files are read from, DERIVED_ROOT where the O320
# files are written, MANIFEST_ROOT where the JSON manifest goes, and START
# restricts the job to a single forecast start given as YYYYMMDD_HH.

source /home/ecm5702/dev/downscaling-tools-broad-regrid/tools/aifsens2_regen/env.sh

# The tools directory of this worktree, not the regeneration's.  env.sh only
# sets AIFSENS2_TOOLS if it is not already set, but it is sourced before this
# line has a chance to run, so the change of directory is repeated here.
export AIFSENS2_TOOLS=/home/ecm5702/dev/downscaling-tools-broad-regrid/tools
cd "$AIFSENS2_TOOLS" || exit 1
echo "tools (broad) : $AIFSENS2_TOOLS"

# The block name is taken from the first argument, or from BLOCK in the
# environment if no argument is given.  The argument form is the reliable one:
# the ECMWF sbatch wrapper does not always carry a caller's environment into
# the job.
BLOCK="${1:-${BLOCK:-}}"
: "${BLOCK:?give the block as the first argument, for example: sbatch regrid_broad.sbatch 202601}"

python -m aifsens2_regen.regrid_broad --block "$BLOCK" \
  --workdir "${TMPDIR:-/tmp}/a2regridbroad_${BLOCK}_$$" \
  ${NATIVE_ROOT:+--native-root "$NATIVE_ROOT"} \
  ${DERIVED_ROOT:+--derived-root "$DERIVED_ROOT"} \
  ${MANIFEST_ROOT:+--manifest-root "$MANIFEST_ROOT"} \
  ${START:+--start "$START"}
rc=$?
echo "REGRID_BROAD_TASK_RC=$rc block=$BLOCK"
exit $rc
