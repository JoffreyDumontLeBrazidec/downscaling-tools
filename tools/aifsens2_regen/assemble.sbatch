#!/bin/bash
#SBATCH --job-name=a2-assemble
#SBATCH --qos=nf
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/logs/%x_%j.out
#
# Stage 2: build one initial-condition file per start and member.
#
#   BLOCK=202601 sbatch assemble.sbatch
#
# Exits non-zero if any member could not be built.  The members that could be
# built are still written, and the gap is recorded in the start's manifest.

source /home/ecm5702/dev/downscaling-tools-aifsens2-regen/tools/aifsens2_regen/env.sh
# The block name is taken from the first argument, or from BLOCK in the
# environment if no argument is given.  The argument form is the reliable one:
# the ECMWF sbatch wrapper does not always carry a caller's environment into the
# job, so "sbatch assemble.sbatch pilot_20260101" is preferred over "BLOCK=... sbatch assemble.sbatch".
BLOCK="${1:-${BLOCK:-}}"
: "${BLOCK:?give the block as the first argument, for example: sbatch assemble.sbatch 202601}"

python -m aifsens2_regen.assemble --block "$BLOCK" ${MEMBERS_ROOT:+--members-root "$MEMBERS_ROOT"}
rc=$?
echo "ASSEMBLE_TASK_RC=$rc block=$BLOCK"
exit $rc
