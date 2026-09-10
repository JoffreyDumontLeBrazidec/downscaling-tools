#!/bin/bash
#SBATCH --job-name=a2-verify
#SBATCH --qos=nf
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/logs/%x_%j.out
#
# Stages 5 and 6: rebuild the block manifest, then decide PASS or FAIL.
#
#   BLOCK=202601 sbatch verify.sbatch

source /home/ecm5702/dev/downscaling-tools-aifsens2-regen/tools/aifsens2_regen/env.sh
# The block name is taken from the first argument, or from BLOCK in the
# environment if no argument is given.  The argument form is the reliable one:
# the ECMWF sbatch wrapper does not always carry a caller's environment into the
# job, so "sbatch verify.sbatch pilot_20260101" is preferred over "BLOCK=... sbatch verify.sbatch".
BLOCK="${1:-${BLOCK:-}}"
: "${BLOCK:?give the block as the first argument, for example: sbatch verify.sbatch 202601}"

python -m aifsens2_regen.manifest --block "$BLOCK" ${CHECKSUMS:+--checksums}
python -m aifsens2_regen.verify_block --block "$BLOCK"
rc=$?
echo "VERIFY_TASK_RC=$rc block=$BLOCK"
exit $rc
