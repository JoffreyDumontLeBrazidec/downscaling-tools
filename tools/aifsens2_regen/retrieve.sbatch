#!/bin/bash
#SBATCH --job-name=a2-retrieve
#SBATCH --qos=nf
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/logs/%x_%A_%a.out
#
# Stage 1: fetch the initial-condition fields from MARS.
#
# The block and the group set are passed in the environment, and the array
# index selects which request the task runs:
#
#   BLOCK=202601 GROUPSET=atm  sbatch --array=1-16%2 retrieve.sbatch
#   BLOCK=202601 GROUPSET=wave sbatch --array=1-4%1  retrieve.sbatch
#
# The two group sets are submitted separately on purpose.  For January to May
# 2026 the atmospheric fields are mostly online while the wave fields are
# mostly on tape, and a tape mount that takes hours must not hold up the
# atmospheric requests that take minutes.  The two concurrency limits above add
# up to three, which is the ceiling this project puts on simultaneous MARS
# requests.
#
# Ask the code how many requests a block needs before choosing the array range:
#   python -m aifsens2_regen.retrieve --block 202601 --groupset atm --list

source /home/ecm5702/dev/downscaling-tools-aifsens2-regen/tools/aifsens2_regen/env.sh

# The block name is taken from the first argument, or from BLOCK in the
# environment if no argument is given.  The argument form is the reliable one:
# the ECMWF sbatch wrapper does not always carry a caller's environment into the
# job, so "sbatch retrieve.sbatch pilot_20260101" is preferred over "BLOCK=... sbatch retrieve.sbatch".
BLOCK="${1:-${BLOCK:-}}"
: "${BLOCK:?give the block as the first argument, for example: sbatch retrieve.sbatch 202601}"
GROUPSET="${2:-${GROUPSET:-}}"
: "${GROUPSET:?give the group set, atm or wave, as the second argument}"

python -m aifsens2_regen.retrieve \
  --block "$BLOCK" --groupset "$GROUPSET" --index "${SLURM_ARRAY_TASK_ID:-1}"
rc=$?
echo "RETRIEVE_TASK_RC=$rc block=$BLOCK groupset=$GROUPSET index=${SLURM_ARRAY_TASK_ID:-1}"
exit $rc
