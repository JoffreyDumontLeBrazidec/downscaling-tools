#!/bin/bash
#SBATCH --job-name=a2-retrieve-chunk
#SBATCH --qos=nf
#SBATCH --time=1-00:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/logs/%x_%A_%a.out
#
# Stage 1, chunked: fetch ONE group of ONE block at ONE input time, in
# contiguous ranges of a few days rather than as a single request covering the
# whole month.
#
# Use this only when the archive will not schedule the grouped monthly request
# for that group.  A request naming a whole month is sometimes never assigned a
# server task at all, while the same fields split into a few days each are
# served promptly.
#
# WARNING.  Never run this while the grouped monthly request for the same
# block, group and time of day is still queued or running.  Both write into the
# same block directory, stage two reads every GRIB file it finds there, and the
# same fields would then be present twice in its input.  Cancel the monthly
# request, or let it finish, before submitting this one.
#
# The arguments are positional, because the ECMWF sbatch wrapper does not
# reliably carry the caller's environment into the job:
#
#   sbatch --array=1-4%2 retrieve_chunk.sbatch 202602 pl 0000
#   sbatch --array=1-4%2 retrieve_chunk.sbatch 202602 pl 0000 8
#
# The fourth argument is the chunk width in days and defaults to 8.  The fifth,
# which is almost never wanted, is the data root; it exists so that a dry run
# can be pointed at a scratch root and kept out of the production block
# directories altogether.  Ask the code how many chunks a combination produces
# before choosing the array range:
#
#   python -m aifsens2_regen.retrieve --block 202602 --group pl --time 0000 --list
#
# The time limit is one day rather than the twelve hours the monthly script
# uses, because a request can sit in the archive queue for more than twelve
# hours before a single field arrives, and the shorter limit has already killed
# requests that were merely waiting.

source /home/ecm5702/dev/downscaling-tools-aifsens2-regen/tools/aifsens2_regen/env.sh

BLOCK="${1:-${BLOCK:-}}"
: "${BLOCK:?give the block as the first argument, for example: sbatch retrieve_chunk.sbatch 202602 pl 0000}"
GROUP="${2:-${GROUP:-}}"
: "${GROUP:?give the group as the second argument, one of sfc pl q wave con}"
TIME_OF_DAY="${3:-${TIME_OF_DAY:-}}"
: "${TIME_OF_DAY:?give the input time of day as the third argument, for example 0000}"
CHUNK_DAYS="${4:-${CHUNK_DAYS:-8}}"
if [ -n "${5:-}" ]; then
  export AIFSENS2_ROOT="$5"
fi
echo "data root used: $AIFSENS2_ROOT"

python -m aifsens2_regen.retrieve \
  --block "$BLOCK" --group "$GROUP" --time "$TIME_OF_DAY" \
  --chunk-days "$CHUNK_DAYS" --root "$AIFSENS2_ROOT" \
  --index "${SLURM_ARRAY_TASK_ID:-1}"
rc=$?
echo "RETRIEVE_CHUNK_TASK_RC=$rc block=$BLOCK group=$GROUP time=$TIME_OF_DAY chunk_days=$CHUNK_DAYS index=${SLURM_ARRAY_TASK_ID:-1}"
exit $rc
