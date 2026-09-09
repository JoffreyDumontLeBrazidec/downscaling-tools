#!/bin/bash
# run_stage2a_head.sh: the one command that finishes stage 2a once both inference
# arrays have completed.
#
# What it does, in order, as a chain of SLURM jobs each waiting for the one before
# it to succeed:
#
#   1. refreshes the first-cut manifest, which opens every prediction file and says
#      which cases have a complete prediction;
#   2. gathers, in an array of eight tasks, the station feature table of every case
#      whose prediction is finished and which has not been gathered yet;
#   3. refreshes the manifest again, so that it records the gathers that just ran;
#   4. builds the stable-network flag once, then assembles, in an array of twelve tasks, the training arrays for the three
#      target variables, applying the stable-network flag, the exclusion of the
#      valid time 1 September 2026 00 UTC and quaver's gross-error screen;
#   5. for each of the three targets, trains the full station head and the control
#      head that sees only the interpolated AIFS input, then scores both of them
#      against the nearest-point rule and against the analysis at the nearest
#      point, and writes the score table as CSV and as markdown under
#      /home/ecm5702/perm/station-head-adapter/head/<run_id>/.
#
# Nothing here submits inference and nothing here touches the prediction files
# other than to read them. Running it again is safe: cases already gathered and
# cases already assembled are skipped.
#
# Usage:
#     bash /home/ecm5702/dev/downscaling-tools-station-head/tools/station_head/stage2a/run_stage2a_head.sh
#
# Environment variables that change what it does:
#     RUN_PREFIX   the name the run folders start with (default firstcut_<date>)
#     TAG          the name of the assembled dataset directory (default firstcut)
#     VAL_FROM     the first initialisation date of the validation split
#                  (default 2026-08-18, which is the split of the design note)
#     VAL_TO       the last initialisation date of the validation split (default none)
#     PIPELINE_TEST=1  label every output as a pipeline test and not a result
#     SKIP_GATHER=1    start at the assembly, for a rerun where the gather is done
set -uo pipefail

S=/home/ecm5702/dev/downscaling-tools-station-head/tools/station_head/stage2a
LOGS=/home/ecm5702/agent-work/20260909-station-head-adapter/logs
mkdir -p "$LOGS"

TAG=${TAG:-firstcut}
RUN_PREFIX=${RUN_PREFIX:-firstcut_$(date -u +%Y%m%d)}
VAL_FROM=${VAL_FROM:-2026-08-18}
VAL_TO=${VAL_TO:-}
PIPELINE_TEST=${PIPELINE_TEST:-0}

echo "stage 2a head chain: tag=$TAG run_prefix=$RUN_PREFIX validation from $VAL_FROM ${VAL_TO:+to $VAL_TO} pipeline_test=$PIPELINE_TEST"

submit() {  # submit <dependency-or-empty> <sbatch args...>
  local dep="$1"; shift
  local out
  if [ -n "$dep" ]; then
    out=$(sbatch --parsable --dependency=afterok:"$dep" "$@")
  else
    out=$(sbatch --parsable "$@")
  fi
  echo "${out%%;*}"
}

DEP=""
if [ "${SKIP_GATHER:-0}" != "1" ]; then
  J1=$(submit "" --job-name=sh2a_manifest --qos=nf --time=01:00:00 --mem=32G --cpus-per-task=4 \
        --output="$LOGS/sh2a_manifest_%j.out" --wrap="bash $S/refresh_manifest.sh")
  echo "manifest refresh: $J1"
  J2=$(submit "$J1" --array=0-7 "$S/gather_array.sbatch")
  echo "gather array:     $J2"
  J3=$(submit "$J2" --job-name=sh2a_manifest --qos=nf --time=01:00:00 --mem=32G --cpus-per-task=4 \
        --output="$LOGS/sh2a_manifest_%j.out" --wrap="bash $S/refresh_manifest.sh")
  echo "manifest refresh: $J3"
  DEP="$J3"
fi

JS=$(submit "$DEP" --job-name=sh2a_stable --qos=nf --time=01:00:00 --mem=64G --cpus-per-task=4 \
      --output="$LOGS/sh2a_stable_%j.out" \
      --wrap="module load python3 vtb ecmwf-toolbox && python3 -u $S/dataset.py --target 2t --stable-only")
echo "stable network:   $JS"
DEP="$JS"

if [ -n "$DEP" ]; then
  J4=$(submit "$DEP" --array=0-11 --export=ALL,TAG="$TAG" "$S/assemble_array.sbatch")
else
  J4=$(submit "" --array=0-11 --export=ALL,TAG="$TAG" "$S/assemble_array.sbatch")
fi
echo "assemble array:   $J4"

for T in 2t 2d 10ff; do
  JT=$(submit "$J4" --job-name="sh2a_head_$T" \
        --export=ALL,TARGET="$T",TAG="$TAG",RUN_PREFIX="$RUN_PREFIX",VAL_FROM="$VAL_FROM",VAL_TO="$VAL_TO",PIPELINE_TEST="$PIPELINE_TEST" \
        "$S/train_score.sbatch")
  echo "head + score $T:  $JT"
done

echo
echo "Watch with: squeue -u ecm5702 -n sh2a_manifest,sh2a_gather,sh2a_assemble,sh2a_head_2t,sh2a_head_2d,sh2a_head_10ff"
echo "Results land in /home/ecm5702/perm/station-head-adapter/head/${RUN_PREFIX}_<target>/"
