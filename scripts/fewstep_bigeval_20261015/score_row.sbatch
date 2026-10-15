#!/bin/bash --login
# Score one row of the few-step big eval (o96->o320) on one case set, on Atos CPU (QOS nf).
#
#   sbatch --export=ALL,ROW=edm_pw30c0,SET=A,PRED_ROOT=...,EVAL_ROOT=...,DT_ROOT=... \
#          scripts/fewstep_bigeval_20261015/score_row.sbatch
#
# Required: ROW (label of rows.yaml), SET (A or B), PRED_ROOT, EVAL_ROOT, DT_ROOT (a worktree of
# downscaling-tools at branch claude/project-thread-72mhur; ~/dev/downscaling-tools is an older
# main and is never used or modified here).
# Optional: PRED_SET_DIR (default set$SET, '' when PRED_ROOT is the set's root), ROWS_FILE,
#           ONLY / SKIP (comma lists of evaluators), ECMWF_SPECTRA=1 (adds spectra_ecmwf_v2, AC
#           only), PY (interpreter, default the certified env's python).
# An evaluator whose results directory holds .complete is skipped, so a job can be resubmitted.
#SBATCH --job-name=bigeval-score
#SBATCH --qos=nf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out
set -euo pipefail

: "${ROW:?ROW is required}" "${SET:?SET is required (A or B)}"
: "${PRED_ROOT:?PRED_ROOT is required}" "${EVAL_ROOT:?EVAL_ROOT is required}"
: "${DT_ROOT:?DT_ROOT (worktree of claude/project-thread-72mhur) is required}"
[ -d "$DT_ROOT/eval" ] || { echo "DT_ROOT=$DT_ROOT has no eval/ package" >&2; exit 2; }
if [ "$(cd "$DT_ROOT" && pwd -P)" = "$(cd "$HOME/dev/downscaling-tools" 2>/dev/null && pwd -P || true)" ]; then
  echo "DT_ROOT must be a separate worktree, not ~/dev/downscaling-tools" >&2; exit 2
fi

# Environment, as eval/config/hosts/atos_ac.yaml declares it.
set +u                       # module and activate scripts read unset variables
module load ecmwf-toolbox
# shellcheck disable=SC1091
source /home/ecm5702/dev/.ds-260612/bin/activate
set -u
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONPATH="$DT_ROOT"   # this worktree only (the certified env runs with no other PYTHONPATH)
PY="${PY:-python}"

cd "$DT_ROOT"
echo "host $(hostname)  commit $(git -C "$DT_ROOT" rev-parse --short HEAD)  row $ROW  set $SET"
mkdir -p "$EVAL_ROOT/$SET/$ROW"

args=(--row "$ROW" --set "$SET" --pred-root "$PRED_ROOT" --eval-root "$EVAL_ROOT"
      --rows "${ROWS_FILE:-$DT_ROOT/scripts/fewstep_bigeval_20261015/rows.yaml}" --python "$PY")
[ -n "${PRED_SET_DIR+x}" ] && args+=(--pred-set-dir "$PRED_SET_DIR")
[ -n "${ONLY:-}" ] && args+=(--only "$ONLY")
[ -n "${SKIP:-}" ] && args+=(--skip "$SKIP")
[ "${ECMWF_SPECTRA:-0}" = 1 ] && args+=(--with-ecmwf-spectra)

"$PY" "$DT_ROOT/scripts/fewstep_bigeval_20261015/score_row.py" "${args[@]}" \
  2>&1 | tee -a "$EVAL_ROOT/$SET/$ROW/score_row.log"
exit "${PIPESTATUS[0]}"
