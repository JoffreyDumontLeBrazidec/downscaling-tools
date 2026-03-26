#!/bin/bash --login
# Canonical one-date local-plot template — works on both AC and AG.
#
# Renders canonical non-TC local plots from an existing predictions tree using:
#   /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/region_plotting/plot_one_date_local.py
#
# Preferred launch path for the `o320 -> o1280` manual-eval lane:
#   /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/submit_o320_o1280_manual_eval_flow.sh
#
# Raw submission compatibility note:
#   the directives below stay cross-cluster-compatible for deliberate direct
#   `sbatch`, but the preferred helper path patches them to a host-safe CPU
#   posture (`nf` without GPUs on AC, `ng` with `--gpus-per-node=0` on AG).

#SBATCH --job-name=local_plots_one_date
#SBATCH --qos=ng
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/ecm5702/perm/eval/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
RUN_ROOT="/home/ecm5702/perm/eval/REPLACE_RUN_ID"
RUN_ID="REPLACE_RUN_ID"
DATE="20230826"
OUT_SUBDIR="local_plots_one_date"
REGION="auto"
SAMPLE_INDEX=0
ENSEMBLE_MEMBER_INDEX=0
EXPECTED_COUNT=5
MODEL_VARIABLES=""          # blank => plot_one_date_local defaults
WEATHER_STATES=""           # blank => plot_one_date_local defaults
###############################################################################

PROJECT_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) CLUSTER="ac" ;;
  ag*) CLUSTER="ag" ;;
  *) die "Unknown cluster from hostname '${HOST_SHORT}'." ;;
esac

[[ "${RUN_ROOT}" != *REPLACE_* ]] || die "Set RUN_ROOT."
[[ "${RUN_ID}" != REPLACE_* ]] || die "Set RUN_ID."
[[ -d "${RUN_ROOT}" ]] || die "RUN_ROOT does not exist: ${RUN_ROOT}"

PREDICTIONS_DIR="${RUN_ROOT}/predictions"
[[ -d "${PREDICTIONS_DIR}" ]] || die "Predictions directory does not exist: ${PREDICTIONS_DIR}"

mkdir -p "${RUN_ROOT}/logs"

case "${CLUSTER}" in
  ac)
    source /home/ecm5702/dev/.ds-dyn/bin/activate
    PYTHON=/home/ecm5702/dev/.ds-dyn/bin/python
    ;;
  ag)
    source /home/ecm5702/dev/.ds-ag/bin/activate
    PYTHON=/home/ecm5702/dev/.ds-ag/bin/python
    ;;
esac

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[INFO] cluster=${CLUSTER} host=${HOST_SHORT}"
echo "[INFO] run_root=${RUN_ROOT}"
echo "[INFO] date=${DATE}"
echo "[INFO] expected_count=${EXPECTED_COUNT}"
echo "[INFO] VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"
echo "[INFO] python=$(which python) ($("${PYTHON}" -V 2>&1))"

CMD=(
  "${PYTHON}" -m eval.region_plotting.plot_one_date_local
  --run-root "${RUN_ROOT}"
  --date "${DATE}"
  --out-subdir "${OUT_SUBDIR}"
  --region "${REGION}"
  --sample-index "${SAMPLE_INDEX}"
  --ensemble-member-index "${ENSEMBLE_MEMBER_INDEX}"
  --expected-count "${EXPECTED_COUNT}"
  --run-label "${RUN_ID}"
)

if [[ -n "${MODEL_VARIABLES}" ]]; then
  CMD+=(--model-variables "${MODEL_VARIABLES}")
fi
if [[ -n "${WEATHER_STATES}" ]]; then
  CMD+=(--weather-states "${WEATHER_STATES}")
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

OUT_ROOT="${RUN_ROOT}/${OUT_SUBDIR}"
[[ -d "${OUT_ROOT}" ]] || die "Expected output directory missing: ${OUT_ROOT}"

PDF_COUNT="$(find "${OUT_ROOT}" -type f -name '*.pdf' | wc -l | tr -d ' ')"
PNG_COUNT="$(find "${OUT_ROOT}" -type f -name '*.png' | wc -l | tr -d ' ')"
[[ "${PDF_COUNT}" -gt 0 ]] || die "No PDF outputs found under ${OUT_ROOT}"
[[ "${PNG_COUNT}" -gt 0 ]] || die "No PNG outputs found under ${OUT_ROOT}"

echo "DONE: ${OUT_ROOT} (pdf=${PDF_COUNT} png=${PNG_COUNT})"
