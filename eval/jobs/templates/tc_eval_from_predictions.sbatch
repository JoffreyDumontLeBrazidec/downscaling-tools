#!/bin/bash --login
# Canonical TC evaluation template — works on both AC and AG.
#
# Generates normalized TC PDF plots from predictions_*.nc files using:
#   <repo>/eval/tc/plot_pdf_tc_from_predictions.py
#
# PREREQUISITE: TC reference GRIB data must exist under BASE_TC_DIR.
# To request/refresh TC reference data for all events, run:
#   sbatch <repo>/eval/tc/all_events_request.sh
# (AC-only, needs MARS access. Edit EXPID in that script for your experiment.)
#
# Edit only the USER SETTINGS block, then submit with:
#   sbatch tc_eval_from_predictions.sbatch
#
# Preferred smooth submit path:
#   bash <repo>/eval/jobs/templates/submit_tc_eval_from_predictions.sh /path/to/edited_copy.sbatch
# The helper renders a host-safe copy and submits it with the right scheduler posture:
#   - AC auto profile: qos=nf with no GPU request
#   - AG auto profile: qos=ng with --gpus-per-node=0
# It keeps the edited template memory/time by default; for the tested O320 AC-slim profile add:
#   TC_SUBMIT_MEM_OVERRIDE=128G TC_SUBMIT_TIME_OVERRIDE=04:00:00 bash <repo>/eval/jobs/templates/submit_tc_eval_from_predictions.sh /path/to/edited_copy.sbatch
#
# Resource baselines (empirical):
#   O320 five-date:  --mem=128G --time=04:00:00
#   O1280 prediction-only TC PDF: --mem=128G --time=06:00:00 on AC `nf`
#   raise above 128G only when a measured queue/memory decision is recorded explicitly
#
# Submission compatibility note:
#   direct sbatch is still allowed, but raw AC `qos=ng` with `--gpus-per-node=0` can fail with `QOSMinGRES`.
#   the conservative raw directives below stay cross-cluster-compatible for users who intentionally submit the file as-is.

#SBATCH --job-name=tc_eval
#SBATCH --qos=ng
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/ecm5702/perm/eval/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
RUN_ROOT="/home/ecm5702/perm/eval/REPLACE_RUN_ID"
RUN_ID="REPLACE_RUN_ID"
PREDICTIONS_DIR="${RUN_ROOT}/predictions"
OUT_DIR="${RUN_ROOT}"
EVENTS="idalia,franklin"        # comma-separated subset; blank => all events
SUPPORT_MODE="native"           # native | regridded
EXTRA_REFERENCE_EXPIDS=""       # example: ENFO_O320_j138,ENFO_O1280_0001
OUT_STEM=""                     # blank => auto-generated from events + RUN_ID
BASE_TC_DIR="/home/ecm5702/hpcperm/data/tc"
WRITE_EXPERIMENT_CONFIG=0       # 1 => write ${RUN_ROOT}/EXPERIMENT_CONFIG.yaml
###############################################################################

PROJECT_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_choice() {
  local value="$1"; shift
  local ok=1
  for c in "$@"; do
    if [[ "${value}" == "${c}" ]]; then
      ok=0
      break
    fi
  done
  if [[ "${ok}" -ne 0 ]]; then
    die "Invalid value '${value}'. Allowed: $*"
  fi
}

# --- Cluster detection and env setup ---
HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) CLUSTER="ac" ;;
  ag*) CLUSTER="ag" ;;
  *)   die "Unknown cluster from hostname '${HOST_SHORT}'." ;;
esac

require_choice "${SUPPORT_MODE}" native regridded
[[ "${RUN_ROOT}" != *REPLACE_* ]] || die "Set RUN_ROOT."
[[ "${RUN_ID}" != REPLACE_* ]] || die "Set RUN_ID."
[[ -d "${RUN_ROOT}" ]] || die "RUN_ROOT does not exist: ${RUN_ROOT}"
[[ -d "${PREDICTIONS_DIR}" ]] || die "Predictions directory does not exist: ${PREDICTIONS_DIR}"
[[ -d "${BASE_TC_DIR}" ]] || die "BASE_TC_DIR does not exist: ${BASE_TC_DIR}"

# Check prediction file count
PRED_COUNT=$(find -L "${PREDICTIONS_DIR}" -maxdepth 1 -name 'predictions_*.nc' -type f | wc -l)
[[ "${PRED_COUNT}" -gt 0 ]] || die "No predictions_*.nc in ${PREDICTIONS_DIR}"
echo "[INFO] Found ${PRED_COUNT} prediction files"

# Check TC reference data exists for requested events
if [[ -n "${EVENTS}" ]]; then
  IFS=',' read -ra EVENT_LIST <<< "${EVENTS}"
  for ev in "${EVENT_LIST[@]}"; do
    if [[ ! -d "${BASE_TC_DIR}/${ev}" ]]; then
      die "TC reference data missing for event '${ev}'. Request it with: sbatch ${PROJECT_ROOT}/eval/tc/all_events_request.sh"
    fi
  done
fi

mkdir -p "${OUT_DIR}" "${RUN_ROOT}/logs"

if [[ -z "${OUT_STEM}" ]]; then
  EVENT_TAG="${EVENTS//,/__}"
  if [[ -z "${EVENT_TAG}" ]]; then
    EVENT_TAG="all_events"
  fi
  OUT_STEM="tc_normed_pdfs_${EVENT_TAG}_${RUN_ID}_${SUPPORT_MODE}_from_predictions"
fi
PDF_NAME="${OUT_STEM}.pdf"
PDF_PATH="${OUT_DIR}/${PDF_NAME}"
STATS_PATH="${OUT_DIR}/${OUT_STEM}.stats.json"

# --- Environment activation (cluster-aware) ---
case "${CLUSTER}" in
  ac)
    source /home/ecm5702/dev/.ds-dyn/bin/activate
    module unload ifs 2>/dev/null || true
    module load ecmwf-toolbox 2>/dev/null || true
    PYTHON=/home/ecm5702/dev/.ds-dyn/bin/python
    ;;
  ag)
    source /home/ecm5702/dev/.ds-ag/bin/activate
    module unload ifs 2>/dev/null || true
    module load ecmwf-toolbox 2>/dev/null || true
    PYTHON=/home/ecm5702/dev/.ds-ag/bin/python
    ;;
esac

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# --- Env validation ---
echo "[INFO] cluster=${CLUSTER} host=${HOST_SHORT}"
echo "[INFO] VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"
echo "[INFO] python=$(which python) ($("${PYTHON}" -V 2>&1))"

"${PYTHON}" -c "import earthkit.data, xarray, matplotlib; print('[INFO] core packages OK')"
"${PYTHON}" -c "import cmcrameri, seaborn; print('[INFO] plotting packages OK')" || \
  die "Missing plotting packages (cmcrameri, seaborn). Install into the active venv."

if [[ "${SUPPORT_MODE}" == "regridded" ]]; then
  "${PYTHON}" -c "import metview; print('[INFO] metview OK')" || \
    die "Regridded mode requires metview. On AG, use native mode or switch to AC."
fi

# --- Run TC evaluation ---
cd "${PROJECT_ROOT}"

CMD=(
  "${PYTHON}" -m eval.tc.plot_pdf_tc_from_predictions
  --predictions-dir "${PREDICTIONS_DIR}"
  --outdir "${OUT_DIR}"
  --run-label "${RUN_ID}"
  --out-name "${PDF_NAME}"
  --base-tc-dir "${BASE_TC_DIR}"
  --support-mode "${SUPPORT_MODE}"
)

if [[ -n "${EVENTS}" ]]; then
  CMD+=(--events "${EVENTS}")
fi
if [[ -n "${EXTRA_REFERENCE_EXPIDS}" ]]; then
  CMD+=(--extra-reference-expids "${EXTRA_REFERENCE_EXPIDS}")
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

# --- Verify outputs ---
[[ -f "${PDF_PATH}" ]] || die "Expected TC PDF missing: ${PDF_PATH}"
[[ -f "${STATS_PATH}" ]] || die "Expected TC stats JSON missing: ${STATS_PATH}"

if [[ "${WRITE_EXPERIMENT_CONFIG}" -eq 1 ]]; then
  cat > "${RUN_ROOT}/EXPERIMENT_CONFIG.yaml" <<EOF
run_id: ${RUN_ID}
purpose: "Prediction-run TC evaluation"
predictions:
  source_dir: ${PREDICTIONS_DIR}
  file_count: ${PRED_COUNT}
tc_evaluation:
  method: prediction-driven
  script: ${PROJECT_ROOT}/eval/tc/plot_pdf_tc_from_predictions.py
  data_request_script: ${PROJECT_ROOT}/eval/tc/all_events_request.sh
  pdf: ${PDF_PATH}
  stats_json: ${STATS_PATH}
  support_mode: ${SUPPORT_MODE}
  events: ${EVENTS:-all_events}
  base_tc_dir: ${BASE_TC_DIR}
  extra_reference_expids: ${EXTRA_REFERENCE_EXPIDS:-none}
EOF
fi

echo "DONE: ${PDF_PATH}"
echo "DONE: ${STATS_PATH}"
