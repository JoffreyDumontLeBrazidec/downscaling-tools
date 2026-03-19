#!/usr/bin/env bash
set -euo pipefail

# ── Proxy TC Eval ─────────────────────────────────────────────────────────────
# Runs the 10 canonical date/step pairs with a configurable member subset.
# Default proxy member scope is member 1 only, so the fast gate writes
# 10 total date-step-member predictions while preserving the canonical pair set.
# That subset captures ~64% of the combined TC extreme signal from the full
# 250-run evaluation.
#
# Top-10 bundles (ranked by combined Idalia+Franklin MSLP/wind extremes):
#   20230829:24, 20230828:48, 20230829:48, 20230828:24, 20230830:24,
#   20230828:72, 20230827:72, 20230830:48, 20230829:72, 20230827:48
#
# Usage:
#   ./launch_proxy_eval.sh --run-id proxy_test1 --ckpt-id <checkpoint_id>
#   ./launch_proxy_eval.sh --run-id proxy_test1 --ckpt-id <checkpoint_id> --dry-run
# ──────────────────────────────────────────────────────────────────────────────

PROXY_BUNDLE_PAIRS="20230829:24,20230828:48,20230829:48,20230828:24,20230830:24,20230828:72,20230827:72,20230830:48,20230829:72,20230827:48"
PROXY_N_FILES=10

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --run-id <id> --ckpt-id <id> [options]

Proxy TC quick-eval: 10 canonical date/step pairs (${PROXY_N_FILES} files).
Default proxy member scope: member 1 only (= 10 total predictions, target ~30 min GPU, qos=dg).

Options:
  --run-id <id>               Required run id (e.g. proxy_v1)
  --eval-root <path>          Eval root (default: /home/ecm5702/perm/eval)
  --input-root <path>         Bundle input root
  --ckpt-id <id>              Checkpoint id (required)
  --name-ckpt <path>          Explicit checkpoint path (overrides --ckpt-id)
  --predict-qos <qos>         Predict job QoS (default: dg)
  --predict-time <hh:mm:ss>   Predict walltime (default: 00:30:00)
  --predict-cpus <n>          Predict cpus-per-task (default: 32)
  --predict-mem <mem>         Predict memory (default: 256G)
  --predict-gpus <n>          Predict gpus-per-node (default: 1)
  --members <csv>             Proxy member ids (default: 1)
  --eval-qos <qos>            Eval job QoS (default: nf)
  --eval-time <hh:mm:ss>      Eval walltime (default: 04:00:00)
  --eval-cpus <n>             Eval cpus-per-task (default: 8)
  --eval-mem <mem>            Eval memory (default: 64G)
  --allow-existing-run-dir    Allow reusing an existing run directory
  --skip-eval                 Only run predictions, skip evaluation
  --dry-run                   Generate scripts only
EOF
}

RUN_ID=""
EVAL_ROOT="/home/ecm5702/perm/eval"
INPUT_ROOT="/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia"
CKPT_ID=""
NAME_CKPT=""
PREDICT_QOS="dg"
PREDICT_TIME="00:30:00"
PREDICT_CPUS="32"
PREDICT_MEM="256G"
PREDICT_GPUS="1"
PREDICT_MEMBERS="1"
EVAL_QOS="nf"
EVAL_TIME="04:00:00"
EVAL_CPUS="8"
EVAL_MEM="64G"
ALLOW_EXISTING_RUN_DIR=0
SKIP_EVAL=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --input-root) INPUT_ROOT="$2"; shift 2 ;;
    --ckpt-id) CKPT_ID="$2"; shift 2 ;;
    --name-ckpt) NAME_CKPT="$2"; shift 2 ;;
    --predict-qos) PREDICT_QOS="$2"; shift 2 ;;
    --predict-time) PREDICT_TIME="$2"; shift 2 ;;
    --predict-cpus) PREDICT_CPUS="$2"; shift 2 ;;
    --predict-mem) PREDICT_MEM="$2"; shift 2 ;;
    --predict-gpus) PREDICT_GPUS="$2"; shift 2 ;;
    --members) PREDICT_MEMBERS="$2"; shift 2 ;;
    --eval-qos) EVAL_QOS="$2"; shift 2 ;;
    --eval-time) EVAL_TIME="$2"; shift 2 ;;
    --eval-cpus) EVAL_CPUS="$2"; shift 2 ;;
    --eval-mem) EVAL_MEM="$2"; shift 2 ;;
    --allow-existing-run-dir) ALLOW_EXISTING_RUN_DIR=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then echo "--run-id is required" >&2; usage; exit 2; fi
if [[ -z "$CKPT_ID" && -z "$NAME_CKPT" ]]; then echo "--ckpt-id or --name-ckpt is required" >&2; usage; exit 2; fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID contains unsafe characters: ${RUN_ID}" >&2; exit 2
fi
IFS=',' read -r -a PROXY_MEMBER_LIST <<< "${PREDICT_MEMBERS}"
PROXY_MEMBER_COUNT=0
for raw_member in "${PROXY_MEMBER_LIST[@]}"; do
  trimmed="${raw_member//[[:space:]]/}"
  if [[ -n "${trimmed}" ]]; then
    PROXY_MEMBER_COUNT=$((PROXY_MEMBER_COUNT + 1))
  fi
done
if [[ "${PROXY_MEMBER_COUNT}" -eq 0 ]]; then
  echo "--members must include at least one member id" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_DIR="${EVAL_ROOT}/${RUN_ID}"
JOBS_DIR="${RUN_DIR}/jobs"
PRED_DIR="${RUN_DIR}/predictions"
EVAL_RUN_ROOT="${RUN_DIR}/eval"

if [[ -e "${RUN_DIR}" && "${ALLOW_EXISTING_RUN_DIR}" -ne 1 ]]; then
  echo "Run directory already exists: ${RUN_DIR}" >&2
  echo "Use a new run-id, or pass --allow-existing-run-dir explicitly." >&2
  exit 2
fi
mkdir -p "${JOBS_DIR}" "${RUN_DIR}/logs" "${PRED_DIR}" "${EVAL_RUN_ROOT}"

PREDICTION_SAFETY_FLAGS=""
if [[ "${ALLOW_EXISTING_RUN_DIR}" -eq 1 ]]; then
  PREDICTION_SAFETY_FLAGS="--allow-existing-out-dir --allow-overwrite-existing-files"
fi

CKPT_FLAG=""
if [[ -n "$NAME_CKPT" ]]; then
  CKPT_FLAG="--name-ckpt ${NAME_CKPT}"
else
  CKPT_FLAG="--ckpt-id ${CKPT_ID}"
fi

# ── predict proxy sbatch ──────────────────────────────────────────────────────
cat > "${JOBS_DIR}/predict_proxy_${RUN_ID}.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=pxy_${RUN_ID}
#SBATCH --qos=${PREDICT_QOS}
#SBATCH --time=${PREDICT_TIME}
#SBATCH --cpus-per-task=${PREDICT_CPUS}
#SBATCH --gpus-per-node=${PREDICT_GPUS}
#SBATCH --mem=${PREDICT_MEM}
#SBATCH --output=${RUN_DIR}/logs/predict_proxy_${RUN_ID}_%j.out
set -euo pipefail
source /home/ecm5702/dev/.ds-dyn/bin/activate
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
export DATA_DIR=/home/mlx/ai-ml/datasets/
export DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export BUNDLE_MODULE_DIR=/home/ecm5702/dev/experiments/2026_02_16_clean_grib_prediction
cd ${PROJECT_ROOT}
python eval/jobs/generate_predictions_25_files.py \\
  --input-root ${INPUT_ROOT} \\
  --out-dir ${PRED_DIR} \\
  ${CKPT_FLAG} \\
  --bundle-pairs "${PROXY_BUNDLE_PAIRS}" \\
  --members "${PREDICT_MEMBERS}" \\
  --device cuda ${PREDICTION_SAFETY_FLAGS}
EOF

# ── eval proxy sbatch ─────────────────────────────────────────────────────────
cat > "${JOBS_DIR}/eval_proxy_${RUN_ID}.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=exy_${RUN_ID}
#SBATCH --qos=${EVAL_QOS}
#SBATCH --time=${EVAL_TIME}
#SBATCH --cpus-per-task=${EVAL_CPUS}
#SBATCH --mem=${EVAL_MEM}
#SBATCH --output=${RUN_DIR}/logs/eval_proxy_${RUN_ID}_%j.out
set -euo pipefail
source /home/ecm5702/dev/.ds-dyn/bin/activate
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
cd ${PROJECT_ROOT}
count=0
for f in ${PRED_DIR}/predictions_*.nc; do
  [ -f "\$f" ] || continue
  base=\$(basename "\$f" .nc)
  python -m eval.run --eval-root ${EVAL_RUN_ROOT} predictions --predictions-nc "\$f" --run-name "\${base}" --skip-region
  count=\$((count+1))
  echo "evaluated \$count file(s): \$f"
done
if [ "\$count" -ne ${PROXY_N_FILES} ]; then
  echo "Expected ${PROXY_N_FILES} prediction files but evaluated \$count" >&2
  exit 3
fi

# ── TC extreme comparison against anchor ──────────────────────────────────────
ANCHOR_JSON="${EVAL_ROOT}/anchor_tc_extremes.json"
if [ -f "\${ANCHOR_JSON}" ]; then
  echo "Running proxy TC comparison against anchor..."
  set +e
  python -m eval.jobs.proxy_tc_compare \\
    --proxy-predictions-dir ${PRED_DIR} \\
    --anchor-json "\${ANCHOR_JSON}" \\
    --out-json ${RUN_DIR}/proxy_tc_compare.json \\
    --support-mode native
  compare_rc=\$?
  set -e
  if [ ! -f "${RUN_DIR}/proxy_tc_compare.json" ]; then
    echo "proxy_tc_compare did not write ${RUN_DIR}/proxy_tc_compare.json" >&2
    exit "\${compare_rc}"
  fi
  echo "TC comparison saved to ${RUN_DIR}/proxy_tc_compare.json"
  if [ "\${compare_rc}" -ne 0 ]; then
    echo "Proxy TC comparison returned \${compare_rc}; preserving scheduler success because verdict JSON was written."
  fi
else
  echo "No anchor TC extremes JSON at \${ANCHOR_JSON} — skipping TC comparison."
  echo "Generate one with: python -m eval.jobs.diagnose_per_bundle_tc_extremes --predictions-dir <anchor_predictions> --out-json \${ANCHOR_JSON}"
fi
EOF

chmod +x "${JOBS_DIR}"/*.sbatch

echo "Proxy eval config:"
echo "  bundle-pairs: ${PROXY_BUNDLE_PAIRS}"
echo "  members: ${PREDICT_MEMBERS}"
echo "  n_files: ${PROXY_N_FILES}"
echo "  n_member_predictions: $((PROXY_N_FILES * PROXY_MEMBER_COUNT))"
echo "  predict qos: ${PREDICT_QOS}, time: ${PREDICT_TIME}"
echo "  scripts: ${JOBS_DIR}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run. Scripts written to ${JOBS_DIR}"
  exit 0
fi

jid_predict=$(sbatch "${JOBS_DIR}/predict_proxy_${RUN_ID}.sbatch" | awk '{print $NF}')

if [[ "${SKIP_EVAL}" -eq 1 ]]; then
  echo "Submitted predict_proxy: ${jid_predict} (eval skipped)"
  echo "Monitor: squeue -j ${jid_predict}"
else
  jid_eval=$(sbatch --dependency=afterok:${jid_predict} "${JOBS_DIR}/eval_proxy_${RUN_ID}.sbatch" | awk '{print $NF}')
  echo "Submitted:"
  echo "  predict_proxy ${jid_predict}"
  echo "  eval_proxy    ${jid_eval} (afterok:${jid_predict})"
  echo "Monitor: squeue -j ${jid_predict},${jid_eval}"
fi
