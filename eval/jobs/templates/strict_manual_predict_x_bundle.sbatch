#!/bin/bash --login
#SBATCH --job-name=manual_predict_x_bundle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --output=/home/ecm5702/perm/eval/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
STACK_FLAVOR="new"         # new | old
LANE="o96_o320"            # o96_o320 | o320_o1280 | o1280_o2560
SOURCE_HPC="ac"            # ac | ag | leonardo | jupiter

CHECKPOINT_REF="REPLACE_CHECKPOINT_REF"
CHECKPOINT_SHORT="REPLACE_CHECKPOINT_SHORT"
CKPT_ROOT="/home/ecm5702/scratch/aifs/checkpoint"
INPUT_ROOT="/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia"

# Date policy:
# - DATE_PRESET=default: 20230826..20230830 (default behavior)
# - DATE_PRESET=aug16_30: 20230816..20230830 (explicitly asked behavior)
# - DATES=<csv> overrides DATE_PRESET completely.
DATE_PRESET="default"      # default | aug16_30
DATES=""                   # explicit CSV override (example: 20230816,20230817)
STEPS="24,48,72,96,120"
MEMBERS="1,2,3,4,5,6,7,8,9,10"

OUTPUT_ROOT="/home/ecm5702/perm/eval"
RUN_DATE_UTC="$(date -u +%Y%m%d)"
RUN_SUFFIX=""              # optional, safe chars only

DEVICE="cuda"
PRECISION="fp32"
VALIDATION_FREQUENCY="50h"  # used by new stack only
NUM_GPUS_PER_MODEL=1        # used by new stack only
# Keep the whole JSON blob shell-quoted as one string; outer single quotes are safest.
EXTRA_ARGS_JSON='{"schedule_type":"experimental_piecewise","num_steps":30,"sigma_max":100000.0,"sigma_transition":100.0,"sigma_min":0.03,"high_schedule_type":"exponential","low_schedule_type":"karras","num_steps_high":10,"num_steps_low":20,"rho":7.0,"sampler":"heun","S_churn":2.5,"S_min":0.75,"S_max":100000.0,"S_noise":1.05}'

# Old stack currently allows missing target in code path; keep default strict.
ALLOW_MISSING_TARGET_OLD=0
# Keep this at 0 unless you intentionally overwrite a previous run.
ALLOW_EXISTING_RUN_DIR=0
###############################################################################

die() {
  echo "ERROR: $*" >&2
  exit 2
}

normalize_extra_args_json() {
  local raw="$1"
  python - "${raw}" <<'PYJSON'
import json
import sys

raw = sys.argv[1]
try:
    parsed = json.loads(raw) if raw else {}
except json.JSONDecodeError as exc:
    raise SystemExit(
        "EXTRA_ARGS_JSON is not valid JSON. Keep the whole value shell-quoted, for example \"EXTRA_ARGS_JSON='\"{...}\"'\". "
        f"Original error: {exc}"
    )
if not isinstance(parsed, dict):
    raise SystemExit(f"EXTRA_ARGS_JSON must decode to a JSON object, got {type(parsed).__name__}")
print(json.dumps(parsed, separators=(",", ":")))
PYJSON
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

require_choice "${STACK_FLAVOR}" new old
require_choice "${LANE}" o96_o320 o320_o1280 o1280_o2560
require_choice "${SOURCE_HPC}" ac ag leonardo jupiter
require_choice "${DATE_PRESET}" default aug16_30
[[ "${CHECKPOINT_REF}" != REPLACE_* ]] || die "Set CHECKPOINT_REF."
[[ "${CHECKPOINT_SHORT}" != REPLACE_* ]] || die "Set CHECKPOINT_SHORT."
[[ -d "${INPUT_ROOT}" ]] || die "INPUT_ROOT does not exist: ${INPUT_ROOT}"
[[ "${INPUT_ROOT}" == *"/${LANE}/"* ]] || die "INPUT_ROOT does not match lane ${LANE}: ${INPUT_ROOT}"

if [[ -z "${DATES}" ]]; then
  if [[ "${DATE_PRESET}" == "default" ]]; then
    DATES="20230826,20230827,20230828,20230829,20230830"
  else
    DATES="20230816,20230817,20230818,20230819,20230820,20230821,20230822,20230823,20230824,20230825,20230826,20230827,20230828,20230829,20230830"
  fi
fi

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) HOST_FAMILY="ac" ;;
  ag*) HOST_FAMILY="ag" ;;
  *) die "Unsupported login node family (${HOST_SHORT}). Run from ac-* or ag-*." ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ "${STACK_FLAVOR}" == "new" ]]; then
  CODE_ROOT="${REPO_ROOT}"
  if [[ "${HOST_FAMILY}" == "ac" ]]; then
    VENV_ACTIVATE="/home/ecm5702/dev/.ds-dyn/bin/activate"
  else
    VENV_ACTIVATE="/home/ecm5702/dev/.ds-ag/bin/activate"
  fi
else
  CODE_ROOT="/home/ecm5702/dev/anemoi_2024/downscaling-tools-OLD"
  if [[ "${HOST_FAMILY}" == "ac" ]]; then
    VENV_ACTIVATE="/home/ecm5702/dev/.ds-old/bin/activate"
  else
    VENV_ACTIVATE="/home/ecm5702/dev/.ds-ag-old/bin/activate"
  fi
fi

[[ -d "${CODE_ROOT}" ]] || die "Code root not found: ${CODE_ROOT}"
[[ -f "${VENV_ACTIVATE}" ]] || die "Env activate script not found: ${VENV_ACTIVATE}"
CHECKPOINT_PROFILE_ROOT="${REPO_ROOT}"
[[ -d "${CHECKPOINT_PROFILE_ROOT}" ]] || die "Checkpoint profile root not found: ${CHECKPOINT_PROFILE_ROOT}"

RUN_ID="manual_${CHECKPOINT_SHORT}_${STACK_FLAVOR}_${LANE}_${RUN_DATE_UTC}"
if [[ -n "${RUN_SUFFIX}" ]]; then
  RUN_ID="${RUN_ID}_${RUN_SUFFIX}"
fi
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "Unsafe RUN_ID: ${RUN_ID}"

RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
PRED_DIR="${RUN_DIR}/predictions"
MANIFEST_PATH="${RUN_DIR}/predictions_manifest.csv"

if [[ -e "${RUN_DIR}" && "${ALLOW_EXISTING_RUN_DIR}" -ne 1 ]]; then
  die "Run directory already exists: ${RUN_DIR}"
fi
mkdir -p "${RUN_DIR}/logs"

source "${VENV_ACTIVATE}"
BASE_PYTHONPATH="${PYTHONPATH:-}"
NORMALIZED_EXTRA_ARGS_JSON="$(normalize_extra_args_json "${EXTRA_ARGS_JSON}")"
EXTRA_ARGS_JSON="${NORMALIZED_EXTRA_ARGS_JSON}"
if [[ -n "${BASE_PYTHONPATH}" ]]; then
  export PYTHONPATH="${CODE_ROOT}:${BASE_PYTHONPATH}"
else
  export PYTHONPATH="${CODE_ROOT}"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR=/home/mlx/ai-ml/datasets/
export DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export BUNDLE_MODULE_DIR=/home/ecm5702/dev/experiments/2026_02_16_clean_grib_prediction

if [[ -n "${BASE_PYTHONPATH}" ]]; then
  CHECKPOINT_PROFILE_PYTHONPATH="${CHECKPOINT_PROFILE_ROOT}:${BASE_PYTHONPATH}"
else
  CHECKPOINT_PROFILE_PYTHONPATH="${CHECKPOINT_PROFILE_ROOT}"
fi
PYTHONPATH="${CHECKPOINT_PROFILE_PYTHONPATH}" \
python -m eval.jobs.checkpoint_profile \
  --name-ckpt "${CHECKPOINT_REF}" \
  --ckpt-root "${CKPT_ROOT}" \
  --source-hpc "${SOURCE_HPC}" \
  --host-short "${HOST_SHORT}" \
  --expected-lane "${LANE}" \
  --expected-stack-flavor "${STACK_FLAVOR}" \
  --expected-venv "${VENV_ACTIVATE}"

{
  echo "run_id=${RUN_ID}"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host_short=${HOST_SHORT}"
  echo "host_family=${HOST_FAMILY}"
  echo "source_hpc=${SOURCE_HPC}"
  echo "stack_flavor=${STACK_FLAVOR}"
  echo "lane=${LANE}"
  echo "venv_activate=${VENV_ACTIVATE}"
  echo "python=$(which python)"
  echo "python_version=$(python -V 2>&1)"
  echo "code_root=${CODE_ROOT}"
  echo "checkpoint_ref=${CHECKPOINT_REF}"
  echo "ckpt_root=${CKPT_ROOT}"
  echo "input_root=${INPUT_ROOT}"
  echo "date_preset=${DATE_PRESET}"
  echo "dates=${DATES}"
  echo "steps=${STEPS}"
  echo "members=${MEMBERS}"
  echo "output_predictions_dir=${PRED_DIR}"
  echo "manifest=${MANIFEST_PATH}"
  echo "device=${DEVICE}"
  echo "precision=${PRECISION}"
  echo "validation_frequency=${VALIDATION_FREQUENCY}"
  echo "num_gpus_per_model=${NUM_GPUS_PER_MODEL}"
  echo "extra_args_json=${EXTRA_ARGS_JSON}"
} > "${RUN_DIR}/manual_inference_run_info.txt"

python - "${RUN_DIR}/EXPERIMENT_CONFIG.yaml" "${RUN_ID}" "${RUN_DIR}" "${STACK_FLAVOR}" "${LANE}" "${SOURCE_HPC}" "${HOST_SHORT}" "${VENV_ACTIVATE}" "${CODE_ROOT}" "${CHECKPOINT_REF}" "${CKPT_ROOT}" "${INPUT_ROOT}" "${DATES}" "${MEMBERS}" "${STEPS}" "${EXTRA_ARGS_JSON}" <<'PY'
import json
import sys
from pathlib import Path


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


out_path = Path(sys.argv[1])
run_id = sys.argv[2]
run_dir = sys.argv[3]
stack_flavor = sys.argv[4]
lane = sys.argv[5]
source_hpc = sys.argv[6]
host_short = sys.argv[7]
venv_activate = sys.argv[8]
code_root = sys.argv[9]
checkpoint_ref = sys.argv[10]
ckpt_root = sys.argv[11]
input_root = sys.argv[12]
dates = parse_int_list(sys.argv[13])
members = parse_int_list(sys.argv[14])
steps = parse_int_list(sys.argv[15])
sampling_config_json = sys.argv[16]

checkpoint_token = checkpoint_ref.split("/", 1)[0]
checkpoint_path = Path(checkpoint_ref)
if not checkpoint_path.is_absolute():
    checkpoint_path = Path(ckpt_root) / checkpoint_ref

config = {
    "run_id": run_id,
    "artifacts_root": run_dir,
    "stack_flavor": stack_flavor,
    "lane": lane,
    "source_hpc": source_hpc,
    "host_short": host_short,
    "venv_activate": venv_activate,
    "code_root": code_root,
    "sampling_config_json": sampling_config_json,
    "checkpoint": {
        "run_id": checkpoint_token,
        "path": str(checkpoint_path),
    },
    "source": {
        "input_root": input_root,
        "bundle_scope": {
            "dates": dates,
            "members": members,
            "steps_hours": steps,
        },
    },
}
try:
    config["model"] = {"development_hacks": {"extra_args": json.loads(sampling_config_json)}}
except json.JSONDecodeError:
    pass

out_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

cd "${CODE_ROOT}"

if [[ "${STACK_FLAVOR}" == "new" ]]; then
  if [[ "${NUM_GPUS_PER_MODEL}" -gt 1 ]]; then
    srun --ntasks="${NUM_GPUS_PER_MODEL}" --ntasks-per-node="${NUM_GPUS_PER_MODEL}" \
      python eval/jobs/generate_predictions_25_files.py \
      --input-root "${INPUT_ROOT}" \
      --out-dir "${PRED_DIR}" \
      --name-ckpt "${CHECKPOINT_REF}" \
      --ckpt-root "${CKPT_ROOT}" \
      --device "${DEVICE}" \
      --num-gpus-per-model "${NUM_GPUS_PER_MODEL}" \
      --precision "${PRECISION}" \
      --validation-frequency "${VALIDATION_FREQUENCY}" \
      --members "${MEMBERS}" \
      --steps "${STEPS}" \
      --dates "${DATES}" \
      --extra-args-json "${EXTRA_ARGS_JSON}" \
      --manifest "${MANIFEST_PATH}"
  else
    python eval/jobs/generate_predictions_25_files.py \
      --input-root "${INPUT_ROOT}" \
      --out-dir "${PRED_DIR}" \
      --name-ckpt "${CHECKPOINT_REF}" \
      --ckpt-root "${CKPT_ROOT}" \
      --device "${DEVICE}" \
      --num-gpus-per-model "${NUM_GPUS_PER_MODEL}" \
      --precision "${PRECISION}" \
      --validation-frequency "${VALIDATION_FREQUENCY}" \
      --members "${MEMBERS}" \
      --steps "${STEPS}" \
      --dates "${DATES}" \
      --extra-args-json "${EXTRA_ARGS_JSON}" \
      --manifest "${MANIFEST_PATH}"
  fi
else
  OLD_ALLOW_FLAG=()
  if [[ "${ALLOW_MISSING_TARGET_OLD}" -eq 1 ]]; then
    OLD_ALLOW_FLAG=(--allow-missing-target)
  fi
  python eval/jobs/generate_predictions_25_files.py \
    --input-root "${INPUT_ROOT}" \
    --out-dir "${PRED_DIR}" \
    --ckpt-id "${CHECKPOINT_REF}" \
    --ckpt-root "${CKPT_ROOT}" \
    --device "${DEVICE}" \
    --precision "${PRECISION}" \
    --members "${MEMBERS}" \
    --steps "${STEPS}" \
    --dates "${DATES}" \
    --extra-args-json "${EXTRA_ARGS_JSON}" \
    --manifest "${MANIFEST_PATH}" \
    "${OLD_ALLOW_FLAG[@]}"
fi

python - "${PRED_DIR}" "${STACK_FLAVOR}" "${ALLOW_MISSING_TARGET_OLD}" <<'PY'
import pathlib
import sys
import xarray as xr

pred_dir = pathlib.Path(sys.argv[1])
stack = sys.argv[2]
allow_missing_target_old = int(sys.argv[3])
paths = sorted(pred_dir.glob("predictions_*_step*.nc"))
if not paths:
    raise SystemExit(f"No predictions_*.nc files found in {pred_dir}")

missing_y_pred = []
missing_y = []
for p in paths:
    with xr.open_dataset(p) as ds:
        if "y_pred" not in ds.variables:
            missing_y_pred.append(str(p))
        if stack == "new":
            if "y" not in ds.variables:
                missing_y.append(str(p))
        elif allow_missing_target_old == 0:
            if "y" not in ds.variables:
                missing_y.append(str(p))

if missing_y_pred:
    raise SystemExit(f"Missing y_pred in files: {missing_y_pred[:5]}")
if missing_y:
    raise SystemExit(f"Missing y in files: {missing_y[:5]}")

print(f"Validated {len(paths)} predictions files in {pred_dir}")
PY

echo "DONE: ${PRED_DIR}"
