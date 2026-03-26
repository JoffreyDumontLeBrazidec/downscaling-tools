#!/bin/bash
# Login-node helper: render and submit the canonical `o320 -> o1280` manual-eval flow.
#
# This helper validates the checkpoint profile, then renders run-specific sbatch
# launchers under `/home/ecm5702/dev/jobscripts/submit/<YYYYMMDD>/` and submits:
#   1) strict multi-bundle manual inference (4 GPU O1280 posture),
#   2) one-date local plots,
#   3) spectra (auto: proxy on AG, ECMWF on AC),
#   4) TC evaluation (auto: native on AG, regridded on AC),
# with `afterok` dependencies on the prediction job for the three downstream jobs.
#
# Usage:
#   bash submit_o320_o1280_manual_eval_flow.sh

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
CHECKPOINT_PATH="${CHECKPOINT_PATH:-REPLACE_CHECKPOINT_PATH}"
SOURCE_HPC="${SOURCE_HPC:-ag}"                        # ac | ag | leonardo | jupiter
INPUT_EVENT="${INPUT_EVENT:-idalia}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ecm5702/perm/eval}"

RUN_DATE_UTC="${RUN_DATE_UTC:-$(date -u +%Y%m%d)}"
RUN_SUFFIX="${RUN_SUFFIX:-manual_eval}"              # appended to the canonical run id

SAMPLER_JSON="${SAMPLER_JSON:-{\"num_steps\":40,\"sigma_max\":1000.0,\"sigma_min\":0.03,\"rho\":7.0,\"sampler\":\"heun\",\"S_max\":1000.0}}"
DATE_PRESET="${DATE_PRESET:-default}"                # default | aug16_30
DATES="${DATES:-}"                                   # explicit CSV override
STEPS="${STEPS:-24,48,72,96,120}"
BUNDLE_PAIRS="${BUNDLE_PAIRS:-}"                     # optional exact date:step list
MEMBERS="${MEMBERS:-1,2,3,4,5,6,7,8,9,10}"

LOCAL_PLOT_DATE="${LOCAL_PLOT_DATE:-}"               # blank => auto first selected date
LOCAL_PLOT_EXPECTED_COUNT="${LOCAL_PLOT_EXPECTED_COUNT:-auto}"  # auto | integer >= 0
LOCAL_PLOT_OUT_SUBDIR="${LOCAL_PLOT_OUT_SUBDIR:-local_plots_one_date}"  # under RUN_ROOT/

SPECTRA_METHOD="${SPECTRA_METHOD:-auto}"             # auto | proxy | ecmwf
TC_SUPPORT_MODE="${TC_SUPPORT_MODE:-auto}"           # auto | native | regridded
TC_EVENTS="${TC_EVENTS:-idalia}"
TC_EXTRA_REFERENCE_EXPIDS="${TC_EXTRA_REFERENCE_EXPIDS:-}"

HOLD="${HOLD:-0}"                                    # 1 => submit held
NO_SUBMIT="${NO_SUBMIT:-0}"                          # 1 => render only
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"              # 1 => overwrite rendered files
SUBMIT_ROOT="${SUBMIT_ROOT:-/home/ecm5702/dev/jobscripts/submit}"
PROFILE_PYTHON="${PROFILE_PYTHON:-}"                 # optional override for checkpoint_profile
###############################################################################

PROJECT_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"
INPUT_ROOT="/home/ecm5702/hpcperm/data/input_data/o320_o1280/${INPUT_EVENT}"

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

require_bool() {
  local value="$1"
  [[ "${value}" == "0" || "${value}" == "1" ]] || die "Expected 0 or 1, got '${value}'"
}

set_var() {
  local file="$1"
  local var="$2"
  local value="$3"
  python3 - "$file" "$var" "$value" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
var = sys.argv[2]
value = sys.argv[3]
escaped = value.replace("\\", "\\\\").replace('"', '\\"')
pattern = re.compile(rf"^{re.escape(var)}=.*$")
lines = path.read_text().splitlines()
updated = False
for idx, line in enumerate(lines):
    if pattern.match(line):
        lines[idx] = f'{var}="{escaped}"'
        updated = True
        break
if not updated:
    raise SystemExit(f"Variable not found in {path}: {var}")
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

set_sbatch_directive() {
  local file="$1"
  local key="$2"
  local value="$3"
  python3 - "$file" "$key" "$value" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
prefix = f"#SBATCH --{key}="
lines = path.read_text().splitlines()
for idx, line in enumerate(lines):
    if line.startswith(prefix):
        lines[idx] = f"{prefix}{value}"
        break
else:
    raise SystemExit(f"Directive not found in {path}: {prefix}")
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

drop_sbatch_directive() {
  local file="$1"
  local key="$2"
  python3 - "$file" "$key" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
prefix = f"#SBATCH --{key}="
lines = [line for line in path.read_text().splitlines() if not line.startswith(prefix)]
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

extract_job_id() {
  local submit_output="$1"
  local job_id
  job_id="$(printf '%s\n' "${submit_output}" | awk '{print $NF}')"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || die "Could not parse job id from sbatch output: ${submit_output}"
  printf '%s\n' "${job_id}"
}

require_choice "${SOURCE_HPC}" ac ag leonardo jupiter
require_choice "${DATE_PRESET}" default aug16_30
require_choice "${SPECTRA_METHOD}" auto proxy ecmwf
require_choice "${TC_SUPPORT_MODE}" auto native regridded
require_bool "${HOLD}"
require_bool "${NO_SUBMIT}"
require_bool "${ALLOW_OVERWRITE}"
[[ "${CHECKPOINT_PATH}" != REPLACE_* ]] || die "Set CHECKPOINT_PATH."
[[ -f "${INPUT_ROOT}" || -d "${INPUT_ROOT}" ]] || die "INPUT_ROOT does not exist: ${INPUT_ROOT}"
[[ "${RUN_SUFFIX}" =~ ^[A-Za-z0-9._-]*$ ]] || die "Unsafe RUN_SUFFIX: ${RUN_SUFFIX}"
if [[ "${LOCAL_PLOT_EXPECTED_COUNT}" != "auto" && ! "${LOCAL_PLOT_EXPECTED_COUNT}" =~ ^[0-9]+$ ]]; then
  die "LOCAL_PLOT_EXPECTED_COUNT must be 'auto' or a non-negative integer."
fi

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) HOST_FAMILY="ac" ;;
  ag*) HOST_FAMILY="ag" ;;
  *) die "Unsupported login node family (${HOST_SHORT}). Run from ac-* or ag-*." ;;
esac

if [[ -z "${PROFILE_PYTHON}" ]]; then
  if [[ "${HOST_FAMILY}" == "ac" ]]; then
    PROFILE_PYTHON="/home/ecm5702/dev/.ds-dyn/bin/python"
  else
    PROFILE_PYTHON="/home/ecm5702/dev/.ds-ag/bin/python"
  fi
fi
[[ -x "${PROFILE_PYTHON}" ]] || die "PROFILE_PYTHON is not executable: ${PROFILE_PYTHON}"

PROFILE_JSON="$(
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    "${PROFILE_PYTHON}" -m eval.jobs.checkpoint_profile \
      --name-ckpt "${CHECKPOINT_PATH}" \
      --source-hpc "${SOURCE_HPC}" \
      --host-short "${HOST_SHORT}" \
      --expected-lane o320_o1280 \
      --json
)"

mapfile -t PROFILE_FIELDS < <(
  python3 - "${PROFILE_JSON}" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(sys.argv[1])
ckpt_path = Path(payload["checkpoint_path"]).resolve()
print(str(ckpt_path))
print(payload["stack_flavor"])
print(payload["lane"])
print(payload["host_family"])
print(payload["recommended_venv"])
print(ckpt_path.parent.name[:8])
PY
)

RESOLVED_CKPT_PATH="${PROFILE_FIELDS[0]}"
STACK_FLAVOR="${PROFILE_FIELDS[1]}"
LANE="${PROFILE_FIELDS[2]}"
PROFILE_HOST_FAMILY="${PROFILE_FIELDS[3]}"
RECOMMENDED_VENV="${PROFILE_FIELDS[4]}"
CHECKPOINT_SHORT="${PROFILE_FIELDS[5]}"

[[ "${LANE}" == "o320_o1280" ]] || die "Expected o320_o1280 checkpoint lane, got ${LANE}"
[[ "${PROFILE_HOST_FAMILY}" == "${HOST_FAMILY}" ]] || die "Checkpoint-profile host mismatch: expected ${HOST_FAMILY}, got ${PROFILE_HOST_FAMILY}"

mapfile -t SCOPE_FIELDS < <(
  python3 - "${DATE_PRESET}" "${DATES}" "${STEPS}" "${BUNDLE_PAIRS}" "${LOCAL_PLOT_DATE}" "${LOCAL_PLOT_EXPECTED_COUNT}" <<'PY'
from collections import Counter
import sys

date_preset = sys.argv[1]
dates_raw = sys.argv[2]
steps_raw = sys.argv[3]
bundle_raw = sys.argv[4]
local_plot_date_raw = sys.argv[5]
expected_raw = sys.argv[6]

if dates_raw:
    dates = [token.strip() for token in dates_raw.split(",") if token.strip()]
elif date_preset == "default":
    dates = ["20230826", "20230827", "20230828", "20230829", "20230830"]
else:
    dates = [
        "20230816", "20230817", "20230818", "20230819", "20230820",
        "20230821", "20230822", "20230823", "20230824", "20230825",
        "20230826", "20230827", "20230828", "20230829", "20230830",
    ]

steps = [str(int(token.strip())) for token in steps_raw.split(",") if token.strip()]
if not steps:
    raise SystemExit("Resolved step list is empty.")

pair_tuples: list[tuple[str, str]] = []
seen: set[tuple[str, str]] = set()
for token in bundle_raw.split(","):
    token = token.strip()
    if not token:
        continue
    if ":" not in token:
        raise SystemExit(
            "BUNDLE_PAIRS entries must use date:step format, for example 20230828:24."
        )
    date_raw, step_raw = token.split(":", 1)
    date_value = date_raw.strip()
    step_value = str(int(step_raw.strip()))
    if len(date_value) != 8 or not date_value.isdigit():
        raise SystemExit(f"Invalid BUNDLE_PAIRS date {date_value!r}.")
    key = (date_value, step_value)
    if key in seen:
        continue
    seen.add(key)
    pair_tuples.append(key)

if pair_tuples:
    date_counts = Counter(date for date, _ in pair_tuples)
    resolved_dates = sorted(date_counts)
    resolved_steps = sorted({step for _, step in pair_tuples}, key=int)
    plot_date = local_plot_date_raw.strip() or resolved_dates[0]
    if plot_date not in date_counts:
        raise SystemExit(
            f"LOCAL_PLOT_DATE={plot_date} is not present in BUNDLE_PAIRS."
        )
    expected_auto = date_counts[plot_date]
else:
    resolved_dates = dates
    resolved_steps = sorted({step for step in steps}, key=int)
    plot_date = local_plot_date_raw.strip() or resolved_dates[0]
    if plot_date not in resolved_dates:
        raise SystemExit(
            f"LOCAL_PLOT_DATE={plot_date} is not present in the resolved date list."
        )
    expected_auto = len(resolved_steps)

if expected_raw == "auto":
    expected_count = expected_auto
else:
    expected_count = int(expected_raw)
    if expected_count < 0:
        raise SystemExit("LOCAL_PLOT_EXPECTED_COUNT must be >= 0.")

print(",".join(resolved_dates))
print(",".join(resolved_steps))
print(",".join(f"{date}:{step}" for date, step in pair_tuples))
print(plot_date)
print(str(expected_count))
print(str(len(pair_tuples)))
print(str(len(resolved_dates)))
print(str(len(resolved_steps)))
PY
)

RESOLVED_DATES="${SCOPE_FIELDS[0]}"
RESOLVED_STEPS="${SCOPE_FIELDS[1]}"
RESOLVED_BUNDLE_PAIRS="${SCOPE_FIELDS[2]}"
RESOLVED_LOCAL_PLOT_DATE="${SCOPE_FIELDS[3]}"
RESOLVED_LOCAL_PLOT_EXPECTED_COUNT="${SCOPE_FIELDS[4]}"
BUNDLE_PAIR_COUNT="${SCOPE_FIELDS[5]}"
DATE_COUNT="${SCOPE_FIELDS[6]}"
STEP_COUNT="${SCOPE_FIELDS[7]}"

if [[ "${SPECTRA_METHOD}" == "auto" ]]; then
  if [[ "${HOST_FAMILY}" == "ac" ]]; then
    RESOLVED_SPECTRA_METHOD="ecmwf"
  else
    RESOLVED_SPECTRA_METHOD="proxy"
  fi
else
  RESOLVED_SPECTRA_METHOD="${SPECTRA_METHOD}"
fi

if [[ "${TC_SUPPORT_MODE}" == "auto" ]]; then
  if [[ "${HOST_FAMILY}" == "ac" ]]; then
    RESOLVED_TC_SUPPORT_MODE="regridded"
  else
    RESOLVED_TC_SUPPORT_MODE="native"
  fi
else
  RESOLVED_TC_SUPPORT_MODE="${TC_SUPPORT_MODE}"
fi

if [[ "${RESOLVED_SPECTRA_METHOD}" == "ecmwf" && "${HOST_FAMILY}" != "ac" ]]; then
  die "SPECTRA_METHOD=ecmwf requires an ac login node."
fi
if [[ "${RESOLVED_TC_SUPPORT_MODE}" == "regridded" && "${HOST_FAMILY}" != "ac" ]]; then
  die "TC_SUPPORT_MODE=regridded requires an ac login node."
fi

RUN_ID="manual_${CHECKPOINT_SHORT}_${STACK_FLAVOR}_${LANE}_${RUN_DATE_UTC}"
if [[ -n "${RUN_SUFFIX}" ]]; then
  RUN_ID="${RUN_ID}_${RUN_SUFFIX}"
fi
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "Unsafe RUN_ID: ${RUN_ID}"

RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
PREDICTIONS_DIR="${RUN_ROOT}/predictions"

MEMBER_COUNT="$(
  python3 - "${MEMBERS}" <<'PY'
import sys
members = [token.strip() for token in sys.argv[1].split(",") if token.strip()]
print(len(members))
PY
)"

if [[ "${BUNDLE_PAIR_COUNT}" -gt 0 ]]; then
  SUBSET_TAG="pairs${BUNDLE_PAIR_COUNT}_${MEMBER_COUNT}m"
else
  SUBSET_TAG="${DATE_COUNT}d_${STEP_COUNT}s_${MEMBER_COUNT}m"
fi

TEMPLATE_DIR="${PROJECT_ROOT}/eval/jobs/templates"
INFER_TEMPLATE="${TEMPLATE_DIR}/strict_manual_predict_x_bundle.sbatch"
LOCAL_TEMPLATE="${TEMPLATE_DIR}/local_plots_one_date_from_predictions.sbatch"
PROXY_SPECTRA_TEMPLATE="${TEMPLATE_DIR}/spectra_proxy_from_predictions.sbatch"
ECMWF_SPECTRA_TEMPLATE="${TEMPLATE_DIR}/spectra_ecmwf_from_predictions.sbatch"
TC_TEMPLATE="${TEMPLATE_DIR}/tc_eval_from_predictions.sbatch"

[[ -f "${INFER_TEMPLATE}" ]] || die "Missing template: ${INFER_TEMPLATE}"
[[ -f "${LOCAL_TEMPLATE}" ]] || die "Missing template: ${LOCAL_TEMPLATE}"
[[ -f "${PROXY_SPECTRA_TEMPLATE}" ]] || die "Missing template: ${PROXY_SPECTRA_TEMPLATE}"
[[ -f "${ECMWF_SPECTRA_TEMPLATE}" ]] || die "Missing template: ${ECMWF_SPECTRA_TEMPLATE}"
[[ -f "${TC_TEMPLATE}" ]] || die "Missing template: ${TC_TEMPLATE}"

SUBMIT_DIR="${SUBMIT_ROOT}/${RUN_DATE_UTC}"
mkdir -p "${SUBMIT_DIR}"

PREDICT_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_predict.sbatch"
LOCAL_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_local_plots.sbatch"
SPECTRA_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_spectra.sbatch"
TC_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_tc_eval.sbatch"

for target in "${PREDICT_SCRIPT}" "${LOCAL_SCRIPT}" "${SPECTRA_SCRIPT}" "${TC_SCRIPT}"; do
  if [[ -e "${target}" && "${ALLOW_OVERWRITE}" -ne 1 ]]; then
    die "Refusing to overwrite existing generated file: ${target} (set ALLOW_OVERWRITE=1 to replace)"
  fi
done

cp "${INFER_TEMPLATE}" "${PREDICT_SCRIPT}"
cp "${LOCAL_TEMPLATE}" "${LOCAL_SCRIPT}"
if [[ "${RESOLVED_SPECTRA_METHOD}" == "proxy" ]]; then
  cp "${PROXY_SPECTRA_TEMPLATE}" "${SPECTRA_SCRIPT}"
else
  cp "${ECMWF_SPECTRA_TEMPLATE}" "${SPECTRA_SCRIPT}"
fi
cp "${TC_TEMPLATE}" "${TC_SCRIPT}"

set_var "${PREDICT_SCRIPT}" STACK_FLAVOR "${STACK_FLAVOR}"
set_var "${PREDICT_SCRIPT}" LANE "${LANE}"
set_var "${PREDICT_SCRIPT}" SOURCE_HPC "${SOURCE_HPC}"
set_var "${PREDICT_SCRIPT}" CHECKPOINT_REF "${RESOLVED_CKPT_PATH}"
set_var "${PREDICT_SCRIPT}" CHECKPOINT_SHORT "${CHECKPOINT_SHORT}"
set_var "${PREDICT_SCRIPT}" INPUT_ROOT "${INPUT_ROOT}"
set_var "${PREDICT_SCRIPT}" DATE_PRESET "${DATE_PRESET}"
set_var "${PREDICT_SCRIPT}" DATES "${RESOLVED_DATES}"
set_var "${PREDICT_SCRIPT}" STEPS "${RESOLVED_STEPS}"
set_var "${PREDICT_SCRIPT}" BUNDLE_PAIRS "${RESOLVED_BUNDLE_PAIRS}"
set_var "${PREDICT_SCRIPT}" MEMBERS "${MEMBERS}"
set_var "${PREDICT_SCRIPT}" OUTPUT_ROOT "${OUTPUT_ROOT}"
set_var "${PREDICT_SCRIPT}" RUN_DATE_UTC "${RUN_DATE_UTC}"
set_var "${PREDICT_SCRIPT}" RUN_SUFFIX "${RUN_SUFFIX}"
set_var "${PREDICT_SCRIPT}" NUM_GPUS_PER_MODEL "4"
set_var "${PREDICT_SCRIPT}" EXTRA_ARGS_JSON "${SAMPLER_JSON}"
set_sbatch_directive "${PREDICT_SCRIPT}" job-name "o1280_pred_${CHECKPOINT_SHORT}"
set_sbatch_directive "${PREDICT_SCRIPT}" ntasks-per-node "4"
set_sbatch_directive "${PREDICT_SCRIPT}" cpus-per-task "32"
set_sbatch_directive "${PREDICT_SCRIPT}" gpus-per-node "4"
set_sbatch_directive "${PREDICT_SCRIPT}" time "24:00:00"
set_sbatch_directive "${PREDICT_SCRIPT}" qos "ng"

set_var "${LOCAL_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
set_var "${LOCAL_SCRIPT}" RUN_ID "${RUN_ID}"
set_var "${LOCAL_SCRIPT}" DATE "${RESOLVED_LOCAL_PLOT_DATE}"
set_var "${LOCAL_SCRIPT}" OUT_SUBDIR "${LOCAL_PLOT_OUT_SUBDIR}"
set_var "${LOCAL_SCRIPT}" EXPECTED_COUNT "${RESOLVED_LOCAL_PLOT_EXPECTED_COUNT}"
set_sbatch_directive "${LOCAL_SCRIPT}" job-name "o1280_local_${CHECKPOINT_SHORT}"

set_var "${TC_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
set_var "${TC_SCRIPT}" RUN_ID "${RUN_ID}"
set_var "${TC_SCRIPT}" PREDICTIONS_DIR "${PREDICTIONS_DIR}"
set_var "${TC_SCRIPT}" EVENTS "${TC_EVENTS}"
set_var "${TC_SCRIPT}" SUPPORT_MODE "${RESOLVED_TC_SUPPORT_MODE}"
set_var "${TC_SCRIPT}" EXTRA_REFERENCE_EXPIDS "${TC_EXTRA_REFERENCE_EXPIDS}"
set_sbatch_directive "${TC_SCRIPT}" job-name "o1280_tc_${CHECKPOINT_SHORT}"

if [[ "${RESOLVED_SPECTRA_METHOD}" == "proxy" ]]; then
  set_var "${SPECTRA_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
  set_var "${SPECTRA_SCRIPT}" RUN_ID "${RUN_ID}"
  set_var "${SPECTRA_SCRIPT}" PREDICTIONS_SOURCE_DIR "${PREDICTIONS_DIR}"
  set_var "${SPECTRA_SCRIPT}" SUBSET_TAG "${SUBSET_TAG}"
  set_var "${SPECTRA_SCRIPT}" SUBSET_DIR "${RUN_ROOT}/predictions_subset_${SUBSET_TAG}"
  set_var "${SPECTRA_SCRIPT}" OUT_DIR "${RUN_ROOT}/spectra_proxy_${SUBSET_TAG}"
  set_var "${SPECTRA_SCRIPT}" DATE_LIST "${RESOLVED_DATES}"
  set_var "${SPECTRA_SCRIPT}" STEP_LIST "${RESOLVED_STEPS}"
else
  set_var "${SPECTRA_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
  set_var "${SPECTRA_SCRIPT}" RUN_ID "${RUN_ID}"
  set_var "${SPECTRA_SCRIPT}" PREDICTIONS_DIR "${PREDICTIONS_DIR}"
  set_var "${SPECTRA_SCRIPT}" DATE_LIST "${RESOLVED_DATES}"
  set_var "${SPECTRA_SCRIPT}" STEP_LIST "${RESOLVED_STEPS}"
fi
set_sbatch_directive "${SPECTRA_SCRIPT}" job-name "o1280_spectra_${CHECKPOINT_SHORT}"

if [[ "${HOST_FAMILY}" == "ac" ]]; then
  CPU_QOS="nf"
  drop_sbatch_directive "${LOCAL_SCRIPT}" gpus-per-node
  drop_sbatch_directive "${TC_SCRIPT}" gpus-per-node
  if [[ "${RESOLVED_SPECTRA_METHOD}" == "proxy" ]]; then
    drop_sbatch_directive "${SPECTRA_SCRIPT}" gpus-per-node
  fi
else
  CPU_QOS="ng"
  set_sbatch_directive "${LOCAL_SCRIPT}" gpus-per-node "0"
  set_sbatch_directive "${TC_SCRIPT}" gpus-per-node "0"
  if [[ "${RESOLVED_SPECTRA_METHOD}" == "proxy" ]]; then
    set_sbatch_directive "${SPECTRA_SCRIPT}" gpus-per-node "0"
  fi
fi
set_sbatch_directive "${LOCAL_SCRIPT}" qos "${CPU_QOS}"
set_sbatch_directive "${TC_SCRIPT}" qos "${CPU_QOS}"
if [[ "${RESOLVED_SPECTRA_METHOD}" == "proxy" ]]; then
  set_sbatch_directive "${SPECTRA_SCRIPT}" qos "${CPU_QOS}"
fi

bash -n "${PREDICT_SCRIPT}" "${LOCAL_SCRIPT}" "${SPECTRA_SCRIPT}" "${TC_SCRIPT}"

echo "[o1280-flow] checkpoint=${RESOLVED_CKPT_PATH}"
echo "[o1280-flow] stack=${STACK_FLAVOR} lane=${LANE}"
echo "[o1280-flow] host=${HOST_SHORT} host_family=${HOST_FAMILY}"
echo "[o1280-flow] recommended_venv=${RECOMMENDED_VENV}"
echo "[o1280-flow] input_root=${INPUT_ROOT}"
echo "[o1280-flow] run_id=${RUN_ID}"
echo "[o1280-flow] run_root=${RUN_ROOT}"
echo "[o1280-flow] dates=${RESOLVED_DATES}"
echo "[o1280-flow] steps=${RESOLVED_STEPS}"
echo "[o1280-flow] bundle_pairs=${RESOLVED_BUNDLE_PAIRS:-none}"
echo "[o1280-flow] local_plot_date=${RESOLVED_LOCAL_PLOT_DATE}"
echo "[o1280-flow] local_plot_expected_count=${RESOLVED_LOCAL_PLOT_EXPECTED_COUNT}"
echo "[o1280-flow] local_plot_out_dir=${RUN_ROOT}/${LOCAL_PLOT_OUT_SUBDIR}"
echo "[o1280-flow] spectra_method=${RESOLVED_SPECTRA_METHOD}"
echo "[o1280-flow] tc_support_mode=${RESOLVED_TC_SUPPORT_MODE}"
echo "[o1280-flow] generated_scripts:"
echo "  - ${PREDICT_SCRIPT}"
echo "  - ${LOCAL_SCRIPT}"
echo "  - ${SPECTRA_SCRIPT}"
echo "  - ${TC_SCRIPT}"

SBATCH_ARGS=()
if [[ "${HOLD}" == "1" ]]; then
  SBATCH_ARGS+=(--hold)
fi

if [[ "${NO_SUBMIT}" == "1" ]]; then
  echo "[o1280-flow] render-only mode enabled; not submitting"
  exit 0
fi

predict_submit="$(sbatch "${SBATCH_ARGS[@]}" "${PREDICT_SCRIPT}")"
predict_job="$(extract_job_id "${predict_submit}")"
echo "[o1280-flow] ${predict_submit}"

local_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${predict_job} "${LOCAL_SCRIPT}")"
local_job="$(extract_job_id "${local_submit}")"
echo "[o1280-flow] ${local_submit}"

spectra_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${predict_job} "${SPECTRA_SCRIPT}")"
spectra_job="$(extract_job_id "${spectra_submit}")"
echo "[o1280-flow] ${spectra_submit}"

tc_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${predict_job} "${TC_SCRIPT}")"
tc_job="$(extract_job_id "${tc_submit}")"
echo "[o1280-flow] ${tc_submit}"

cat <<EOF

=== O320 -> O1280 MANUAL EVAL FLOW SUBMITTED ===
run_id:             ${RUN_ID}
run_root:           ${RUN_ROOT}
checkpoint_path:    ${RESOLVED_CKPT_PATH}
submit_hold:        ${HOLD}
local_plots_dir:    ${RUN_ROOT}/${LOCAL_PLOT_OUT_SUBDIR}
spectra_method:     ${RESOLVED_SPECTRA_METHOD}
tc_support_mode:    ${RESOLVED_TC_SUPPORT_MODE}
generated_scripts:
  - ${PREDICT_SCRIPT}
  - ${LOCAL_SCRIPT}
  - ${SPECTRA_SCRIPT}
  - ${TC_SCRIPT}
job_ids:
  predict:          ${predict_job}
  local_plots:      ${local_job} (afterok:${predict_job})
  spectra:          ${spectra_job} (afterok:${predict_job})
  tc_eval:          ${tc_job} (afterok:${predict_job})

Monitor:
  squeue -j ${predict_job},${local_job},${spectra_job},${tc_job}
EOF
