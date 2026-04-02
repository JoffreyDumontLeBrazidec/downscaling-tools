#!/bin/bash
# Login-node helper: render and submit the canonical `o48 -> o96` manual-eval flow.
#
# This helper validates the checkpoint profile, then renders run-specific sbatch
# launchers under `/home/ecm5702/dev/jobscripts/submit/<YYYYMMDD>/` and submits:
#   1) truth-aware bundle rebuild into `${RUN_ROOT}/bundles_with_y`,
#   2) strict multi-bundle manual inference from that rebuilt bundle root,
#   3) MLflow loss plots (best-effort; writes status JSON if no match),
#   4) sigma evaluator sweep,
#   5) one-date local plots,
#   6) multi-region suites for selected Humberto steps,
#   7) storm-area contour suites,
#   8) spectra (auto: ECMWF on AC, proxy on AG),
# with `afterok` dependencies from prediction to the diagnostics that need
# `predictions_*.nc`.

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
CHECKPOINT_PATH="${CHECKPOINT_PATH:-REPLACE_CHECKPOINT_PATH}"
SOURCE_HPC="${SOURCE_HPC:-ac}"                        # ac | ag | leonardo | jupiter
INPUT_EVENT="${INPUT_EVENT:-humberto}"
SOURCE_GRIB_ROOT="${SOURCE_GRIB_ROOT:-/home/ecm5702/hpcperm/data/input_data/o48_o96/${INPUT_EVENT}_20250926_20250930}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ecm5702/perm/eval}"
PHASE="${PHASE:-proxy}"                              # proxy | continue-full | full-only
PREBUILT_BUNDLE_ROOT="${PREBUILT_BUNDLE_ROOT:-}"    # optional bundles_with_y* root to skip rebuild

RUN_DATE_UTC="${RUN_DATE_UTC:-$(date -u +%Y%m%d)}"
RUN_SUFFIX="${RUN_SUFFIX:-manual_eval}"
RUN_ID_OVERRIDE="${RUN_ID_OVERRIDE:-}"

PROXY_BUNDLE_PAIRS="${PROXY_BUNDLE_PAIRS:-20250926:24,20250926:48,20250927:24,20250927:48,20250928:24,20250928:48,20250929:24,20250929:48,20250930:24,20250930:48}"
SAMPLER_JSON="${SAMPLER_JSON:-{}}"
DATES="${DATES:-}"
STEPS="${STEPS:-24,48,72,96,120}"
BUNDLE_PAIRS="${BUNDLE_PAIRS:-}"
MEMBERS="${MEMBERS:-}"                               # blank => phase default (member 1)

MLFLOW_EXPERIMENT_DIR="${MLFLOW_EXPERIMENT_DIR:-/home/ecm5702/scratch/aifs/logs/mlflow/909682684414341917}"
TRAINING_RUN_FILTER="${TRAINING_RUN_FILTER:-}"       # blank => defaults to checkpoint short,lane
TRAINING_RUN_MIN_STEPS="${TRAINING_RUN_MIN_STEPS:-0}"

SIGMAS="${SIGMAS:-0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000}"
SIGMA_N_SAMPLES="${SIGMA_N_SAMPLES:-3}"
SIGMA_VALIDATION_FREQUENCY="${SIGMA_VALIDATION_FREQUENCY:-50h}"

RUN_ONE_DATE_LOCAL="${RUN_ONE_DATE_LOCAL:-1}"
LOCAL_PLOT_DATE="${LOCAL_PLOT_DATE:-}"               # blank => auto first selected date
LOCAL_PLOT_EXPECTED_COUNT="${LOCAL_PLOT_EXPECTED_COUNT:-auto}"
LOCAL_PLOT_OUT_SUBDIR="${LOCAL_PLOT_OUT_SUBDIR:-local_plots_one_date}"

RUN_REGIONAL_SUITES="${RUN_REGIONAL_SUITES:-1}"
REGIONAL_SUITE_DATE="${REGIONAL_SUITE_DATE:-}"       # blank => prefer 20250928 when available
REGIONAL_SUITE_STEPS="${REGIONAL_SUITE_STEPS:-24,48,120}"
REGIONAL_SUITE_REGION_NAMES="${REGIONAL_SUITE_REGION_NAMES:-amazon_forest_core,eastern_us_coast,andes_central,himalayas_central,maritime_continent,congo_basin}"
REGIONAL_SUITE_MODEL_VARIABLES="${REGIONAL_SUITE_MODEL_VARIABLES:-x_0,x_interp_0,y_0,y_pred_0,residuals_0,residuals_pred_0}"
REGIONAL_SUITE_OUT_PREFIX="${REGIONAL_SUITE_OUT_PREFIX:-local_plots_regions}"

RUN_STORM_PLOTS="${RUN_STORM_PLOTS:-1}"
STORM_PLOT_REGIONS="${STORM_PLOT_REGIONS:-eastern_us_coast,idalia_center}"
STORM_PLOT_OUT_PREFIX="${STORM_PLOT_OUT_PREFIX:-storm_local_plots}"

SPECTRA_METHOD="${SPECTRA_METHOD:-auto}"             # auto | proxy | ecmwf
SPECTRA_WEATHER_STATES="${SPECTRA_WEATHER_STATES:-10u,10v,2t,msl,t_850,z_500}"
SPECTRA_NSIDE="${SPECTRA_NSIDE:-64}"
SPECTRA_LMAX="${SPECTRA_LMAX:-95}"
SPECTRA_MEMBER_AGGREGATION="${SPECTRA_MEMBER_AGGREGATION:-per-file-mean}"

RUN_TC_PDF="${RUN_TC_PDF:-0}"                        # optional; set to 1 to include Humberto TC PDFs
TC_SUPPORT_MODE="${TC_SUPPORT_MODE:-auto}"           # auto | native | regridded
TC_EVENTS="${TC_EVENTS:-humberto}"
TC_EXTRA_REFERENCE_EXPIDS="${TC_EXTRA_REFERENCE_EXPIDS:-}"

HOLD="${HOLD:-0}"
NO_SUBMIT="${NO_SUBMIT:-0}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
SUBMIT_ROOT="${SUBMIT_ROOT:-/home/ecm5702/dev/jobscripts/submit}"
PROFILE_PYTHON="${PROFILE_PYTHON:-}"
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

require_source_gribs_for_date() {
  local root="$1"
  local date="$2"
  local required=(
    "${root}/enfo_o48_0001_date${date}_time0000_mem1to10_step24to120_sfc.grib"
    "${root}/enfo_o48_0001_date${date}_time0000_mem1to10_step24to120_pl.grib"
    "${root}/enfo_o96_0001_date${date}_time0000_step24to120_sfc.grib"
    "${root}/iekm_o96_iekm_date${date}_time0000_step24to120_sfc_y.grib"
    "${root}/iekm_o96_iekm_date${date}_time0000_step24to120_pl_y.grib"
  )
  local path
  for path in "${required[@]}"; do
    [[ -f "${path}" ]] || die "Missing required source GRIB: ${path}"
  done
}

require_choice "${SOURCE_HPC}" ac ag leonardo jupiter
require_choice "${PHASE}" proxy continue-full full-only
require_choice "${SPECTRA_METHOD}" auto proxy ecmwf
require_choice "${TC_SUPPORT_MODE}" auto native regridded
require_bool "${RUN_ONE_DATE_LOCAL}"
require_bool "${RUN_REGIONAL_SUITES}"
require_bool "${RUN_STORM_PLOTS}"
require_bool "${RUN_TC_PDF}"
require_bool "${HOLD}"
require_bool "${NO_SUBMIT}"
require_bool "${ALLOW_OVERWRITE}"
[[ "${CHECKPOINT_PATH}" != REPLACE_* ]] || die "Set CHECKPOINT_PATH."
[[ -d "${SOURCE_GRIB_ROOT}" ]] || die "SOURCE_GRIB_ROOT does not exist: ${SOURCE_GRIB_ROOT}"
[[ "${RUN_SUFFIX}" =~ ^[A-Za-z0-9._-]*$ ]] || die "Unsafe RUN_SUFFIX: ${RUN_SUFFIX}"
[[ "${RUN_ID_OVERRIDE}" =~ ^[A-Za-z0-9._-]*$ ]] || die "Unsafe RUN_ID_OVERRIDE: ${RUN_ID_OVERRIDE}"
if [[ "${PHASE}" == "continue-full" && -z "${RUN_ID_OVERRIDE}" ]]; then
  die "PHASE=continue-full requires RUN_ID_OVERRIDE so the full run reuses the proxy run id."
fi
if [[ "${LOCAL_PLOT_EXPECTED_COUNT}" != "auto" && ! "${LOCAL_PLOT_EXPECTED_COUNT}" =~ ^[0-9]+$ ]]; then
  die "LOCAL_PLOT_EXPECTED_COUNT must be 'auto' or a non-negative integer."
fi
[[ "${TRAINING_RUN_MIN_STEPS}" =~ ^[0-9]+$ ]] || die "TRAINING_RUN_MIN_STEPS must be a non-negative integer."

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
      --expected-lane o48_o96 \
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
print(ckpt_path.parent.name)
name = ckpt_path.name
print(name.removeprefix("inference-") if name.startswith("inference-") else name)
PY
)

RESOLVED_CKPT_PATH="${PROFILE_FIELDS[0]}"
STACK_FLAVOR="${PROFILE_FIELDS[1]}"
LANE="${PROFILE_FIELDS[2]}"
PROFILE_HOST_FAMILY="${PROFILE_FIELDS[3]}"
RECOMMENDED_VENV="${PROFILE_FIELDS[4]}"
CHECKPOINT_SHORT="${PROFILE_FIELDS[5]}"
CHECKPOINT_REF="${PROFILE_FIELDS[6]}"
SIGMA_CKPT_NAME="${PROFILE_FIELDS[7]}"

[[ "${LANE}" == "o48_o96" ]] || die "Expected o48_o96 checkpoint lane, got ${LANE}"
[[ "${PROFILE_HOST_FAMILY}" == "${HOST_FAMILY}" ]] || die "Checkpoint-profile host mismatch: expected ${HOST_FAMILY}, got ${PROFILE_HOST_FAMILY}"

if [[ -n "${MEMBERS}" ]]; then
  EFFECTIVE_MEMBERS="${MEMBERS}"
else
  EFFECTIVE_MEMBERS="1"
fi

if [[ "${PHASE}" == "proxy" && -z "${BUNDLE_PAIRS}" ]]; then
  EFFECTIVE_BUNDLE_PAIRS="${PROXY_BUNDLE_PAIRS}"
else
  EFFECTIVE_BUNDLE_PAIRS="${BUNDLE_PAIRS}"
fi

mapfile -t SCOPE_FIELDS < <(
  python3 - "${DATES}" "${STEPS}" "${EFFECTIVE_BUNDLE_PAIRS}" "${LOCAL_PLOT_DATE}" "${LOCAL_PLOT_EXPECTED_COUNT}" "${REGIONAL_SUITE_DATE}" "${REGIONAL_SUITE_STEPS}" "${PHASE}" <<'PY'
from collections import Counter
import sys

dates_raw, steps_raw, bundle_raw, local_plot_date_raw, expected_raw, regional_date_raw, regional_steps_raw, phase = sys.argv[1:9]

if dates_raw:
    dates = [token.strip() for token in dates_raw.split(",") if token.strip()]
else:
    dates = ["20250926", "20250927", "20250928", "20250929", "20250930"]

steps = [str(int(token.strip())) for token in steps_raw.split(",") if token.strip()]
if not steps:
    raise SystemExit("Resolved step list is empty.")

pair_tuples = []
seen = set()
for token in bundle_raw.split(","):
    token = token.strip()
    if not token:
        continue
    if ":" not in token:
        raise SystemExit("BUNDLE_PAIRS entries must use date:step format.")
    date_raw, step_raw = token.split(":", 1)
    date_value = date_raw.strip()
    step_value = str(int(step_raw.strip()))
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
        raise SystemExit(f"LOCAL_PLOT_DATE={plot_date} is not present in BUNDLE_PAIRS.")
    expected_auto = date_counts[plot_date]
else:
    resolved_dates = dates
    resolved_steps = sorted({step for step in steps}, key=int)
    plot_date = local_plot_date_raw.strip() or resolved_dates[0]
    if plot_date not in resolved_dates:
        raise SystemExit(f"LOCAL_PLOT_DATE={plot_date} is not present in the resolved date list.")
    expected_auto = len(resolved_steps)

if expected_raw == "auto":
    expected_count = expected_auto
else:
    expected_count = int(expected_raw)

preferred_regional_date = regional_date_raw.strip() or ("20250928" if "20250928" in resolved_dates else plot_date)
if preferred_regional_date not in resolved_dates:
    raise SystemExit(f"REGIONAL_SUITE_DATE={preferred_regional_date} is not present in the resolved date list.")

requested_regional_steps = [str(int(token.strip())) for token in regional_steps_raw.split(",") if token.strip()]
selected_regional_steps = [step for step in requested_regional_steps if step in resolved_steps]

print(",".join(resolved_dates))
print(",".join(resolved_steps))
print(",".join(f"{date}:{step}" for date, step in pair_tuples))
print(plot_date)
print(str(expected_count))
print(str(len(pair_tuples)))
print(preferred_regional_date)
print(",".join(selected_regional_steps))
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
RESOLVED_REGIONAL_SUITE_DATE="${SCOPE_FIELDS[6]}"
RESOLVED_REGIONAL_SUITE_STEPS="${SCOPE_FIELDS[7]}"
DATE_COUNT="${SCOPE_FIELDS[8]}"
STEP_COUNT="${SCOPE_FIELDS[9]}"

if [[ "${SPECTRA_METHOD}" == "auto" ]]; then
  if [[ "${HOST_FAMILY}" == "ac" ]]; then
    RESOLVED_SPECTRA_METHOD="ecmwf"
  else
    RESOLVED_SPECTRA_METHOD="proxy"
  fi
else
  RESOLVED_SPECTRA_METHOD="${SPECTRA_METHOD}"
fi
if [[ "${RESOLVED_SPECTRA_METHOD}" == "ecmwf" && "${HOST_FAMILY}" != "ac" ]]; then
  die "SPECTRA_METHOD=ecmwf requires an ac login node."
fi
if [[ "${RESOLVED_SPECTRA_METHOD}" == "ecmwf" ]]; then
  RESOLVED_ECMWF_SPECTRA_WEATHER_STATES="$(python3 - "${SPECTRA_WEATHER_STATES}" <<'PY'
import sys
supported = {"10u", "10v", "2t", "sp", "t_850", "z_500"}
seen = []
for token in sys.argv[1].split(","):
    token = token.strip()
    if token and token in supported and token not in seen:
        seen.append(token)
print(",".join(seen))
PY
)"
  [[ -n "${RESOLVED_ECMWF_SPECTRA_WEATHER_STATES}" ]] || \
    die "SPECTRA_WEATHER_STATES has no ECMWF-compatible entries: ${SPECTRA_WEATHER_STATES}"
else
  RESOLVED_ECMWF_SPECTRA_WEATHER_STATES=""
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
if [[ "${RUN_TC_PDF}" == "1" && "${RESOLVED_TC_SUPPORT_MODE}" == "regridded" && "${HOST_FAMILY}" != "ac" ]]; then
  die "RUN_TC_PDF=1 with TC_SUPPORT_MODE=regridded requires an ac login node."
fi

if [[ -n "${RUN_ID_OVERRIDE}" ]]; then
  RUN_ID="${RUN_ID_OVERRIDE}"
else
  RUN_ID="manual_${CHECKPOINT_SHORT}_${STACK_FLAVOR}_${LANE}_${RUN_DATE_UTC}"
  if [[ -n "${RUN_SUFFIX}" ]]; then
    RUN_ID="${RUN_ID}_${RUN_SUFFIX}"
  fi
fi
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "Unsafe RUN_ID: ${RUN_ID}"

RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
if [[ -n "${PREBUILT_BUNDLE_ROOT}" ]]; then
  [[ -d "${PREBUILT_BUNDLE_ROOT}" ]] || die "PREBUILT_BUNDLE_ROOT does not exist: ${PREBUILT_BUNDLE_ROOT}"
  case "$(basename "${PREBUILT_BUNDLE_ROOT}")" in
    bundles_with_y*) ;;
    *) die "PREBUILT_BUNDLE_ROOT basename must look like bundles_with_y*: ${PREBUILT_BUNDLE_ROOT}" ;;
  esac
  BUNDLE_DIR="${PREBUILT_BUNDLE_ROOT}"
  USE_PREBUILT_BUNDLES=1
else
  BUNDLE_DIR="${RUN_ROOT}/bundles_with_y"
  USE_PREBUILT_BUNDLES=0
fi
PREDICTIONS_DIR="${RUN_ROOT}/predictions"

if [[ "${PHASE}" == "continue-full" && ! -d "${RUN_ROOT}" ]]; then
  die "PHASE=continue-full requires an existing run root: ${RUN_ROOT}"
fi

if [[ -z "${TRAINING_RUN_FILTER}" ]]; then
  RESOLVED_TRAINING_RUN_FILTER="${CHECKPOINT_SHORT},${LANE}"
else
  RESOLVED_TRAINING_RUN_FILTER="${TRAINING_RUN_FILTER}"
fi

if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  while IFS= read -r date; do
    [[ -n "${date}" ]] || continue
    require_source_gribs_for_date "${SOURCE_GRIB_ROOT}" "${date}"
  done < <(python3 - "${RESOLVED_DATES}" <<'PY'
import sys
for token in sys.argv[1].split(","):
    token = token.strip()
    if token:
        print(token)
PY
)
fi

MEMBER_COUNT="$(
  python3 - "${EFFECTIVE_MEMBERS}" <<'PY'
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
BUILD_TEMPLATE="${TEMPLATE_DIR}/build_o48_o96_truth_bundles.sbatch"
INFER_TEMPLATE="${TEMPLATE_DIR}/strict_manual_predict_x_bundle.sbatch"
LOSS_TEMPLATE="${TEMPLATE_DIR}/training_loss_plots_from_mlflow.sbatch"
SIGMA_TEMPLATE="${TEMPLATE_DIR}/scoreboard_sigma_eval.sbatch"
LOCAL_TEMPLATE="${TEMPLATE_DIR}/local_plots_one_date_from_predictions.sbatch"
REGIONAL_TEMPLATE="${TEMPLATE_DIR}/regional_suite_from_predictions.sbatch"
PROXY_SPECTRA_TEMPLATE="${TEMPLATE_DIR}/spectra_proxy_from_predictions.sbatch"
ECMWF_SPECTRA_TEMPLATE="${TEMPLATE_DIR}/spectra_ecmwf_from_predictions.sbatch"
TC_TEMPLATE="${TEMPLATE_DIR}/tc_eval_from_predictions.sbatch"

[[ -f "${INFER_TEMPLATE}" ]] || die "Missing template: ${INFER_TEMPLATE}"
[[ -f "${LOSS_TEMPLATE}" ]] || die "Missing template: ${LOSS_TEMPLATE}"
[[ -f "${SIGMA_TEMPLATE}" ]] || die "Missing template: ${SIGMA_TEMPLATE}"
[[ -f "${LOCAL_TEMPLATE}" ]] || die "Missing template: ${LOCAL_TEMPLATE}"
[[ -f "${REGIONAL_TEMPLATE}" ]] || die "Missing template: ${REGIONAL_TEMPLATE}"
[[ -f "${PROXY_SPECTRA_TEMPLATE}" ]] || die "Missing template: ${PROXY_SPECTRA_TEMPLATE}"
[[ -f "${ECMWF_SPECTRA_TEMPLATE}" ]] || die "Missing template: ${ECMWF_SPECTRA_TEMPLATE}"
if [[ "${RUN_TC_PDF}" == "1" ]]; then
  [[ -f "${TC_TEMPLATE}" ]] || die "Missing template: ${TC_TEMPLATE}"
fi
if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  [[ -f "${BUILD_TEMPLATE}" ]] || die "Missing template: ${BUILD_TEMPLATE}"
fi

SUBMIT_DIR="${SUBMIT_ROOT}/${RUN_DATE_UTC}"
mkdir -p "${SUBMIT_DIR}"

BUILD_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_build_truth_bundles.sbatch"
PREDICT_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_predict.sbatch"
LOSS_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_training_loss.sbatch"
SIGMA_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_sigma_eval.sbatch"
LOCAL_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_local_plots.sbatch"
SPECTRA_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_spectra.sbatch"
TC_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_tc_eval.sbatch"
REGIONAL_SCRIPTS=()
STORM_SCRIPTS=()

if [[ "${RUN_REGIONAL_SUITES}" == "1" && -n "${RESOLVED_REGIONAL_SUITE_STEPS}" ]]; then
  while IFS= read -r step; do
    [[ -n "${step}" ]] || continue
    step_padded="$(printf '%03d' "${step}")"
    REGIONAL_SCRIPTS+=("${SUBMIT_DIR}/${RUN_ID}_regions_step${step_padded}.sbatch")
  done < <(python3 - "${RESOLVED_REGIONAL_SUITE_STEPS}" <<'PY'
import sys
for token in sys.argv[1].split(","):
    token = token.strip()
    if token:
        print(token)
PY
)
fi

if [[ "${RUN_STORM_PLOTS}" == "1" && -n "${RESOLVED_REGIONAL_SUITE_STEPS}" ]]; then
  while IFS= read -r step; do
    [[ -n "${step}" ]] || continue
    step_padded="$(printf '%03d' "${step}")"
    STORM_SCRIPTS+=("${SUBMIT_DIR}/${RUN_ID}_storms_step${step_padded}.sbatch")
  done < <(python3 - "${RESOLVED_REGIONAL_SUITE_STEPS}" <<'PY'
import sys
for token in sys.argv[1].split(","):
    token = token.strip()
    if token:
        print(token)
PY
)
fi

TARGET_SCRIPTS=("${PREDICT_SCRIPT}" "${LOSS_SCRIPT}" "${SIGMA_SCRIPT}" "${LOCAL_SCRIPT}" "${SPECTRA_SCRIPT}")
if [[ "${RUN_TC_PDF}" == "1" ]]; then
  TARGET_SCRIPTS+=("${TC_SCRIPT}")
fi
if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  TARGET_SCRIPTS=("${BUILD_SCRIPT}" "${TARGET_SCRIPTS[@]}")
fi
TARGET_SCRIPTS+=("${REGIONAL_SCRIPTS[@]}" "${STORM_SCRIPTS[@]}")
for target in "${TARGET_SCRIPTS[@]}"; do
  if [[ -e "${target}" && "${ALLOW_OVERWRITE}" -ne 1 ]]; then
    die "Refusing to overwrite existing generated file: ${target} (set ALLOW_OVERWRITE=1 to replace)"
  fi
done

if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  cp "${BUILD_TEMPLATE}" "${BUILD_SCRIPT}"
fi
cp "${INFER_TEMPLATE}" "${PREDICT_SCRIPT}"
cp "${LOSS_TEMPLATE}" "${LOSS_SCRIPT}"
cp "${SIGMA_TEMPLATE}" "${SIGMA_SCRIPT}"
cp "${LOCAL_TEMPLATE}" "${LOCAL_SCRIPT}"
if [[ "${RESOLVED_SPECTRA_METHOD}" == "proxy" ]]; then
  cp "${PROXY_SPECTRA_TEMPLATE}" "${SPECTRA_SCRIPT}"
else
  cp "${ECMWF_SPECTRA_TEMPLATE}" "${SPECTRA_SCRIPT}"
fi
if [[ "${RUN_TC_PDF}" == "1" ]]; then
  cp "${TC_TEMPLATE}" "${TC_SCRIPT}"
fi
for regional_script in "${REGIONAL_SCRIPTS[@]}"; do
  cp "${REGIONAL_TEMPLATE}" "${regional_script}"
done
for storm_script in "${STORM_SCRIPTS[@]}"; do
  cp "${REGIONAL_TEMPLATE}" "${storm_script}"
done

if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  set_var "${BUILD_SCRIPT}" STACK_FLAVOR "${STACK_FLAVOR}"
  set_var "${BUILD_SCRIPT}" SOURCE_HPC "${SOURCE_HPC}"
  set_var "${BUILD_SCRIPT}" CHECKPOINT_SHORT "${CHECKPOINT_SHORT}"
  set_var "${BUILD_SCRIPT}" SOURCE_GRIB_ROOT "${SOURCE_GRIB_ROOT}"
  set_var "${BUILD_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
  set_var "${BUILD_SCRIPT}" BUNDLE_DIR "${BUNDLE_DIR}"
  set_var "${BUILD_SCRIPT}" DATES "${RESOLVED_DATES}"
  set_var "${BUILD_SCRIPT}" STEPS "${RESOLVED_STEPS}"
  set_var "${BUILD_SCRIPT}" BUNDLE_PAIRS "${RESOLVED_BUNDLE_PAIRS}"
  set_var "${BUILD_SCRIPT}" MEMBERS "${EFFECTIVE_MEMBERS}"
  set_sbatch_directive "${BUILD_SCRIPT}" job-name "o48_bundle_${CHECKPOINT_SHORT}"
fi

set_var "${PREDICT_SCRIPT}" STACK_FLAVOR "${STACK_FLAVOR}"
set_var "${PREDICT_SCRIPT}" LANE "${LANE}"
set_var "${PREDICT_SCRIPT}" SOURCE_HPC "${SOURCE_HPC}"
set_var "${PREDICT_SCRIPT}" CHECKPOINT_REF "${RESOLVED_CKPT_PATH}"
set_var "${PREDICT_SCRIPT}" CHECKPOINT_SHORT "${CHECKPOINT_SHORT}"
set_var "${PREDICT_SCRIPT}" INPUT_ROOT "${BUNDLE_DIR}"
set_var "${PREDICT_SCRIPT}" DATE_PRESET "default"
set_var "${PREDICT_SCRIPT}" DATES "${RESOLVED_DATES}"
set_var "${PREDICT_SCRIPT}" STEPS "${RESOLVED_STEPS}"
set_var "${PREDICT_SCRIPT}" BUNDLE_PAIRS "${RESOLVED_BUNDLE_PAIRS}"
set_var "${PREDICT_SCRIPT}" MEMBERS "${EFFECTIVE_MEMBERS}"
set_var "${PREDICT_SCRIPT}" OUTPUT_ROOT "${OUTPUT_ROOT}"
set_var "${PREDICT_SCRIPT}" RUN_DATE_UTC "${RUN_DATE_UTC}"
set_var "${PREDICT_SCRIPT}" RUN_SUFFIX "${RUN_SUFFIX}"
set_var "${PREDICT_SCRIPT}" RUN_ID_OVERRIDE "${RUN_ID}"
set_var "${PREDICT_SCRIPT}" NUM_GPUS_PER_MODEL "1"
set_var "${PREDICT_SCRIPT}" ALLOW_EXISTING_RUN_DIR "1"
set_var "${PREDICT_SCRIPT}" ALLOW_REBUILT_BUNDLE_ROOT "1"
set_var "${PREDICT_SCRIPT}" EXTRA_ARGS_JSON "${SAMPLER_JSON}"
set_sbatch_directive "${PREDICT_SCRIPT}" job-name "o48_pred_${CHECKPOINT_SHORT}"
set_sbatch_directive "${PREDICT_SCRIPT}" ntasks-per-node "1"
set_sbatch_directive "${PREDICT_SCRIPT}" cpus-per-task "16"
set_sbatch_directive "${PREDICT_SCRIPT}" gpus-per-node "1"
set_sbatch_directive "${PREDICT_SCRIPT}" time "12:00:00"
set_sbatch_directive "${PREDICT_SCRIPT}" qos "ng"

set_var "${LOSS_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
set_var "${LOSS_SCRIPT}" RUN_ID "${RUN_ID}"
set_var "${LOSS_SCRIPT}" MLFLOW_EXPERIMENT_DIR "${MLFLOW_EXPERIMENT_DIR}"
set_var "${LOSS_SCRIPT}" MIN_STEPS "${TRAINING_RUN_MIN_STEPS}"
set_var "${LOSS_SCRIPT}" NAME_FILTER "${RESOLVED_TRAINING_RUN_FILTER}"
set_sbatch_directive "${LOSS_SCRIPT}" job-name "o48_loss_${CHECKPOINT_SHORT}"

set_var "${SIGMA_SCRIPT}" RUN_ID "${RUN_ID}"
set_var "${SIGMA_SCRIPT}" CHECKPOINT_REF "${CHECKPOINT_REF}"
set_var "${SIGMA_SCRIPT}" CKPT_NAME "${SIGMA_CKPT_NAME}"
set_var "${SIGMA_SCRIPT}" EXPECTED_STACK_FLAVOR "${STACK_FLAVOR}"
set_var "${SIGMA_SCRIPT}" OUT_CSV "${RUN_ROOT}/sigma_eval_table.csv"
set_var "${SIGMA_SCRIPT}" SIGMAS "${SIGMAS}"
set_var "${SIGMA_SCRIPT}" N_SAMPLES "${SIGMA_N_SAMPLES}"
set_var "${SIGMA_SCRIPT}" VALIDATION_FREQUENCY "${SIGMA_VALIDATION_FREQUENCY}"
set_var "${SIGMA_SCRIPT}" EXPECTED_LANE "${LANE}"
set_sbatch_directive "${SIGMA_SCRIPT}" job-name "o48_sigma_${CHECKPOINT_SHORT}"

set_var "${LOCAL_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
set_var "${LOCAL_SCRIPT}" RUN_ID "${RUN_ID}"
set_var "${LOCAL_SCRIPT}" DATE "${RESOLVED_LOCAL_PLOT_DATE}"
set_var "${LOCAL_SCRIPT}" OUT_SUBDIR "${LOCAL_PLOT_OUT_SUBDIR}"
set_var "${LOCAL_SCRIPT}" EXPECTED_COUNT "${RESOLVED_LOCAL_PLOT_EXPECTED_COUNT}"
set_var "${LOCAL_SCRIPT}" MODEL_VARIABLES "${REGIONAL_SUITE_MODEL_VARIABLES}"
set_sbatch_directive "${LOCAL_SCRIPT}" job-name "o48_local_${CHECKPOINT_SHORT}"

if [[ "${RESOLVED_SPECTRA_METHOD}" == "proxy" ]]; then
  set_var "${SPECTRA_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
  set_var "${SPECTRA_SCRIPT}" RUN_ID "${RUN_ID}"
  set_var "${SPECTRA_SCRIPT}" PREDICTIONS_SOURCE_DIR "${PREDICTIONS_DIR}"
  set_var "${SPECTRA_SCRIPT}" SUBSET_TAG "${SUBSET_TAG}"
  set_var "${SPECTRA_SCRIPT}" SUBSET_DIR "${RUN_ROOT}/predictions_subset_${SUBSET_TAG}"
  set_var "${SPECTRA_SCRIPT}" OUT_DIR "${RUN_ROOT}/spectra_proxy_${SUBSET_TAG}"
  set_var "${SPECTRA_SCRIPT}" DATE_LIST "${RESOLVED_DATES}"
  set_var "${SPECTRA_SCRIPT}" STEP_LIST "${RESOLVED_STEPS}"
  set_var "${SPECTRA_SCRIPT}" WEATHER_STATES "${SPECTRA_WEATHER_STATES}"
  set_var "${SPECTRA_SCRIPT}" NSIDE "${SPECTRA_NSIDE}"
  set_var "${SPECTRA_SCRIPT}" LMAX "${SPECTRA_LMAX}"
  set_var "${SPECTRA_SCRIPT}" MEMBER_AGGREGATION "${SPECTRA_MEMBER_AGGREGATION}"
else
  set_var "${SPECTRA_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
  set_var "${SPECTRA_SCRIPT}" RUN_ID "${RUN_ID}"
  set_var "${SPECTRA_SCRIPT}" PREDICTIONS_DIR "${PREDICTIONS_DIR}"
  set_var "${SPECTRA_SCRIPT}" DATE_LIST "${RESOLVED_DATES}"
  set_var "${SPECTRA_SCRIPT}" STEP_LIST "${RESOLVED_STEPS}"
  set_var "${SPECTRA_SCRIPT}" MEMBER_LIST "${EFFECTIVE_MEMBERS}"
  set_var "${SPECTRA_SCRIPT}" WEATHER_STATES "${RESOLVED_ECMWF_SPECTRA_WEATHER_STATES}"
  set_var "${SPECTRA_SCRIPT}" TEMPLATE_ROOT "/home/ecm5702/hpcperm/reference_spectra/eefo_o96"
  set_var "${SPECTRA_SCRIPT}" TEMPLATE_GRIB_ROOT "${SOURCE_GRIB_ROOT}"
fi
set_sbatch_directive "${SPECTRA_SCRIPT}" job-name "o48_spectra_${CHECKPOINT_SHORT}"

if [[ "${RUN_TC_PDF}" == "1" ]]; then
  set_var "${TC_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
  set_var "${TC_SCRIPT}" RUN_ID "${RUN_ID}"
  set_var "${TC_SCRIPT}" PREDICTIONS_DIR "${PREDICTIONS_DIR}"
  set_var "${TC_SCRIPT}" EVENTS "${TC_EVENTS}"
  set_var "${TC_SCRIPT}" SUPPORT_MODE "${RESOLVED_TC_SUPPORT_MODE}"
  set_var "${TC_SCRIPT}" EXTRA_REFERENCE_EXPIDS "${TC_EXTRA_REFERENCE_EXPIDS}"
  set_sbatch_directive "${TC_SCRIPT}" job-name "o48_tc_${CHECKPOINT_SHORT}"
fi

for regional_script in "${REGIONAL_SCRIPTS[@]}"; do
  step_padded="$(basename "${regional_script}" | sed -n 's/.*_step\([0-9][0-9][0-9]\)\.sbatch/\1/p')"
  [[ -n "${step_padded}" ]] || die "Could not parse step from regional script: ${regional_script}"
  set_var "${regional_script}" RUN_ROOT "${RUN_ROOT}"
  set_var "${regional_script}" RUN_ID "${RUN_ID}"
  set_var "${regional_script}" PREDICTIONS_FILE "${PREDICTIONS_DIR}/predictions_${RESOLVED_REGIONAL_SUITE_DATE}_step${step_padded}.nc"
  set_var "${regional_script}" OUT_DIR "${RUN_ROOT}/${REGIONAL_SUITE_OUT_PREFIX}_step${step_padded}"
  set_var "${regional_script}" REGION_NAMES "${REGIONAL_SUITE_REGION_NAMES}"
  set_var "${regional_script}" MODEL_VARIABLES "${REGIONAL_SUITE_MODEL_VARIABLES}"
  set_sbatch_directive "${regional_script}" job-name "o48_regions${step_padded}_${CHECKPOINT_SHORT}"
done

for storm_script in "${STORM_SCRIPTS[@]}"; do
  step_padded="$(basename "${storm_script}" | sed -n 's/.*_step\([0-9][0-9][0-9]\)\.sbatch/\1/p')"
  [[ -n "${step_padded}" ]] || die "Could not parse step from storm script: ${storm_script}"
  set_var "${storm_script}" RUN_ROOT "${RUN_ROOT}"
  set_var "${storm_script}" RUN_ID "${RUN_ID}"
  set_var "${storm_script}" PREDICTIONS_FILE "${PREDICTIONS_DIR}/predictions_${RESOLVED_REGIONAL_SUITE_DATE}_step${step_padded}.nc"
  set_var "${storm_script}" OUT_DIR "${RUN_ROOT}/${STORM_PLOT_OUT_PREFIX}_step${step_padded}"
  set_var "${storm_script}" REGION_NAMES "${STORM_PLOT_REGIONS}"
  set_var "${storm_script}" MODEL_VARIABLES "${REGIONAL_SUITE_MODEL_VARIABLES}"
  set_sbatch_directive "${storm_script}" job-name "o48_storms${step_padded}_${CHECKPOINT_SHORT}"
done

if [[ "${HOST_FAMILY}" == "ac" ]]; then
  CPU_QOS="nf"
  if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
    drop_sbatch_directive "${BUILD_SCRIPT}" gpus-per-node
  fi
  drop_sbatch_directive "${LOSS_SCRIPT}" gpus-per-node 2>/dev/null || true
  drop_sbatch_directive "${LOCAL_SCRIPT}" gpus-per-node
  for regional_script in "${REGIONAL_SCRIPTS[@]}"; do
    drop_sbatch_directive "${regional_script}" gpus-per-node
  done
  for storm_script in "${STORM_SCRIPTS[@]}"; do
    drop_sbatch_directive "${storm_script}" gpus-per-node
  done
  if [[ "${RESOLVED_SPECTRA_METHOD}" == "proxy" ]]; then
    drop_sbatch_directive "${SPECTRA_SCRIPT}" gpus-per-node
  fi
  if [[ "${RUN_TC_PDF}" == "1" ]]; then
    drop_sbatch_directive "${TC_SCRIPT}" gpus-per-node
  fi
else
  CPU_QOS="ng"
  if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
    set_sbatch_directive "${BUILD_SCRIPT}" gpus-per-node "0"
  fi
  set_sbatch_directive "${LOCAL_SCRIPT}" gpus-per-node "0"
  for regional_script in "${REGIONAL_SCRIPTS[@]}"; do
    set_sbatch_directive "${regional_script}" gpus-per-node "0"
  done
  for storm_script in "${STORM_SCRIPTS[@]}"; do
    set_sbatch_directive "${storm_script}" gpus-per-node "0"
  done
  if [[ "${RESOLVED_SPECTRA_METHOD}" == "proxy" ]]; then
    set_sbatch_directive "${SPECTRA_SCRIPT}" gpus-per-node "0"
  fi
  if [[ "${RUN_TC_PDF}" == "1" ]]; then
    set_sbatch_directive "${TC_SCRIPT}" gpus-per-node "0"
  fi
fi
if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  set_sbatch_directive "${BUILD_SCRIPT}" qos "${CPU_QOS}"
fi
set_sbatch_directive "${LOSS_SCRIPT}" qos "${CPU_QOS}"
set_sbatch_directive "${LOCAL_SCRIPT}" qos "${CPU_QOS}"
for regional_script in "${REGIONAL_SCRIPTS[@]}"; do
  set_sbatch_directive "${regional_script}" qos "${CPU_QOS}"
done
for storm_script in "${STORM_SCRIPTS[@]}"; do
  set_sbatch_directive "${storm_script}" qos "${CPU_QOS}"
done
if [[ "${RESOLVED_SPECTRA_METHOD}" == "proxy" ]]; then
  set_sbatch_directive "${SPECTRA_SCRIPT}" qos "${CPU_QOS}"
fi
if [[ "${RUN_TC_PDF}" == "1" ]]; then
  set_sbatch_directive "${TC_SCRIPT}" qos "${CPU_QOS}"
fi

if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  if [[ "${RUN_TC_PDF}" == "1" ]]; then
    bash -n "${BUILD_SCRIPT}" "${PREDICT_SCRIPT}" "${LOSS_SCRIPT}" "${SIGMA_SCRIPT}" "${LOCAL_SCRIPT}" "${SPECTRA_SCRIPT}" "${TC_SCRIPT}" "${REGIONAL_SCRIPTS[@]}" "${STORM_SCRIPTS[@]}"
  else
    bash -n "${BUILD_SCRIPT}" "${PREDICT_SCRIPT}" "${LOSS_SCRIPT}" "${SIGMA_SCRIPT}" "${LOCAL_SCRIPT}" "${SPECTRA_SCRIPT}" "${REGIONAL_SCRIPTS[@]}" "${STORM_SCRIPTS[@]}"
  fi
else
  if [[ "${RUN_TC_PDF}" == "1" ]]; then
    bash -n "${PREDICT_SCRIPT}" "${LOSS_SCRIPT}" "${SIGMA_SCRIPT}" "${LOCAL_SCRIPT}" "${SPECTRA_SCRIPT}" "${TC_SCRIPT}" "${REGIONAL_SCRIPTS[@]}" "${STORM_SCRIPTS[@]}"
  else
    bash -n "${PREDICT_SCRIPT}" "${LOSS_SCRIPT}" "${SIGMA_SCRIPT}" "${LOCAL_SCRIPT}" "${SPECTRA_SCRIPT}" "${REGIONAL_SCRIPTS[@]}" "${STORM_SCRIPTS[@]}"
  fi
fi

echo "[o48-flow] checkpoint=${RESOLVED_CKPT_PATH}"
echo "[o48-flow] sigma_ckpt=${CHECKPOINT_REF}/${SIGMA_CKPT_NAME}"
echo "[o48-flow] stack=${STACK_FLAVOR} lane=${LANE}"
echo "[o48-flow] host=${HOST_SHORT} host_family=${HOST_FAMILY}"
echo "[o48-flow] recommended_venv=${RECOMMENDED_VENV}"
echo "[o48-flow] phase=${PHASE}"
echo "[o48-flow] source_grib_root=${SOURCE_GRIB_ROOT}"
echo "[o48-flow] bundle_dir=${BUNDLE_DIR}"
echo "[o48-flow] use_prebuilt_bundles=${USE_PREBUILT_BUNDLES}"
echo "[o48-flow] run_id=${RUN_ID}"
echo "[o48-flow] run_root=${RUN_ROOT}"
echo "[o48-flow] dates=${RESOLVED_DATES}"
echo "[o48-flow] steps=${RESOLVED_STEPS}"
echo "[o48-flow] bundle_pairs=${RESOLVED_BUNDLE_PAIRS:-none}"
echo "[o48-flow] members=${EFFECTIVE_MEMBERS}"
echo "[o48-flow] training_run_filter=${RESOLVED_TRAINING_RUN_FILTER}"
echo "[o48-flow] sigmas=${SIGMAS}"
echo "[o48-flow] local_plot_date=${RESOLVED_LOCAL_PLOT_DATE}"
echo "[o48-flow] regional_suite_date=${RESOLVED_REGIONAL_SUITE_DATE}"
echo "[o48-flow] regional_suite_steps=${RESOLVED_REGIONAL_SUITE_STEPS:-none}"
echo "[o48-flow] regional_suite_regions=${REGIONAL_SUITE_REGION_NAMES}"
echo "[o48-flow] storm_plot_regions=${STORM_PLOT_REGIONS}"
echo "[o48-flow] spectra_method=${RESOLVED_SPECTRA_METHOD}"
if [[ "${RESOLVED_SPECTRA_METHOD}" == "ecmwf" ]]; then
  echo "[o48-flow] ecmwf_spectra_weather_states=${RESOLVED_ECMWF_SPECTRA_WEATHER_STATES}"
fi
echo "[o48-flow] run_tc_pdf=${RUN_TC_PDF}"

SBATCH_ARGS=()
if [[ "${HOLD}" == "1" ]]; then
  SBATCH_ARGS+=(--hold)
fi

if [[ "${NO_SUBMIT}" == "1" ]]; then
  echo "[o48-flow] render-only mode enabled; not submitting"
  exit 0
fi

if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  build_submit="$(sbatch "${SBATCH_ARGS[@]}" "${BUILD_SCRIPT}")"
  build_job="$(extract_job_id "${build_submit}")"
  echo "[o48-flow] ${build_submit}"
  predict_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${build_job} "${PREDICT_SCRIPT}")"
else
  build_submit=""
  build_job=""
  predict_submit="$(sbatch "${SBATCH_ARGS[@]}" "${PREDICT_SCRIPT}")"
fi
predict_job="$(extract_job_id "${predict_submit}")"
echo "[o48-flow] ${predict_submit}"

loss_submit="$(sbatch "${SBATCH_ARGS[@]}" "${LOSS_SCRIPT}")"
loss_job="$(extract_job_id "${loss_submit}")"
echo "[o48-flow] ${loss_submit}"

sigma_submit="$(sbatch "${SBATCH_ARGS[@]}" "${SIGMA_SCRIPT}")"
sigma_job="$(extract_job_id "${sigma_submit}")"
echo "[o48-flow] ${sigma_submit}"

local_job="skipped"
if [[ "${RUN_ONE_DATE_LOCAL}" == "1" ]]; then
  local_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${predict_job} "${LOCAL_SCRIPT}")"
  local_job="$(extract_job_id "${local_submit}")"
  echo "[o48-flow] ${local_submit}"
fi

regional_jobs=()
for regional_script in "${REGIONAL_SCRIPTS[@]}"; do
  regional_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${predict_job} "${regional_script}")"
  regional_job="$(extract_job_id "${regional_submit}")"
  regional_jobs+=("${regional_job}")
  echo "[o48-flow] ${regional_submit}"
done

storm_jobs=()
for storm_script in "${STORM_SCRIPTS[@]}"; do
  storm_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${predict_job} "${storm_script}")"
  storm_job="$(extract_job_id "${storm_submit}")"
  storm_jobs+=("${storm_job}")
  echo "[o48-flow] ${storm_submit}"
done

spectra_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${predict_job} "${SPECTRA_SCRIPT}")"
spectra_job="$(extract_job_id "${spectra_submit}")"
echo "[o48-flow] ${spectra_submit}"

tc_job="skipped"
if [[ "${RUN_TC_PDF}" == "1" ]]; then
  tc_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${predict_job} "${TC_SCRIPT}")"
  tc_job="$(extract_job_id "${tc_submit}")"
  echo "[o48-flow] ${tc_submit}"
fi

cat <<EOF

=== O48 -> O96 MANUAL EVAL FLOW SUBMITTED ===
run_id:             ${RUN_ID}
run_root:           ${RUN_ROOT}
checkpoint_path:    ${RESOLVED_CKPT_PATH}
phase:              ${PHASE}
source_grib_root:   ${SOURCE_GRIB_ROOT}
bundle_dir:         ${BUNDLE_DIR}
training_run_filter:${RESOLVED_TRAINING_RUN_FILTER}
sigmas:             ${SIGMAS}
regional_steps:     ${RESOLVED_REGIONAL_SUITE_STEPS:-none}
regional_regions:   ${REGIONAL_SUITE_REGION_NAMES}
storm_regions:      ${STORM_PLOT_REGIONS}
spectra_method:     ${RESOLVED_SPECTRA_METHOD}
run_tc_pdf:         ${RUN_TC_PDF}
job_ids:
  build_bundles:    ${build_job:-skipped_prebuilt_bundle_root}
  predict:          ${predict_job}
  training_losses:  ${loss_job}
  sigma:            ${sigma_job}
  local_plots:      ${local_job}
  spectra:          ${spectra_job}
  tc_pdf:           ${tc_job}
  regional_suites:  ${regional_jobs[*]:-skipped}
  storm_plots:      ${storm_jobs[*]:-skipped}
EOF
