#!/bin/bash
# Login-node helper: render and submit the canonical `o320 -> o1280` manual-eval flow.
#
# This helper validates the checkpoint profile, then renders run-specific sbatch
# launchers under `/home/ecm5702/dev/jobscripts/submit/<YYYYMMDD>/` and submits:
#   1) truth-aware bundle rebuild into `${RUN_ROOT}/bundles_with_y`,
#   2) strict multi-bundle manual inference from that rebuilt bundle root,
#   3) one-date local plots,
#   4) spectra (auto: proxy on AG, ECMWF on AC),
#   5) TC evaluation (auto: native on AG, regridded on AC),
# with `afterok` dependencies across the chain.
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
SOURCE_GRIB_ROOT="${SOURCE_GRIB_ROOT:-/home/ecm5702/hpcperm/data/input_data/o320_o1280/${INPUT_EVENT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ecm5702/perm/eval}"
PHASE="${PHASE:-proxy}"                              # proxy | continue-full | full-only
PREBUILT_BUNDLE_ROOT="${PREBUILT_BUNDLE_ROOT:-}"    # optional bundles_with_y* root to skip rebuild

RUN_DATE_UTC="${RUN_DATE_UTC:-$(date -u +%Y%m%d)}"
RUN_SUFFIX="${RUN_SUFFIX:-manual_eval}"              # appended to the canonical run id
RUN_ID_OVERRIDE="${RUN_ID_OVERRIDE:-}"              # required for continue-full across days

PROXY_BUNDLE_PAIRS="${PROXY_BUNDLE_PAIRS:-20230829:24,20230828:48,20230829:48,20230828:24,20230830:24,20230828:72,20230827:72,20230830:48,20230829:72,20230827:48}"
SAMPLER_JSON="${SAMPLER_JSON:-{\"num_steps\":40,\"sigma_max\":1000.0,\"sigma_min\":0.03,\"rho\":7.0,\"sampler\":\"heun\",\"S_max\":1000.0}}"
DATE_PRESET="${DATE_PRESET:-default}"                # default | aug16_30
DATES="${DATES:-}"                                   # explicit CSV override
STEPS="${STEPS:-24,48,72,96,120}"
BUNDLE_PAIRS="${BUNDLE_PAIRS:-}"                     # optional exact date:step list
MEMBERS="${MEMBERS:-}"                               # blank => phase default

LOCAL_PLOT_DATE="${LOCAL_PLOT_DATE:-}"               # blank => auto first selected date
LOCAL_PLOT_EXPECTED_COUNT="${LOCAL_PLOT_EXPECTED_COUNT:-auto}"  # auto | integer >= 0
LOCAL_PLOT_OUT_SUBDIR="${LOCAL_PLOT_OUT_SUBDIR:-local_plots_one_date}"  # under RUN_ROOT/
RUN_REGIONAL_SUITES="${RUN_REGIONAL_SUITES:-1}"      # 1 => render regional step suites when steps exist
REGIONAL_SUITE_DATE="${REGIONAL_SUITE_DATE:-}"       # blank => prefer 20230829 when available
REGIONAL_SUITE_STEPS="${REGIONAL_SUITE_STEPS:-24,120}"
REGIONAL_SUITE_MODEL_VARIABLES="${REGIONAL_SUITE_MODEL_VARIABLES:-x_0,x_interp_0,y_0,y_pred_0,residuals_0,residuals_pred_0}"
REGIONAL_SUITE_OUT_PREFIX="${REGIONAL_SUITE_OUT_PREFIX:-local_plots_regions}"
RUN_STORM_PLOTS="${RUN_STORM_PLOTS:-1}"              # 1 => render storm boxes for selected steps
STORM_PLOT_REGIONS="${STORM_PLOT_REGIONS:-idalia,franklin}"
STORM_PLOT_OUT_PREFIX="${STORM_PLOT_OUT_PREFIX:-tc_local_plots}"
RUN_TC_CONTOUR_PLOTS="${RUN_TC_CONTOUR_PLOTS:-1}"   # 1 => render contour-style TC suites for selected steps
TC_CONTOUR_OUT_PREFIX="${TC_CONTOUR_OUT_PREFIX:-tc_contour_plots}"

SPECTRA_METHOD="${SPECTRA_METHOD:-auto}"             # auto | proxy | ecmwf
TC_SUPPORT_MODE="${TC_SUPPORT_MODE:-auto}"           # auto | native | regridded
TC_EVENTS="${TC_EVENTS:-idalia,franklin}"
TC_EXTRA_REFERENCE_EXPIDS="${TC_EXTRA_REFERENCE_EXPIDS:-}"

HOLD="${HOLD:-0}"                                    # 1 => submit held
NO_SUBMIT="${NO_SUBMIT:-0}"                          # 1 => render only
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"              # 1 => overwrite rendered files
SUBMIT_ROOT="${SUBMIT_ROOT:-/home/ecm5702/dev/jobscripts/submit}"
PROFILE_PYTHON="${PROFILE_PYTHON:-}"                 # optional override for checkpoint_profile
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
    "${root}/eefo_o320_0001_date${date}_time0000_mem1to10_step24to120_sfc.grib"
    "${root}/eefo_o320_0001_date${date}_time0000_mem1to10_step24to120_pl.grib"
    "${root}/enfo_o1280_0001_date${date}_time0000_step24to120_sfc.grib"
    "${root}/enfo_o1280_0001_date${date}_time0000_mem1to10_step24to120_sfc_y.grib"
    "${root}/enfo_o1280_0001_date${date}_time0000_mem1to10_step24to120_pl_y.grib"
  )
  local path
  for path in "${required[@]}"; do
    [[ -f "${path}" ]] || die "Missing required source GRIB: ${path}"
  done
}

require_choice "${SOURCE_HPC}" ac ag leonardo jupiter
require_choice "${PHASE}" proxy continue-full full-only
require_choice "${DATE_PRESET}" default aug16_30
require_choice "${SPECTRA_METHOD}" auto proxy ecmwf
require_choice "${TC_SUPPORT_MODE}" auto native regridded
require_bool "${HOLD}"
require_bool "${NO_SUBMIT}"
require_bool "${ALLOW_OVERWRITE}"
require_bool "${RUN_REGIONAL_SUITES}"
require_bool "${RUN_STORM_PLOTS}"
require_bool "${RUN_TC_CONTOUR_PLOTS}"
[[ "${CHECKPOINT_PATH}" != REPLACE_* ]] || die "Set CHECKPOINT_PATH."
[[ -d "${SOURCE_GRIB_ROOT}" ]] || die "SOURCE_GRIB_ROOT does not exist: ${SOURCE_GRIB_ROOT}"
[[ "${RUN_SUFFIX}" =~ ^[A-Za-z0-9._-]*$ ]] || die "Unsafe RUN_SUFFIX: ${RUN_SUFFIX}"
[[ "${RUN_ID_OVERRIDE}" =~ ^[A-Za-z0-9._-]*$ ]] || die "Unsafe RUN_ID_OVERRIDE: ${RUN_ID_OVERRIDE}"
[[ "${PREBUILT_BUNDLE_ROOT}" =~ ^[A-Za-z0-9._/:-]*$ ]] || die "Unsafe PREBUILT_BUNDLE_ROOT: ${PREBUILT_BUNDLE_ROOT}"
if [[ "${PHASE}" == "continue-full" && -z "${RUN_ID_OVERRIDE}" ]]; then
  die "PHASE=continue-full requires RUN_ID_OVERRIDE so the full run reuses the proxy run id."
fi
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

if [[ -n "${MEMBERS}" ]]; then
  EFFECTIVE_MEMBERS="${MEMBERS}"
elif [[ "${PHASE}" == "proxy" ]]; then
  EFFECTIVE_MEMBERS="1"
else
  EFFECTIVE_MEMBERS="1,2,3,4,5,6,7,8,9,10"
fi

if [[ "${PHASE}" == "proxy" && -z "${BUNDLE_PAIRS}" ]]; then
  EFFECTIVE_BUNDLE_PAIRS="${PROXY_BUNDLE_PAIRS}"
else
  EFFECTIVE_BUNDLE_PAIRS="${BUNDLE_PAIRS}"
fi

mapfile -t SCOPE_FIELDS < <(
  python3 - "${DATE_PRESET}" "${DATES}" "${STEPS}" "${EFFECTIVE_BUNDLE_PAIRS}" "${LOCAL_PLOT_DATE}" "${LOCAL_PLOT_EXPECTED_COUNT}" <<'PY'
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

mapfile -t REGIONAL_FIELDS < <(
  python3 - "${RESOLVED_DATES}" "${RESOLVED_STEPS}" "${REGIONAL_SUITE_DATE}" "${REGIONAL_SUITE_STEPS}" "${RESOLVED_LOCAL_PLOT_DATE}" <<'PY'
import sys

resolved_dates = [token.strip() for token in sys.argv[1].split(",") if token.strip()]
resolved_steps = [str(int(token.strip())) for token in sys.argv[2].split(",") if token.strip()]
regional_suite_date = sys.argv[3].strip()
regional_suite_steps = [str(int(token.strip())) for token in sys.argv[4].split(",") if token.strip()]
local_plot_date = sys.argv[5].strip()

if not resolved_dates:
    raise SystemExit("Resolved date list is empty.")
if not resolved_steps:
    raise SystemExit("Resolved step list is empty.")

preferred_date = regional_suite_date or ("20230829" if "20230829" in resolved_dates else local_plot_date or resolved_dates[0])
if preferred_date not in resolved_dates:
    raise SystemExit(f"REGIONAL_SUITE_DATE={preferred_date} is not present in the resolved date list.")

selected_steps = [step for step in regional_suite_steps if step in resolved_steps]

print(preferred_date)
print(",".join(selected_steps))
PY
)

RESOLVED_REGIONAL_SUITE_DATE="${REGIONAL_FIELDS[0]}"
RESOLVED_REGIONAL_SUITE_STEPS="${REGIONAL_FIELDS[1]}"

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
    bundles_with_y*)
      ;;
    *)
      die "PREBUILT_BUNDLE_ROOT basename must look like bundles_with_y*: ${PREBUILT_BUNDLE_ROOT}"
      ;;
  esac
  BUNDLE_DIR="${PREBUILT_BUNDLE_ROOT}"
  USE_PREBUILT_BUNDLES=1
else
  BUNDLE_DIR="${RUN_ROOT}/bundles_with_y"
  USE_PREBUILT_BUNDLES=0
fi
PREDICTIONS_DIR="${RUN_ROOT}/predictions"

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

if [[ "${PHASE}" == "continue-full" && ! -d "${RUN_ROOT}" ]]; then
  die "PHASE=continue-full requires an existing run root: ${RUN_ROOT}"
fi

TEMPLATE_DIR="${PROJECT_ROOT}/eval/jobs/templates"
BUILD_TEMPLATE="${TEMPLATE_DIR}/build_o320_o1280_truth_bundles.sbatch"
INFER_TEMPLATE="${TEMPLATE_DIR}/strict_manual_predict_x_bundle.sbatch"
LOCAL_TEMPLATE="${TEMPLATE_DIR}/local_plots_one_date_from_predictions.sbatch"
REGIONAL_TEMPLATE="${TEMPLATE_DIR}/regional_suite_from_predictions.sbatch"
PROXY_SPECTRA_TEMPLATE="${TEMPLATE_DIR}/spectra_proxy_from_predictions.sbatch"
ECMWF_SPECTRA_TEMPLATE="${TEMPLATE_DIR}/spectra_ecmwf_from_predictions.sbatch"
TC_TEMPLATE="${TEMPLATE_DIR}/tc_eval_from_predictions.sbatch"
TC_CONTOUR_TEMPLATE="${TEMPLATE_DIR}/tc_contour_suite_from_predictions.sbatch"

[[ -f "${INFER_TEMPLATE}" ]] || die "Missing template: ${INFER_TEMPLATE}"
[[ -f "${LOCAL_TEMPLATE}" ]] || die "Missing template: ${LOCAL_TEMPLATE}"
[[ -f "${REGIONAL_TEMPLATE}" ]] || die "Missing template: ${REGIONAL_TEMPLATE}"
[[ -f "${PROXY_SPECTRA_TEMPLATE}" ]] || die "Missing template: ${PROXY_SPECTRA_TEMPLATE}"
[[ -f "${ECMWF_SPECTRA_TEMPLATE}" ]] || die "Missing template: ${ECMWF_SPECTRA_TEMPLATE}"
[[ -f "${TC_TEMPLATE}" ]] || die "Missing template: ${TC_TEMPLATE}"
[[ -f "${TC_CONTOUR_TEMPLATE}" ]] || die "Missing template: ${TC_CONTOUR_TEMPLATE}"
if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  [[ -f "${BUILD_TEMPLATE}" ]] || die "Missing template: ${BUILD_TEMPLATE}"
fi

SUBMIT_DIR="${SUBMIT_ROOT}/${RUN_DATE_UTC}"
mkdir -p "${SUBMIT_DIR}"

BUILD_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_build_truth_bundles.sbatch"
PREDICT_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_predict.sbatch"
LOCAL_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_local_plots.sbatch"
SPECTRA_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_spectra.sbatch"
TC_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_tc_eval.sbatch"
REGIONAL_SCRIPTS=()
STORM_SCRIPTS=()
TC_CONTOUR_SCRIPTS=()

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

if [[ "${RUN_TC_CONTOUR_PLOTS}" == "1" && -n "${RESOLVED_REGIONAL_SUITE_STEPS}" ]]; then
  while IFS= read -r step; do
    [[ -n "${step}" ]] || continue
    step_padded="$(printf '%03d' "${step}")"
    TC_CONTOUR_SCRIPTS+=("${SUBMIT_DIR}/${RUN_ID}_tc_contours_step${step_padded}.sbatch")
  done < <(python3 - "${RESOLVED_REGIONAL_SUITE_STEPS}" <<'PY'
import sys
for token in sys.argv[1].split(","):
    token = token.strip()
    if token:
        print(token)
PY
)
fi

TARGET_SCRIPTS=("${PREDICT_SCRIPT}" "${LOCAL_SCRIPT}" "${SPECTRA_SCRIPT}" "${TC_SCRIPT}")
if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  TARGET_SCRIPTS=("${BUILD_SCRIPT}" "${TARGET_SCRIPTS[@]}")
fi
TARGET_SCRIPTS+=("${REGIONAL_SCRIPTS[@]}" "${STORM_SCRIPTS[@]}" "${TC_CONTOUR_SCRIPTS[@]}")
for target in "${TARGET_SCRIPTS[@]}"; do
  if [[ -e "${target}" && "${ALLOW_OVERWRITE}" -ne 1 ]]; then
    die "Refusing to overwrite existing generated file: ${target} (set ALLOW_OVERWRITE=1 to replace)"
  fi
done

if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  cp "${BUILD_TEMPLATE}" "${BUILD_SCRIPT}"
fi
cp "${INFER_TEMPLATE}" "${PREDICT_SCRIPT}"
cp "${LOCAL_TEMPLATE}" "${LOCAL_SCRIPT}"
if [[ "${RESOLVED_SPECTRA_METHOD}" == "proxy" ]]; then
  cp "${PROXY_SPECTRA_TEMPLATE}" "${SPECTRA_SCRIPT}"
else
  cp "${ECMWF_SPECTRA_TEMPLATE}" "${SPECTRA_SCRIPT}"
fi
cp "${TC_TEMPLATE}" "${TC_SCRIPT}"
for regional_script in "${REGIONAL_SCRIPTS[@]}"; do
  cp "${REGIONAL_TEMPLATE}" "${regional_script}"
done
for storm_script in "${STORM_SCRIPTS[@]}"; do
  cp "${REGIONAL_TEMPLATE}" "${storm_script}"
done
for tc_contour_script in "${TC_CONTOUR_SCRIPTS[@]}"; do
  cp "${TC_CONTOUR_TEMPLATE}" "${tc_contour_script}"
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
  set_sbatch_directive "${BUILD_SCRIPT}" job-name "o1280_bundle_${CHECKPOINT_SHORT}"
fi

set_var "${PREDICT_SCRIPT}" STACK_FLAVOR "${STACK_FLAVOR}"
set_var "${PREDICT_SCRIPT}" LANE "${LANE}"
set_var "${PREDICT_SCRIPT}" SOURCE_HPC "${SOURCE_HPC}"
set_var "${PREDICT_SCRIPT}" CHECKPOINT_REF "${RESOLVED_CKPT_PATH}"
set_var "${PREDICT_SCRIPT}" CHECKPOINT_SHORT "${CHECKPOINT_SHORT}"
set_var "${PREDICT_SCRIPT}" INPUT_ROOT "${BUNDLE_DIR}"
set_var "${PREDICT_SCRIPT}" DATE_PRESET "${DATE_PRESET}"
set_var "${PREDICT_SCRIPT}" DATES "${RESOLVED_DATES}"
set_var "${PREDICT_SCRIPT}" STEPS "${RESOLVED_STEPS}"
set_var "${PREDICT_SCRIPT}" BUNDLE_PAIRS "${RESOLVED_BUNDLE_PAIRS}"
set_var "${PREDICT_SCRIPT}" MEMBERS "${EFFECTIVE_MEMBERS}"
set_var "${PREDICT_SCRIPT}" OUTPUT_ROOT "${OUTPUT_ROOT}"
set_var "${PREDICT_SCRIPT}" RUN_DATE_UTC "${RUN_DATE_UTC}"
set_var "${PREDICT_SCRIPT}" RUN_SUFFIX "${RUN_SUFFIX}"
set_var "${PREDICT_SCRIPT}" RUN_ID_OVERRIDE "${RUN_ID}"
set_var "${PREDICT_SCRIPT}" NUM_GPUS_PER_MODEL "4"
set_var "${PREDICT_SCRIPT}" ALLOW_EXISTING_RUN_DIR "1"
set_var "${PREDICT_SCRIPT}" ALLOW_REBUILT_BUNDLE_ROOT "1"
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
set_var "${LOCAL_SCRIPT}" MODEL_VARIABLES "${REGIONAL_SUITE_MODEL_VARIABLES}"
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
  set_var "${SPECTRA_SCRIPT}" WEATHER_STATES "10u,10v,2t,sp,t_850,z_500"
  set_var "${SPECTRA_SCRIPT}" TEMPLATE_ROOT "/home/ecm5702/hpcperm/reference_spectra/enfo_o1280"
fi
set_sbatch_directive "${SPECTRA_SCRIPT}" job-name "o1280_spectra_${CHECKPOINT_SHORT}"

for regional_script in "${REGIONAL_SCRIPTS[@]}"; do
  step_padded="$(basename "${regional_script}" | sed -n 's/.*_step\([0-9][0-9][0-9]\)\.sbatch/\1/p')"
  [[ -n "${step_padded}" ]] || die "Could not parse step from regional script: ${regional_script}"
  set_var "${regional_script}" RUN_ROOT "${RUN_ROOT}"
  set_var "${regional_script}" RUN_ID "${RUN_ID}"
  set_var "${regional_script}" PREDICTIONS_FILE "${PREDICTIONS_DIR}/predictions_${RESOLVED_REGIONAL_SUITE_DATE}_step${step_padded}.nc"
  set_var "${regional_script}" OUT_DIR "${RUN_ROOT}/${REGIONAL_SUITE_OUT_PREFIX}_step${step_padded}"
  set_var "${regional_script}" REGION_NAMES ""
  set_var "${regional_script}" SUITE_KIND "regions"
  set_var "${regional_script}" MODEL_VARIABLES "${REGIONAL_SUITE_MODEL_VARIABLES}"
  set_sbatch_directive "${regional_script}" job-name "o1280_regions${step_padded}_${CHECKPOINT_SHORT}"
done

for storm_script in "${STORM_SCRIPTS[@]}"; do
  step_padded="$(basename "${storm_script}" | sed -n 's/.*_step\([0-9][0-9][0-9]\)\.sbatch/\1/p')"
  [[ -n "${step_padded}" ]] || die "Could not parse step from storm script: ${storm_script}"
  set_var "${storm_script}" RUN_ROOT "${RUN_ROOT}"
  set_var "${storm_script}" RUN_ID "${RUN_ID}"
  set_var "${storm_script}" PREDICTIONS_FILE "${PREDICTIONS_DIR}/predictions_${RESOLVED_REGIONAL_SUITE_DATE}_step${step_padded}.nc"
  set_var "${storm_script}" OUT_DIR "${RUN_ROOT}/${STORM_PLOT_OUT_PREFIX}_step${step_padded}"
  set_var "${storm_script}" REGION_NAMES "${STORM_PLOT_REGIONS}"
  set_var "${storm_script}" SUITE_KIND "storm"
  set_var "${storm_script}" MODEL_VARIABLES "${REGIONAL_SUITE_MODEL_VARIABLES}"
  set_sbatch_directive "${storm_script}" job-name "o1280_storms${step_padded}_${CHECKPOINT_SHORT}"
done

for tc_contour_script in "${TC_CONTOUR_SCRIPTS[@]}"; do
  step_padded="$(basename "${tc_contour_script}" | sed -n 's/.*_step\([0-9][0-9][0-9]\)\.sbatch/\1/p')"
  [[ -n "${step_padded}" ]] || die "Could not parse step from TC contour script: ${tc_contour_script}"
  set_var "${tc_contour_script}" RUN_ROOT "${RUN_ROOT}"
  set_var "${tc_contour_script}" RUN_ID "${RUN_ID}"
  set_var "${tc_contour_script}" PREDICTIONS_FILE "${PREDICTIONS_DIR}/predictions_${RESOLVED_REGIONAL_SUITE_DATE}_step${step_padded}.nc"
  set_var "${tc_contour_script}" OUT_DIR "${RUN_ROOT}/${TC_CONTOUR_OUT_PREFIX}_step${step_padded}"
  set_var "${tc_contour_script}" REGION_NAMES "${STORM_PLOT_REGIONS}"
  set_sbatch_directive "${tc_contour_script}" job-name "o1280_tcmap${step_padded}_${CHECKPOINT_SHORT}"
done

if [[ "${HOST_FAMILY}" == "ac" ]]; then
  CPU_QOS="nf"
  if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
    drop_sbatch_directive "${BUILD_SCRIPT}" gpus-per-node
  fi
  drop_sbatch_directive "${LOCAL_SCRIPT}" gpus-per-node
  drop_sbatch_directive "${TC_SCRIPT}" gpus-per-node
  for regional_script in "${REGIONAL_SCRIPTS[@]}"; do
    drop_sbatch_directive "${regional_script}" gpus-per-node
  done
  for storm_script in "${STORM_SCRIPTS[@]}"; do
    drop_sbatch_directive "${storm_script}" gpus-per-node
  done
  for tc_contour_script in "${TC_CONTOUR_SCRIPTS[@]}"; do
    drop_sbatch_directive "${tc_contour_script}" gpus-per-node
  done
  if [[ "${RESOLVED_SPECTRA_METHOD}" == "proxy" ]]; then
    drop_sbatch_directive "${SPECTRA_SCRIPT}" gpus-per-node
  fi
else
  CPU_QOS="ng"
  if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
    set_sbatch_directive "${BUILD_SCRIPT}" gpus-per-node "0"
  fi
  set_sbatch_directive "${LOCAL_SCRIPT}" gpus-per-node "0"
  set_sbatch_directive "${TC_SCRIPT}" gpus-per-node "0"
  for regional_script in "${REGIONAL_SCRIPTS[@]}"; do
    set_sbatch_directive "${regional_script}" gpus-per-node "0"
  done
  for storm_script in "${STORM_SCRIPTS[@]}"; do
    set_sbatch_directive "${storm_script}" gpus-per-node "0"
  done
  for tc_contour_script in "${TC_CONTOUR_SCRIPTS[@]}"; do
    set_sbatch_directive "${tc_contour_script}" gpus-per-node "0"
  done
  if [[ "${RESOLVED_SPECTRA_METHOD}" == "proxy" ]]; then
    set_sbatch_directive "${SPECTRA_SCRIPT}" gpus-per-node "0"
  fi
fi
if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  set_sbatch_directive "${BUILD_SCRIPT}" qos "${CPU_QOS}"
fi
set_sbatch_directive "${LOCAL_SCRIPT}" qos "${CPU_QOS}"
set_sbatch_directive "${TC_SCRIPT}" qos "${CPU_QOS}"
for regional_script in "${REGIONAL_SCRIPTS[@]}"; do
  set_sbatch_directive "${regional_script}" qos "${CPU_QOS}"
done
for storm_script in "${STORM_SCRIPTS[@]}"; do
  set_sbatch_directive "${storm_script}" qos "${CPU_QOS}"
done
for tc_contour_script in "${TC_CONTOUR_SCRIPTS[@]}"; do
  set_sbatch_directive "${tc_contour_script}" qos "${CPU_QOS}"
done
if [[ "${RESOLVED_SPECTRA_METHOD}" == "proxy" ]]; then
  set_sbatch_directive "${SPECTRA_SCRIPT}" qos "${CPU_QOS}"
fi

if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  bash -n "${BUILD_SCRIPT}" "${PREDICT_SCRIPT}" "${LOCAL_SCRIPT}" "${SPECTRA_SCRIPT}" "${TC_SCRIPT}" "${REGIONAL_SCRIPTS[@]}" "${STORM_SCRIPTS[@]}" "${TC_CONTOUR_SCRIPTS[@]}"
else
  bash -n "${PREDICT_SCRIPT}" "${LOCAL_SCRIPT}" "${SPECTRA_SCRIPT}" "${TC_SCRIPT}" "${REGIONAL_SCRIPTS[@]}" "${STORM_SCRIPTS[@]}" "${TC_CONTOUR_SCRIPTS[@]}"
fi

echo "[o1280-flow] checkpoint=${RESOLVED_CKPT_PATH}"
echo "[o1280-flow] stack=${STACK_FLAVOR} lane=${LANE}"
echo "[o1280-flow] host=${HOST_SHORT} host_family=${HOST_FAMILY}"
echo "[o1280-flow] recommended_venv=${RECOMMENDED_VENV}"
echo "[o1280-flow] phase=${PHASE}"
echo "[o1280-flow] source_grib_root=${SOURCE_GRIB_ROOT}"
echo "[o1280-flow] bundle_dir=${BUNDLE_DIR}"
echo "[o1280-flow] use_prebuilt_bundles=${USE_PREBUILT_BUNDLES}"
echo "[o1280-flow] run_id=${RUN_ID}"
echo "[o1280-flow] run_root=${RUN_ROOT}"
echo "[o1280-flow] dates=${RESOLVED_DATES}"
echo "[o1280-flow] steps=${RESOLVED_STEPS}"
echo "[o1280-flow] bundle_pairs=${RESOLVED_BUNDLE_PAIRS:-none}"
echo "[o1280-flow] members=${EFFECTIVE_MEMBERS}"
echo "[o1280-flow] local_plot_date=${RESOLVED_LOCAL_PLOT_DATE}"
echo "[o1280-flow] local_plot_expected_count=${RESOLVED_LOCAL_PLOT_EXPECTED_COUNT}"
echo "[o1280-flow] local_plot_out_dir=${RUN_ROOT}/${LOCAL_PLOT_OUT_SUBDIR}"
echo "[o1280-flow] run_regional_suites=${RUN_REGIONAL_SUITES}"
echo "[o1280-flow] regional_suite_date=${RESOLVED_REGIONAL_SUITE_DATE}"
echo "[o1280-flow] regional_suite_steps=${RESOLVED_REGIONAL_SUITE_STEPS:-none}"
echo "[o1280-flow] regional_suite_model_variables=${REGIONAL_SUITE_MODEL_VARIABLES}"
echo "[o1280-flow] run_storm_plots=${RUN_STORM_PLOTS}"
echo "[o1280-flow] storm_plot_regions=${STORM_PLOT_REGIONS}"
echo "[o1280-flow] run_tc_contour_plots=${RUN_TC_CONTOUR_PLOTS}"
echo "[o1280-flow] tc_contour_out_prefix=${TC_CONTOUR_OUT_PREFIX}"
echo "[o1280-flow] spectra_method=${RESOLVED_SPECTRA_METHOD}"
echo "[o1280-flow] tc_support_mode=${RESOLVED_TC_SUPPORT_MODE}"
echo "[o1280-flow] generated_scripts:"
if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  echo "  - ${BUILD_SCRIPT}"
fi
echo "  - ${PREDICT_SCRIPT}"
echo "  - ${LOCAL_SCRIPT}"
echo "  - ${SPECTRA_SCRIPT}"
echo "  - ${TC_SCRIPT}"
for regional_script in "${REGIONAL_SCRIPTS[@]}"; do
  echo "  - ${regional_script}"
done
for storm_script in "${STORM_SCRIPTS[@]}"; do
  echo "  - ${storm_script}"
done
for tc_contour_script in "${TC_CONTOUR_SCRIPTS[@]}"; do
  echo "  - ${tc_contour_script}"
done

SBATCH_ARGS=()
if [[ "${HOLD}" == "1" ]]; then
  SBATCH_ARGS+=(--hold)
fi

if [[ "${NO_SUBMIT}" == "1" ]]; then
  echo "[o1280-flow] render-only mode enabled; not submitting"
  exit 0
fi

if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  build_submit="$(sbatch "${SBATCH_ARGS[@]}" "${BUILD_SCRIPT}")"
  build_job="$(extract_job_id "${build_submit}")"
  echo "[o1280-flow] ${build_submit}"
  predict_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${build_job} "${PREDICT_SCRIPT}")"
else
  build_submit=""
  build_job=""
  predict_submit="$(sbatch "${SBATCH_ARGS[@]}" "${PREDICT_SCRIPT}")"
fi
predict_job="$(extract_job_id "${predict_submit}")"
echo "[o1280-flow] ${predict_submit}"

local_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${predict_job} "${LOCAL_SCRIPT}")"
local_job="$(extract_job_id "${local_submit}")"
echo "[o1280-flow] ${local_submit}"

regional_jobs=()
for regional_script in "${REGIONAL_SCRIPTS[@]}"; do
  regional_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${predict_job} "${regional_script}")"
  regional_job="$(extract_job_id "${regional_submit}")"
  regional_jobs+=("${regional_job}")
  echo "[o1280-flow] ${regional_submit}"
done

spectra_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${predict_job} "${SPECTRA_SCRIPT}")"
spectra_job="$(extract_job_id "${spectra_submit}")"
echo "[o1280-flow] ${spectra_submit}"

tc_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${predict_job} "${TC_SCRIPT}")"
tc_job="$(extract_job_id "${tc_submit}")"
echo "[o1280-flow] ${tc_submit}"

storm_jobs=()
for storm_script in "${STORM_SCRIPTS[@]}"; do
  storm_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${predict_job} "${storm_script}")"
  storm_job="$(extract_job_id "${storm_submit}")"
  storm_jobs+=("${storm_job}")
  echo "[o1280-flow] ${storm_submit}"
done

tc_contour_jobs=()
for tc_contour_script in "${TC_CONTOUR_SCRIPTS[@]}"; do
  tc_contour_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${predict_job} "${tc_contour_script}")"
  tc_contour_job="$(extract_job_id "${tc_contour_submit}")"
  tc_contour_jobs+=("${tc_contour_job}")
  echo "[o1280-flow] ${tc_contour_submit}"
done

regional_job_summary="${regional_jobs[*]:-skipped}"
storm_job_summary="${storm_jobs[*]:-skipped}"
tc_contour_job_summary="${tc_contour_jobs[*]:-skipped}"
monitor_jobs=()
if [[ -n "${build_job}" ]]; then
  monitor_jobs+=("${build_job}")
fi
monitor_jobs+=("${predict_job}" "${local_job}" "${spectra_job}" "${tc_job}")
monitor_jobs+=("${regional_jobs[@]}")
monitor_jobs+=("${storm_jobs[@]}")
monitor_jobs+=("${tc_contour_jobs[@]}")
monitor_job_csv="$(IFS=,; echo "${monitor_jobs[*]}")"
GENERATED_SCRIPT_LINES=""
if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then
  GENERATED_SCRIPT_LINES+="  - ${BUILD_SCRIPT}"$'\n'
fi
for script_path in "${PREDICT_SCRIPT}" "${LOCAL_SCRIPT}" "${SPECTRA_SCRIPT}" "${TC_SCRIPT}" "${REGIONAL_SCRIPTS[@]}" "${STORM_SCRIPTS[@]}" "${TC_CONTOUR_SCRIPTS[@]}"; do
  GENERATED_SCRIPT_LINES+="  - ${script_path}"$'\n'
done

cat <<EOF

=== O320 -> O1280 MANUAL EVAL FLOW SUBMITTED ===
run_id:             ${RUN_ID}
run_root:           ${RUN_ROOT}
checkpoint_path:    ${RESOLVED_CKPT_PATH}
phase:              ${PHASE}
submit_hold:        ${HOLD}
source_grib_root:   ${SOURCE_GRIB_ROOT}
bundle_dir:         ${BUNDLE_DIR}
local_plots_dir:    ${RUN_ROOT}/${LOCAL_PLOT_OUT_SUBDIR}
regional_suite_date:${RESOLVED_REGIONAL_SUITE_DATE}
regional_steps:     ${RESOLVED_REGIONAL_SUITE_STEPS:-none}
spectra_method:     ${RESOLVED_SPECTRA_METHOD}
tc_support_mode:    ${RESOLVED_TC_SUPPORT_MODE}
generated_scripts:
${GENERATED_SCRIPT_LINES}job_ids:
  build_bundles:    ${build_job:-skipped_prebuilt_bundle_root}
  predict:          ${predict_job} $(if [[ "${USE_PREBUILT_BUNDLES}" -eq 0 ]]; then printf '(afterok:%s)' "${build_job}"; else printf '(no build dependency)'; fi)
  local_plots:      ${local_job} (afterok:${predict_job})
  spectra:          ${spectra_job} (afterok:${predict_job})
  tc_eval:          ${tc_job} (afterok:${predict_job})
  regional_suites:  ${regional_job_summary} (afterok:${predict_job})
  storm_plots:      ${storm_job_summary} (afterok:${predict_job})
  tc_contours:      ${tc_contour_job_summary} (afterok:${predict_job})

Monitor:
  squeue -j ${monitor_job_csv}
EOF
