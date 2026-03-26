#!/bin/bash --login
# Canonical ECMWF spectra template — AC-only (requires gptosp.ser).
#
# Computes trusted spectral energy spectra from prediction NetCDF files using
# the ECMWF spectral-transform route (gptosp), following the same pipeline as:
#   <repo>/eval/spectra/grb_to_spectra.sh
#
# Pipeline:
#   1. Stage prediction fields as nopoles GRIB files (from predictions_*.nc)
#   2. Convert GRIBs to spectral harmonics via gptosp.ser
#   3. Compute spectra amplitudes via compute_spectra-3.py
#
# For the AG-side healpy proxy alternative, use:
#   /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/spectra_proxy_from_predictions.sbatch
#
# IMPORTANT: This template is AC-only. gptosp.ser and the required modules
# (ifs, eclib, pifsenv) are not available on AG nodes.
#
# Resource baselines (empirical):
#   300 transforms (6 params x 50 files): ~24h
#   Full pipeline with compute: 48:00:00 recommended
#   Memory: 128G sufficient
#
# Resumability: The template skips already-completed gptosp transforms
# (checks for existing *_sh files) so it can be resubmitted after walltime.

#SBATCH --job-name=spectra_ecmwf
#SBATCH --qos=nf
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/perm/eval/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
RUN_ROOT="/home/ecm5702/perm/eval/REPLACE_RUN_ID"
RUN_ID="REPLACE_RUN_ID"
PREDICTIONS_DIR="${RUN_ROOT}/predictions"

# Date/step/member filters (CSV or ALL)
DATE_LIST="20230826,20230827,20230828,20230829,20230830"
STEP_LIST="120"
MEMBER_LIST="ALL"                # CSV of member indices (1-based) or ALL

# Parameters to compute spectra for
PARAMS="2t:sfc 10u:sfc 10v:sfc sp:sfc t:850 z:500"

# Output layout — mirrors grb_to_spectra.sh convention
SPECTRA_ROOT="${RUN_ROOT}/spectra_ecmwf"
GRB_DIR="${SPECTRA_ROOT}/grb"
SH_DIR="${SPECTRA_ROOT}/spectral_harmonics"
SPECTRA_DIR="${SPECTRA_ROOT}/spectra"

# Conda env for step 3 (compute_spectra-3.py)
CONDA_ENV="/home/ecm5702/dev/.ds-dyn/bin/activate"

# Compute spectra script
COMPUTE_SPECTRA_PY="/home/ecm5702/dev/post_prepml/spectra/spectra_ml/individual_files/compute_spectra-3.py"

WRITE_EXPERIMENT_CONFIG=0
###############################################################################

PROJECT_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

csv_matches() {
  local csv="$1"
  local needle="$2"
  if [[ "${csv}" == "ALL" ]]; then return 0; fi
  [[ ",${csv}," == *",${needle},"* ]]
}

# --- Cluster check ---
HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) ;;
  *) die "This template is AC-only (requires gptosp.ser). Current host: ${HOST_SHORT}" ;;
esac

[[ "${RUN_ROOT}" != *REPLACE_* ]] || die "Set RUN_ROOT."
[[ "${RUN_ID}" != REPLACE_* ]] || die "Set RUN_ID."
[[ -d "${PREDICTIONS_DIR}" ]] || die "Predictions directory does not exist: ${PREDICTIONS_DIR}"

mkdir -p "${GRB_DIR}" "${SH_DIR}" "${SPECTRA_DIR}" "${RUN_ROOT}/logs"

echo "=== STEP 1: Stage prediction fields as nopoles GRIBs ==="

# Use a short TMPDIR symlink to avoid gptosp path-length issues
WORK_LINK="${TMPDIR:-/tmp}/spectra_${SLURM_JOB_ID:-$$}"
ln -sfn "${SPECTRA_ROOT}" "${WORK_LINK}"
SHORT_GRB="${WORK_LINK}/grb"
SHORT_SH="${WORK_LINK}/spectral_harmonics"

source "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
PYTHON="$(which python)"
echo "[INFO] Python: ${PYTHON} ($(${PYTHON} -V 2>&1))"

# Stage predictions → GRIB (one file per param/date/step/member)
# This extracts fields from predictions_*.nc and writes them as GRIB on the
# prediction's native grid with polar rows trimmed (area=89/0/-89/360).
STAGED=0
SKIPPED=0

for pred_file in "${PREDICTIONS_DIR}"/predictions_*.nc; do
  [[ -f "${pred_file}" ]] || continue
  base="$(basename "${pred_file}")"

  # Extract date and step from filename
  if [[ "${base}" =~ ^predictions_([0-9]{8})_step([0-9]{2,3})\.nc$ ]]; then
    file_date="${BASH_REMATCH[1]}"
    file_step="${BASH_REMATCH[2]}"
  else
    echo "[WARN] Skipping unrecognized filename: ${base}"
    continue
  fi

  csv_matches "${DATE_LIST}" "${file_date}" || continue
  csv_matches "${STEP_LIST}" "${file_step}" || continue

  # Stage each param from this prediction file
  for param_level in ${PARAMS}; do
    param="${param_level%%:*}"
    level="${param_level##*:}"
    param_dir="${param}_${level}"
    mkdir -p "${GRB_DIR}/${param_dir}"

    # Determine member count from the file (do once per file)
    if [[ -z "${_MEMBER_COUNT:-}" ]]; then
      _MEMBER_COUNT=$("${PYTHON}" -c "
import xarray as xr
ds = xr.open_dataset('${pred_file}')
if 'member' in ds.dims:
    print(ds.dims['member'])
elif 'number' in ds.dims:
    print(ds.dims['number'])
else:
    print(1)
" 2>/dev/null || echo "10")
    fi

    for mem_idx in $(seq 1 "${_MEMBER_COUNT}"); do
      csv_matches "${MEMBER_LIST}" "${mem_idx}" || continue
      grb_out="${GRB_DIR}/${param_dir}/${RUN_ID}_${file_date}_${file_step}_${mem_idx}_nopoles.grb"

      if [[ -f "${grb_out}" ]] && [[ -s "${grb_out}" ]]; then
        SKIPPED=$((SKIPPED + 1))
        continue
      fi

      "${PYTHON}" -c "
import xarray as xr, numpy as np, earthkit.data

ds = xr.open_dataset('${pred_file}')
# Resolve variable name (may be param or y_pred_param)
var_candidates = ['y_pred_${param}', '${param}']
var_name = None
for v in var_candidates:
    if v in ds:
        var_name = v
        break
if var_name is None:
    raise KeyError('No variable matching ${param} in ${pred_file}')

arr = ds[var_name]
# Select member
mem_dim = 'member' if 'member' in arr.dims else 'number' if 'number' in arr.dims else None
if mem_dim:
    arr = arr.isel({mem_dim: ${mem_idx} - 1})

# Flatten to 1D values
vals = arr.values.flatten()
lats = ds['latitude'].values if 'latitude' in ds else ds['lat'].values
lons = ds['longitude'].values if 'longitude' in ds else ds['lon'].values

# Trim polar rows (area=89/0/-89/360)
mask = np.abs(lats) <= 89.0
vals = vals[mask]
lats = lats[mask]
lons = lons[mask]

print(f'Staged ${param}_${level} date=${file_date} step=${file_step} mem=${mem_idx}: {len(vals)} points')
# Write as GRIB using earthkit
import eccodes
with open('${grb_out}', 'wb') as f:
    msgid = eccodes.codes_grib_new_from_samples('reduced_gg_ml_grib2')
    eccodes.codes_set(msgid, 'shortName', '${param}')
    eccodes.codes_set(msgid, 'values', vals)
    eccodes.codes_write(msgid, f)
    eccodes.codes_release(msgid)
" 2>&1 && STAGED=$((STAGED + 1)) || echo "[WARN] Failed to stage ${param} from ${base} mem=${mem_idx}"

    done
  done
done

echo "[INFO] Staged: ${STAGED}, Skipped (existing): ${SKIPPED}"

echo ""
echo "=== STEP 2: Convert GRIBs to spectral harmonics via gptosp.ser ==="

module unload ecmwf-toolbox 2>/dev/null || true
module load eclib 2>/dev/null || true
module load pifsenv 2>/dev/null || true
module load ifs 2>/dev/null || true
export DR_HOOK_ASSERT_MPI_INITIALIZED=0

TRANSFORMS_DONE=0
TRANSFORMS_SKIPPED=0

for param_level in ${PARAMS}; do
  param="${param_level%%:*}"
  level="${param_level##*:}"
  param_dir="${param}_${level}"
  mkdir -p "${SH_DIR}/${param_dir}"

  for grb_file in "${GRB_DIR}/${param_dir}"/*.grb; do
    [[ -f "${grb_file}" ]] || continue
    grb_base="$(basename "${grb_file}")"
    sh_out="${SH_DIR}/${param_dir}/${grb_base}_sh"

    if [[ -f "${sh_out}" ]] && [[ -s "${sh_out}" ]]; then
      TRANSFORMS_SKIPPED=$((TRANSFORMS_SKIPPED + 1))
      continue
    fi

    # Use short paths to avoid gptosp filename truncation
    short_in="${SHORT_GRB}/${param_dir}/${grb_base}"
    short_out="${SHORT_SH}/${param_dir}/${grb_base}_sh"

    echo "[gptosp] ${grb_base} -> ${grb_base}_sh"
    gptosp.ser -l -g "${short_in}" -S "${short_out}" && \
      TRANSFORMS_DONE=$((TRANSFORMS_DONE + 1)) || \
      echo "[WARN] gptosp failed for ${grb_base}"
  done
done

echo "[INFO] gptosp transforms done: ${TRANSFORMS_DONE}, skipped (existing): ${TRANSFORMS_SKIPPED}"

echo ""
echo "=== STEP 3: Compute spectra amplitudes ==="

module unload ifs 2>/dev/null || true
module load ecmwf-toolbox 2>/dev/null || true

source "${CONDA_ENV}"

# The canonical compute script expects data at $PERM/ai_spectra/<expid>/spectral_harmonics.
# Use $PERM when available, otherwise default to the canonical perm root.
PERM_ROOT="${PERM:-/home/ecm5702/perm}"
SPECTRA_COMPAT="${PERM_ROOT}/ai_spectra/${RUN_ID}"
mkdir -p "${SPECTRA_COMPAT}"
ln -sfn "${SH_DIR}" "${SPECTRA_COMPAT}/spectral_harmonics"
ln -sfn "${SPECTRA_DIR}" "${SPECTRA_COMPAT}/spectra"
ln -sfn "${GRB_DIR}" "${SPECTRA_COMPAT}/grb"

"${PYTHON}" "${COMPUTE_SPECTRA_PY}" --expver "${RUN_ID}"

echo "[INFO] Spectra computation complete"

# Clean up symlinks
rm -f "${WORK_LINK}"

if [[ "${WRITE_EXPERIMENT_CONFIG}" -eq 1 ]]; then
  cat > "${SPECTRA_ROOT}/EXPERIMENT_CONFIG.yaml" <<EOF
run_id: ${RUN_ID}
purpose: "ECMWF spectral-transform spectra from predictions"
pipeline:
  canonical_reference: ${PROJECT_ROOT}/eval/spectra/grb_to_spectra.sh
  stage_1: "predictions_*.nc -> nopoles GRIB (polar trimmed)"
  stage_2: "gptosp.ser (gridpoint -> spectral harmonics)"
  stage_3: "compute_spectra-3.py (spectral harmonics -> spectra amplitudes)"
predictions:
  source_dir: ${PREDICTIONS_DIR}
  date_list: ${DATE_LIST}
  step_list: ${STEP_LIST}
  member_list: ${MEMBER_LIST}
params: ${PARAMS}
output:
  grb_dir: ${GRB_DIR}
  sh_dir: ${SH_DIR}
  spectra_dir: ${SPECTRA_DIR}
EOF
fi

echo "DONE: spectra in ${SPECTRA_DIR}"
