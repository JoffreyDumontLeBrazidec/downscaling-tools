#!/bin/bash --login
# ECMWF spectra template — AC-only (requires gptosp.ser).
#
# Computes trusted spectral energy spectra from prediction NetCDF files using
# the ECMWF spectral-transform route (gptosp), via eval.cli evaluate.
#
# Pipeline (handled by eval.evaluators.spectra_ecmwf.runner):
#   1. Stage prediction fields as nopoles GRIB files (from predictions_*.nc)
#   2. Convert GRIBs to spectral harmonics via gptosp.ser
#   3. Compute spectra amplitudes from the staged spectral harmonics
#
# All lane-level settings (template_root, weather_states, steps, members,
# reference_dir) are read from eval/config/lanes/<lane>.yaml.
#
# Resource baselines (empirical):
#   30 transforms  (6 params × 5 dates × 1 member):  ~2-3h
#   300 transforms (6 params × 5 dates × 10 members): ~24h
#   Memory: 128G sufficient
#
# Resumability: gptosp stage skips existing *_sh files on resubmission.
#
# IMPORTANT: This template is AC-only. gptosp.ser and the required modules
# (ifs, eclib, pifsenv) are not available on AG nodes.

#SBATCH --job-name=spectra_ecmwf
#SBATCH --qos=nf
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
RUN_ROOT="${RUN_ROOT:-/home/ecm5702/scratch/eval/REPLACE_RUN_ID}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-${RUN_ROOT}/data/predictions}"
LANE="${LANE:-o96_o320}"
###############################################################################

REPO_ROOT="/home/ecm5702/dev/downscaling-tools"

[[ "${RUN_ROOT}" != *REPLACE_* ]] || { echo "ERROR: Set RUN_ROOT." >&2; exit 2; }
[[ -d "${PREDICTIONS_DIR}" ]]     || { echo "ERROR: PREDICTIONS_DIR not found: ${PREDICTIONS_DIR}" >&2; exit 2; }

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) ;;
  *) echo "ERROR: AC-only (requires gptosp.ser). Current host: ${HOST_SHORT}" >&2; exit 2 ;;
esac

source /home/ecm5702/dev/.ds-dyn/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "=== Spectra ECMWF (gptosp pipeline) ==="
echo "run_root=${RUN_ROOT}"
echo "predictions_dir=${PREDICTIONS_DIR}"
echo "lane=${LANE}"
echo "host=${HOST_SHORT}"
echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "${REPO_ROOT}"
python -m eval.cli evaluate \
    --predictions-dir "${PREDICTIONS_DIR}" \
    --lane "${LANE}" \
    --host atos_ac \
    --only spectra_ecmwf \
    --overwrite

echo "DONE Spectra ECMWF eval run_root=${RUN_ROOT}"
echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
