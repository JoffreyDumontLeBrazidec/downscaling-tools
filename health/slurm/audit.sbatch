#!/bin/bash
# Network-health audit of several checkpoints, one job (review note s4 tier C). CPU checks by default; set
# HEALTH_GPU_ARGS="--gpu --bundles <b1.nc> <b2.nc>" for checks 2, 4, 7, 8, 9. Readings only; nothing is modified.
#   Jupiter: sbatch --account=<e-sta-destine|gkpdm> --partition=booster --gres=gpu:1 health/slurm/audit.sbatch CKPT...
#   Atos AG: sbatch --qos=ng --gpus=1 health/slurm/audit.sbatch CKPT...   (CPU only: --qos=nf, no --gpus)
# Environment: HEALTH_ENV = a script that puts python + numpy + yaml (+ torch and the model code for --gpu) on PATH.
#   Jupiter: module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3; source /e/data1/jureap-data/ecmwf/users/jdumont/venvs/.ds
#   AG:      ~/dev/.ds-ag-260616/bin/python3 (aarch64; never the x86 .ds-260612 there)
# Output: $HEALTH_ROOT/<ckpt id>/{audit.json,SUMMARY.txt} (default ~/perm/health); HEALTH_LANE sets the reference diff.
#SBATCH --job-name=health_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=health_audit_%j.out
set -uo pipefail
[[ -n "${HEALTH_ENV:-}" ]] && source "$HEALTH_ENV"
cd "${HEALTH_REPO:-$HOME/dev/downscaling-tools}"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
echo "job=${SLURM_JOB_ID:-local} host=$(hostname -s) $(date -u +%FT%TZ) git=$(git rev-parse --short HEAD)"
PY=${HEALTH_PYTHON:-python3}
for c in "$@"; do
  echo "== $c"
  [[ -f "$c" ]] || { echo "MISSING $c"; continue; }
  # CPU checks need nothing from PYTHONPATH; the GPU checks may (model code, datasets fork): keep it then
  if [[ -z "${HEALTH_GPU_ARGS:-}" ]]; then RUN=(env -u PYTHONPATH "$PY"); else RUN=("$PY"); fi
  "${RUN[@]}" -m health audit "$c" ${HEALTH_LANE:+--lane "$HEALTH_LANE"} ${HEALTH_GPU_ARGS:-} || echo "FAILED rc=$? $c"
done
echo "[health audit] done $(date -u +%FT%TZ)"
