#!/bin/bash
# MI-S25 gate before any month row: the month path (full-field packed output) must reproduce the XSET-1 GL512 500k row bit for
# bit at its evaluation points (2 cases of 20240804 x 10 members, seed base 1234) and round-trip the packing within its bound.
# Writes outputs/month_inference_s25/xcheck/<ROW>/MI_XCHECK_PASS. Optional argument: another row with an XSET-1 row of the same name (e.g. FS242_300k).
#SBATCH --job-name=mi_xcheck
#SBATCH --account=gkpdm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261117-month-inference/logs/%x-%j.out
set -o pipefail; ulimit -c 0
K=${MI_K:-/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261117-month-inference/downscaling-tools/kits/20261117-month-inference/jupiter}   # MI_K: another checkout (sbatch spools this file; give --output too)
bash $K/mi_run.sh "${1:-GL512_500k}" xcheck "${XCASES:-20240804:24,20240804:72}" 29643 || { echo "[mi] MI_XCHECK_FAIL $(date -u +%FT%TZ)"; exit 1; }
echo "[mi] MI_XCHECK_JOB_DONE $(date -u +%FT%TZ)"
