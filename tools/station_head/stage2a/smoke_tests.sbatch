#!/bin/bash
#SBATCH --job-name=sh2a_smoke
#SBATCH --qos=nf
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/ecm5702/agent-work/20260909-station-head-adapter/logs/%x_%j.out
# The smoke tests of section 8 of the design note, on the proved case. Torch comes
# from the sandbox virtual environment; everything else from the module stack.
set -uo pipefail
S=/home/ecm5702/dev/downscaling-tools-station-head/tools/station_head/stage2a
PRED=${PRED:-/home/ecm5702/scratch/eval/station_head_2a/proof_case/predictions_20260512_step006.nc}
CASE=${CASE:-2026051212_step006}
module load ecmwf-toolbox
source /home/ecm5702/hpcperm/sandbox/20260902-hres-local-branch/activate.sh || exit 3
env -u PYTHONPATH python -u "$S/smoke_tests.py" --case "$CASE" --prediction "$PRED"
echo "SMOKE_RC=$?"
