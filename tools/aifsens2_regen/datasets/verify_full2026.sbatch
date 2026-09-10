#!/bin/bash
#SBATCH --job-name=aifscrps-full2026-verify
#SBATCH --qos=nf
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/datasets-build/runs/full2026/logs/%x_%j.out
set -uo pipefail
module load ecmwf-toolbox
source /home/ecm5702/hpcperm/venvs/pristine-uv-x86_64/bin/activate
export PYTHONPATH=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/anemoi-datasets/src
python /home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/datasets-build/verify_full2026.py; echo "FULL2026_VERIFY_RC=$?"
