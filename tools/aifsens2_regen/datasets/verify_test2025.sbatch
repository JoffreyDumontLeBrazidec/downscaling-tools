#!/bin/bash
#SBATCH --job-name=aifscrps-test2025-verify
#SBATCH --qos=nf
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/datasets-build/runs/test2025/logs/%x_%j.out
set -uo pipefail
module load ecmwf-toolbox
source /home/ecm5702/hpcperm/venvs/pristine-uv-x86_64/bin/activate
export PYTHONPATH=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/anemoi-datasets/src
python /home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/datasets-build/verify_full2025.py \
  --dir /home/ecm5702/scratch/data/aifscrps_2026_full_20260910/test2025 --suffix "-test4" --fixture
echo "TEST2025_VERIFY_RC=$?"
