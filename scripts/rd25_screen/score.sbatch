#!/bin/bash
#SBATCH --job-name=rd25_score
#SBATCH --qos=np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ecm5702/agent-work/20260915-rd25-evaluation/logs/%x_%j.out
set -euo pipefail
ARM="$1"
ROOT=/home/ecm5702/scratch/eval/o320_o1280/rd25_screen_$ARM
WORK=/home/ecm5702/agent-work/20260915-rd25-evaluation
PY=/home/ecm5702/dev/.ds-260612/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$WORK/tools"
"$PY" -m eval.cli evaluate --lane rd25_screen_20260915 --host atos_ac --predictions-dir "$ROOT/predictions" --output-dir "$ROOT/scores" --steps 24,120 --members 1,2,3 --dates 20250926 --only spectra_ecmwf_v2,surface,probabilistic,texture
"$PY" /home/ecm5702/agent-work/20260904-fine-scale-short-eval/scripts/band_ladder_pred_v2.py --pred-dir "$ROOT/predictions" --dates 20250926 --steps 24,120 --members 1,2,3 --vars 10u,10v,2t,msl --arm "$ARM" --out "$WORK/outputs/ladder_$ARM.csv"
echo "The six-sample screen completed. Three-member spread is a diagnostic, not a calibration verdict."
