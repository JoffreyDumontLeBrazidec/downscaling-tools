#!/bin/bash
#SBATCH --job-name=eval_mars_j24v
#SBATCH --qos=nf
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ecm5702/perm/eval/j24v/logs/eval_mars_j24v_%j.out

set -euo pipefail
mkdir -p /home/ecm5702/perm/eval/j24v/logs
source /home/ecm5702/dev/.ds-dyn/bin/activate
export PYTHONPATH="/home/ecm5702/dev/downscaling-tools:${PYTHONPATH:-}"

python -m eval.run \
  --eval-root /home/ecm5702/perm/eval \
  mars-expver \
  --expver j24v \
  --run-name j24v \
  --date 20230801/20230802 \
  --number 1/2 \
  --step 24/120
