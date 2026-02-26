#!/bin/bash
#SBATCH --job-name=tc_plot_j24v
#SBATCH --qos=nf
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ecm5702/perm/eval/j24v/logs/tc_plot_j24v_%j.out

set -euo pipefail
mkdir -p /home/ecm5702/perm/eval/j24v/logs /home/ecm5702/perm/eval/j24v/tc
source /home/ecm5702/dev/.ds-dyn/bin/activate
export PYTHONPATH="/home/ecm5702/dev/downscaling-tools:${PYTHONPATH:-}"

python -m eval.tc.plot_pdf_tc \
  --expver j24v \
  --outdir /home/ecm5702/perm/eval/j24v/tc \
  --out-name tc_normed_pdfs_all_events_j24v.pdf
