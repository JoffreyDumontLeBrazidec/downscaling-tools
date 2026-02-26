#!/bin/bash
#SBATCH --job-name=quaver_j24v
#SBATCH --qos=nf
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/ecm5702/perm/eval/j24v/logs/quaver_j24v_%j.out

set -euo pipefail
mkdir -p /home/ecm5702/perm/eval/j24v/logs
module load quaver

PARAMS="--expver j24v \
--nmem 2 \
--first_reference_date 20230826 \
--last_reference_date 20230827 \
--date_step 24 \
--first_lead_time 24 \
--last_lead_time 120 \
--lead_time_step 24 \
--grid O320 \
--class rd \
--database fdb"

quaver /home/ecm5702/dev/downscaling-tools/eval/quaver/q_compute_probabilistic.py ${PARAMS}
