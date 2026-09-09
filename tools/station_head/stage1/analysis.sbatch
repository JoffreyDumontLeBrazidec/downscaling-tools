#!/bin/bash
#SBATCH --job-name=stnhead-analysis
#SBATCH --qos=nf
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ecm5702/agent-work/20260909-station-head-adapter/logs/%x_%j.out
set -uo pipefail
R=/home/ecm5702/agent-work/20260909-station-head-adapter
module load python3 vtb ecmwf-toolbox
cd $R
python3 -u scripts/analysis_at_stations.py
echo "ANALYSIS_RC=$?"
