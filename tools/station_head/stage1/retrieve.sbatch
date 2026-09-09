#!/bin/bash
#SBATCH --job-name=stnhead-retrieve
#SBATCH --qos=nf
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ecm5702/agent-work/20260909-station-head-adapter/logs/%x_%j.out
set -uo pipefail
R=/home/ecm5702/agent-work/20260909-station-head-adapter
module load python3 vtb ecmwf-toolbox
export METVIEW_PYTHON_START_TIMEOUT=120
cd $R
python3 -u scripts/retrieve_stations.py
echo "RETRIEVE_RC=$?"
