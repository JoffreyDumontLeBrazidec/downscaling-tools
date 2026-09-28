#!/bin/bash
#SBATCH --job-name=sh2a_analysis
#SBATCH --qos=nf
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/ecm5702/agent-work/20260909-station-head-adapter/logs/%x_%j.out
set -uo pipefail
module load python3 vtb ecmwf-toolbox
python3 -u /home/ecm5702/dev/downscaling-tools-station-head/tools/station_head/stage2a/analysis_at_stations_full.py
echo "ANALYSIS_RC=$?"
