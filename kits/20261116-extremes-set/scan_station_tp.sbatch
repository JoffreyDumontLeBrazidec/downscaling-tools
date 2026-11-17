#!/bin/bash
#SBATCH --job-name=xset-scan
#SBATCH --qos=nf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set/logs/scan_%j.out
# Station tp scan of one period (Atos, reads only). Submit as
#   sbatch --export=ALL,START=20240101,END=20240331 scan_station_tp.sbatch
# START / END are YYYYMMDD. TOOLS = downscaling-tools checkout holding this kit.
set -eu
WORK=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set
TOOLS=${TOOLS:-$WORK/downscaling-tools}
module load python3 vtb ecmwf-toolbox
export TMPDIR=$WORK/tmp/$SLURM_JOB_ID
mkdir -p "$TMPDIR" "$WORK/scan"
date -u
PYTHONPATH=$TOOLS python3 "$TOOLS/kits/20261116-extremes-set/scan_station_tp.py" --start "$START" --end "$END" --out "$WORK/scan"
date -u
