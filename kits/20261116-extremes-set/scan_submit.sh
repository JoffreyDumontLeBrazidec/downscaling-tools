#!/bin/bash
# Submit the station tp scan of 2024-2025, one job per quarter (8 jobs). Atos. Run: bash scan_submit.sh
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p /ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set/logs
for p in 20240101:20240331 20240401:20240630 20240701:20240930 20241001:20241231 \
         20250101:20250331 20250401:20250630 20250701:20250930 20251001:20251231; do
  sbatch --export=ALL,START=${p%%:*},END=${p##*:} "$HERE/scan_station_tp.sbatch"
done
