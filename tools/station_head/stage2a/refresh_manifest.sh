#!/bin/bash
# Refresh the stage 2a first-cut manifest. Run this whenever you want to know how
# far the inference has got; it only reads, and it takes a couple of minutes
# because it opens every prediction file to check the member count.
#
#   bash /home/ecm5702/dev/downscaling-tools-station-head/tools/station_head/stage2a/refresh_manifest.sh
#
# Add --no-open as an argument to get a faster answer that checks the file size
# only, which is enough while an array job is still running.
set -uo pipefail
module load python3 vtb ecmwf-toolbox >/dev/null 2>&1
python3 -u /home/ecm5702/dev/downscaling-tools-station-head/tools/station_head/stage2a/manifest.py "$@"
