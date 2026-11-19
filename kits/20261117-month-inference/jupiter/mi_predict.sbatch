#!/bin/bash
# MI-S25: one model row on the given start dates (20 leads x 10 members = 200 draws per date, about 2 h per date on 4 GH200).
# Refuses to start without the xcheck gate. The predictor skips files that exist, so a requeue continues.
# Usage: sbatch --job-name=mi_<row>_<i> mi_predict.sbatch <ROW> <yyyymmdd> [<yyyymmdd> ...]   (LEADS env overrides the lead list)
#SBATCH --job-name=mi_p
#SBATCH --account=gkpdm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261117-month-inference/logs/%x-%j.out
set -o pipefail; ulimit -c 0
M=/e/data1/jureap-data/ecmwf/users/jdumont/outputs/month_inference_s25   # the xcheck gate lives in the S25 folder for every set
K=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261117-month-inference/downscaling-tools/kits/20261117-month-inference/jupiter
source $K/../mi_set.sh
[[ -f $M/xcheck/GL512_500k/MI_XCHECK_PASS ]] || { echo "[mi] FATAL no MI_XCHECK_PASS"; exit 2; }
row=${1:?row}; shift
export MI_SET=$(mi_set_of ${1:?date}) || exit 2
for d in "$@"; do [[ $(mi_set_of $d) == $MI_SET ]] || { echo "[mi] FATAL mixed sets in one job: $*"; exit 2; }; done
LEADS=${LEADS:-"12 24 36 48 60 72 84 96 108 120 132 144 156 168 180 192 204 216 228 240"}
cases=""; for d in "$@"; do for L in $LEADS; do cases="$cases${cases:+,}$d:$L"; done; done
bash $K/mi_run.sh "$row" predict "$cases" 29593 || { echo "[mi] PREDICT_FAIL $row $(date -u +%FT%TZ)"; exit 1; }
echo "[mi] PREDICT_DONE $row $* $(date -u +%FT%TZ)"
