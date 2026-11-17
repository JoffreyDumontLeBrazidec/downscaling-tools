#!/bin/bash
# Extremes set XSET-1: one model row on the given start dates (6 leads x 10 members each, about 35 min per date on 4 GH200).
# Refuses to start without the bit-identity gate. The predictor skips files that exist, so a requeue continues.
# Usage: sbatch --job-name=xset_<row>_<i> xset_predict.sbatch <ROW> <yyyymmdd> [<yyyymmdd> ...]
#SBATCH --job-name=xset_p
#SBATCH --account=gkpdm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set/logs/%x-%j.out
set -o pipefail; ulimit -c 0
X=/e/data1/jureap-data/ecmwf/users/jdumont/outputs/xset1_20261116
K=${XSET_JWORK:-/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set}/downscaling-tools/kits/20261116-extremes-set/jupiter
[[ -f /e/data1/jureap-data/ecmwf/users/jdumont/outputs/xset1_20261116/test/GL512_237k/XSET_TEST_PASS ]] || { echo "[xset] FATAL no XSET_TEST_PASS"; exit 2; }
row=${1:?row}; shift
cases=""; for d in "$@"; do for L in 12 24 36 48 60 72; do cases="$cases${cases:+,}$d:$L"; done; done
bash $K/xset_run.sh "$row" predict "$cases" 29591 || { echo "[xset] PREDICT_FAIL $row $(date -u +%FT%TZ)"; exit 1; }
echo "[xset] PREDICT_DONE $row $* $(date -u +%FT%TZ)"
