#!/bin/bash
# Forcing-date check (17 Nov 2026): the defect arm GL512_450k_FTINIT (time forcings at the start date) on the given XSET-1
# start dates, 6 leads x 10 members, same seeds as GL512_450k. Runs the kit of the worktree $XK/dt-ftcheck (branch
# claude/project-thread-5uvbba), so the live XSET checkout is untouched. About 35 min per date on 4 GH200.
# Usage: [ROW=GL512_450k_FT240 LEADS="24 48 72"] sbatch --export=ALL --job-name=ftchk_<i> ft_predict.sbatch <yyyymmdd> [<yyyymmdd> ...]
#SBATCH --job-name=ftchk
#SBATCH --account=gkpdm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set/logs/%x-%j.out
set -o pipefail; ulimit -c 0
X=/e/data1/jureap-data/ecmwf/users/jdumont/outputs/xset1_20261116
export XKIT_OVERRIDE=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set/dt-ftcheck/kits/20261116-extremes-set
K=$XKIT_OVERRIDE/jupiter
[[ -f $X/test/GL512_237k/XSET_TEST_PASS ]] || { echo "[ftchk] FATAL no XSET_TEST_PASS"; exit 2; }
grep -q 'FORCING_TIME.startswith' $XKIT_OVERRIDE/xset_predict_global.py || { echo "[ftchk] FATAL worktree kit lacks FORCING_TIME"; exit 3; }
ROW=${ROW:-GL512_450k_FTINIT}; LEADS=${LEADS:-12 24 36 48 60 72}
cases=""; for d in "$@"; do for L in $LEADS; do cases="$cases${cases:+,}$d:$L"; done; done
bash $K/xset_run.sh $ROW predict "$cases" 29593 || { echo "[ftchk] PREDICT_FAIL $(date -u +%FT%TZ)"; exit 1; }
echo "[ftchk] PREDICT_DONE $* $(date -u +%FT%TZ)"
