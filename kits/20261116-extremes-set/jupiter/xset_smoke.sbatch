#!/bin/bash
# Extremes set XSET-1: smoke of a NEW MODEL before its first row (about 15 min on 4 GH200, 2 draws): the trainer builds
# from the model file, the weights load strictly, the anchor case proves the input paths, the five outputs are found by
# name, one case is drawn with 2 members and compared with the 9 km control at the evaluation points (bounds fatal).
# Writes XOUT/smoke/MODEL_<MODEL>_PASS and XOUT/smoke/<ROW>/XSET_SMOKE_PASS. The row used can be any export of the model.
# Usage: sbatch --job-name=xsmoke_<MODEL> xset_smoke.sbatch <ROW> [yyyymmdd:lead]
#SBATCH --job-name=xsmoke
#SBATCH --account=gkpdm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set/logs/%x-%j.out
set -o pipefail; ulimit -c 0
X=${XSET_JOUT:-/e/data1/jureap-data/ecmwf/users/jdumont/outputs/xset1_20261116}
K=${XSET_JWORK:-/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set}/downscaling-tools/kits/20261116-extremes-set/jupiter
[[ -f $X/test/GL512_237k/XSET_TEST_PASS ]] || { echo "[xset] FATAL no XSET_TEST_PASS"; exit 2; }
row=${1:?row}; case=${2:-20240926:48}
model=$(grep -m1 '^MODEL=' $X/rowdefs/$row.env | cut -d= -f2); [[ -n "$model" ]] || { echo "[xset] FATAL row $row is not defined"; exit 3; }
f=$X/smoke/$row/predictions_${case%%:*}_step$(printf %03d ${case##*:}).nc
[[ ! -e $f ]] || mv "$f" "$f.prev.$SLURM_JOB_ID"      # the predictor skips files that exist: a smoke always draws
bash $K/xset_run.sh "$row" smoke "$case" 29591 || { echo "[xset] SMOKE_FAIL $model row $row $(date -u +%FT%TZ)"; exit 1; }
[[ -s $f ]] || { echo "[xset] SMOKE_FAIL $model: no file $f"; exit 1; }
# kit commit without git (compute nodes have none outside the launcher's environment): read it from the checkout
G=$(cd $K && while [[ ! -e .git && $PWD != / ]]; do cd ..; done; pwd)/.git
h=$(cat $G/HEAD 2>/dev/null); [[ $h == ref:* ]] && h=$(cat $G/${h#ref: } 2>/dev/null || grep -m1 " ${h#ref: }\$" $G/packed-refs 2>/dev/null | cut -d' ' -f1)
echo "row=$row case=$case job=$SLURM_JOB_ID kit=${h:0:7} utc=$(date -u +%FT%TZ)" > $X/smoke/MODEL_${model}_PASS
cp $X/smoke/MODEL_${model}_PASS $X/smoke/$row/XSET_SMOKE_PASS
echo "[xset] SMOKE_PASS $model row $row case $case $(date -u +%FT%TZ)"
