#!/bin/bash
# mt2_e2_20k read B (2026-10-20): the TD-227 temporal read (eval.tools.temporal_consistency.read, downscaling-tools
# 26c6f89) per run and date on the eval.cli files, merged by the TD-227 kit's merge_temporal_parts.py into pairs:
#   mt2 stride 1 over stride 2; stride 1 and stride 2 over do0 20k (independent noise); stride 1 over the TD-228 shared-
#   noise and independent-noise arms (do0 50k EMA-10000, parts already on disk, linked, not recomputed);
# and temporal_consistency_metrics.py (tools/multitime) on the npz trajectories, stride 1 with stride 2 as control.
# Bands 31-56, 56-125, >125 km; ratio of the mean |f(L+12 h) - f(L)| to the truth's. Status lines in reads/status_b.txt.
set -o pipefail
ulimit -c 0
: "${MT_W:?}"
W=$MT_W; DT=$W/downscaling-tools; O=$W/reads; P=$O/parts
TD=/home/ecm5702/scratch/agent-work/20261016-temporal-td227
DATES="20230226 20230822 20230829 20230912 20230919"
source /home/ecm5702/scratch/sandbox/20261008-patch-decoder-r8-e2/activate.sh || { echo "FATAL: activation refused" >&2; exit 3; }
export PYTHONPATH=$DT PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp MPLCONFIGDIR=/tmp/$USER/mpl OMP_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$P"; cd "$DT"
ST=$O/status_b.txt
echo "[scoreB] start commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)" | tee -a "$ST"
for d in $DATES; do
  [[ -e $P/td228sh_$d ]] || ln -s "$TD/reads2/parts/shared_$d" "$P/td228sh_$d"
  [[ -e $P/td228ind_$d ]] || ln -s "$TD/reads2/parts/independent_$d" "$P/td228ind_$d"
done
declare -A DIRS=([mt2s1]=$W/runs/mt2_s1/predictions [mt2s2]=$W/runs/mt2_s2/predictions [do0r20]=$W/runs/do0r20_ind/predictions)
pids=()
for arm in mt2s1 mt2s2 do0r20; do
  for d in $DATES; do
    n=$(ls "${DIRS[$arm]}"/predictions_${d}_step0*.nc 2>/dev/null | wc -l)
    if [[ $n -ne 6 ]]; then echo "temporal $arm $d SKIPPED: $n of 6 files" | tee -a "$ST"; continue; fi
    ( python -m eval.tools.temporal_consistency.read --run $arm=${DIRS[$arm]} --dates $d --output-dir $P/${arm}_$d \
        --box e2 --storm-box 10,45,-80,-30 > $P/log_${arm}_$d.txt 2>&1
      echo "temporal $arm $d exit=$? $(date -u +%FT%TZ)" | tee -a "$ST" ) &
    pids+=($!)
  done
done
VPY=/home/ecm5702/hpcperm/sandbox/20260917-vera-spatial-downscaler/.venv-x86_64/bin/python
MTT=/home/ecm5702/scratch/sandbox/20261016-multitime-e2/code/downscaling-tools/tools/multitime
( s1=$(ls $W/runs/mt2_s1/npz/*/mt2_*_m??.npz 2>/dev/null); s2=$(ls $W/runs/mt2_s2/npz/*/mt2_*_m??.npz 2>/dev/null)
  echo "npz files: stride1 $(echo $s1 | wc -w) stride2 $(echo $s2 | wc -w)" | tee -a "$ST"
  $VPY $MTT/temporal_consistency_metrics.py $s1 --control $s2 --out $O/tcm_mt2s1_vs_mt2s2 > $O/log_tcm.txt 2>&1
  echo "tcm exit=$? $(date -u +%FT%TZ)" | tee -a "$ST" ) &
tcm=$!
for p in "${pids[@]}"; do wait $p; done
merge() {  # out A B  (read.md gives B over A)
  python $W/tools/merge_temporal_parts.py $P $O/$1 $2 $3 > $O/log_merge_$1.txt 2>&1
  echo "merge $1 exit=$? $(date -u +%FT%TZ)" | tee -a "$ST"
}
merge temporal_mt2s1_over_mt2s2 mt2s2 mt2s1
merge temporal_mt2s2_over_do0r20 do0r20 mt2s2
merge temporal_mt2s1_over_do0r20 do0r20 mt2s1
merge temporal_mt2s1_over_td228sh td228sh mt2s1
merge temporal_mt2s1_over_td228ind td228ind mt2s1
merge temporal_do0r20_over_td228ind td228ind do0r20
wait $tcm
echo "[scoreB] done $(date -u +%FT%TZ)" | tee -a "$ST"
exit 0
