#!/bin/bash
# mt2_e2_20k read A (2026-10-20): eval.cli judge in the rule-18 order (zoom_maps Alps and Franklin, shape, tc_structure,
# per-bin box_spectra full field and residual, extremes, probabilistic) on the five 2023 test dates, 10 members:
#   +24 h  mt2 stride 2 (isolated windows = the model's own per-frame output) against do0_e2_50k_ag 20k EMA-2000
#   +24 h  mt2 stride 1 (averaged windows) against mt2 stride 2 (the no-smearing guard of read B)
#   +72 h  mt2 stride 2 against do0 20k (the last lead)
# downscaling-tools at 26c6f89 (worktree in the work folder), the patch decoder r8 sandbox venv (the TD-227 read's).
# A run whose 30 files are not all there is not judged; status lines in reads/status_a.txt.
set -o pipefail
ulimit -c 0
: "${MT_W:?}"
W=$MT_W; DT=$W/downscaling-tools; O=$W/reads
S1=$W/runs/mt2_s1/predictions; S2=$W/runs/mt2_s2/predictions; R=$W/runs/do0r20_ind/predictions
source /home/ecm5702/scratch/sandbox/20261008-patch-decoder-r8-e2/activate.sh || { echo "FATAL: activation refused" >&2; exit 3; }
export PYTHONPATH=$DT PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp MPLCONFIGDIR=/tmp/$USER/mpl \
       DOCS_ROOT=/home/ecm5702/dev/docs OMP_NUM_THREADS=4
mkdir -p "$MPLCONFIGDIR" "$O"; cd "$DT"
ST=$O/status_a.txt
DATES=20230226,20230822,20230829,20230912,20230919
echo "[scoreA] start commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)" | tee -a "$ST"
ok() { local n; n=$(ls "$1"/predictions_2023*_step0*.nc 2>/dev/null | wc -l); echo "files $1: $n" | tee -a "$ST"; [[ $n -eq 30 ]]; }
judge() {  # tag runlabel=dir reflabel=dir step weights
  ( python -m eval.cli judge --lane o320_o1280 --run "$2" --reference "$3" --output-dir "$O/$1" --step "$4" \
      --dates $DATES --weights "$5" > "$O/log_$1.txt" 2>&1
    echo "judge $1 exit=$? $(date -u +%FT%TZ)" | tee -a "$ST" ) &
}
ok "$S2" && ok "$R" && judge judgeA_step24_mt2s2_vs_do0r20 mt2s2=$S2 do0r20=$R 24 "EMA-2000 / EMA-2000"
ok "$S1" && ok "$S2" && judge judge_step24_mt2s1_vs_mt2s2 mt2s1=$S1 mt2s2=$S2 24 "EMA-2000 / EMA-2000"
ok "$S2" && ok "$R" && judge judgeA_step72_mt2s2_vs_do0r20 mt2s2=$S2 do0r20=$R 72 "EMA-2000 / EMA-2000"
wait
echo "[scoreA] done $(date -u +%FT%TZ)" | tee -a "$ST"
exit 0
