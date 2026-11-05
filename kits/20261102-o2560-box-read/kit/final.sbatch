#!/bin/bash
# o2560 Europe-box read (2 Nov 2026): after all scorings. (1) zoom maps at +24 h (Alps and the deepest-low date) and at +72 h;
# (2) summary tables per lead per variable for A100, B24, B50 and IN (summarize.py); (3) eval.cli judge (rule 32 v2,
# downscaling-tools main 4e659c3) A100 vs B24 and A100 vs B50 at every lead, lane o1280_o2560. The judge has never run on
# o2560 rows: a failure is recorded with its traceback, not patched. Log ends O2560_FINAL_DONE.
#SBATCH --job-name=o25_final
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/logs/%x-%j.out
set -eo pipefail
ulimit -c 0
source /e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/kit/eval_env.sh
set +e
LEADS=${*:-12 24 36 48 60 72}   # leads as arguments (sbatch --export splits at commas)
for lead in 24 72; do
  [[ " $LEADS " == *" $lead "* ]] || { echo "[final] zoom maps +$lead h skipped: not among the read's leads ($LEADS)"; continue; }
  L=$(printf %03d $lead)
  [[ -f $OUT/maps/L$L/cases.json ]] || python $KIT/scripts/zoom_maps_o2560.py $lead $OUT/maps/L$L A100=$OUT/pred/A100_L$L B24=$OUT/pred/B24_L$L B50=$OUT/pred/B50_L$L > $OUT/maps_L$L.log 2>&1
  echo "[final] zoom maps L$L rc=$? $(grep -h CASES $OUT/maps_L$L.log | head -1) $(date -u +%FT%TZ)"
done
python $KIT/scripts/summarize.py $OUT "$LEADS" > $OUT/summary.log 2>&1; echo "[final] summarize rc=$? $(tail -1 $OUT/summary.log)"
mkdir -p $OUT/judge
for lead in $LEADS; do
  L=$(printf %03d $lead)
  for pair in "A100 B24" "A100 B50"; do
    set -- $pair
    o=$OUT/judge/${1}_vs_${2}_L$L
    cmd="python -m eval.cli judge --lane o1280_o2560 --host jupiter --run $1=$OUT/pred/${1}_L$L --reference $2=$OUT/pred/${2}_L$L --output-dir $o --step $lead --weights 'EMA-10000 / EMA-10000'"
    echo "[final] CMD $cmd"; t0=$(date +%s)
    ( cd $DST && python -m eval.cli judge --lane o1280_o2560 --host jupiter --run $1=$OUT/pred/${1}_L$L --reference $2=$OUT/pred/${2}_L$L \
        --output-dir $o --step $lead --weights "EMA-10000 / EMA-10000" > $o.log 2>&1 )
    rc=$?; echo "[final] judge ${1}_vs_${2}_L$L rc=$rc wall=$(( $(date +%s) - t0 ))s $(date -u +%FT%TZ)"
    [[ $rc == 0 ]] || tail -25 $o.log
  done
done
echo "[final] O2560_FINAL_DONE $(date -u +%FT%TZ)"
