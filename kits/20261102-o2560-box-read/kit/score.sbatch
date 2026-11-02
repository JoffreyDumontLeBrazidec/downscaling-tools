#!/bin/bash
# o2560 Europe-box read (2 Nov 2026): scoring of one lead, CPU on one booster node. Waits (afterany) on the three prediction
# jobs of that lead and checks its own inputs: 12 complete files per row. Steps, each skipped when its output exists:
#   1. box_spectra per row and region (eval.evaluators.box_spectra at 4e659c3: linear sampling at 0.035 degrees = o2560
#      spacing, whole box less a 1-degree margin and Europe 35-70 N 25 W-40 E, variables ws tp 10u 10v 2t msl; full field,
#      residual and input ratios per bin);
#   2. spectra_table.py per region (the three rows side by side, per bin);
#   3. paired_scores_o2560.py (fair CRPS, spread, RMSE per date with paired SE; corrected spread/RMSE pooled over dates
#      with date bootstrap; tp tail; rows and IN; ring area weights; edge strip excluded).
# Usage: sbatch --job-name=o25s_L<lead> score.sbatch <lead>. Log ends O2560_SCORE_DONE or O2560_SCORE_FAIL.
#SBATCH --job-name=o25s
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/logs/%x-%j.out
set -eo pipefail
ulimit -c 0
lead=${1:?lead}; L=$(printf %03d $((10#$lead)))
source /e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/kit/eval_env.sh
set +e
ROWS="A100 B24 B50"
fail() { echo "[o25s] O2560_SCORE_FAIL L$L: $1 $(date -u +%FT%TZ)"; exit ${2:-1}; }
echo "[o25s] L$L start, dstools $(git -C $DST rev-parse --short HEAD) $(date -u +%FT%TZ)"
for r in $ROWS; do
  n=$(ls $OUT/pred/${r}_L$L/predictions_*_step$L.nc 2>/dev/null | wc -l || true)
  [[ $n == 12 ]] || fail "$r holds $n of 12 files" 3
done
# box_spectra per row on two regions: the whole box (file bounds less a 1-degree margin) and Europe 35-70 N 25 W-40 E
for r in $ROWS; do
  for reg in box europe; do
    O=$OUT/rows/${r}_L$L/box_spectra_$reg
    [[ $reg == box ]] && BOX=None || BOX="[35.0, 70.0, -25.0, 40.0]"
    if [[ ! -f $O/bins.csv ]]; then
      mkdir -p $O
      ( cd $DST && python -c "
import logging; logging.basicConfig(level=logging.INFO)
from eval.evaluators.box_spectra.runner import run
run('$OUT/pred/${r}_L$L', {}, {'steps': [$lead], 'variables': ['ws','tp','10u','10v','2t','msl'], 'res_deg': 0.035, 'box': $BOX}, output_dir='$O', overwrite=True, run_label='${r}_L${L}_$reg')
" > $O.log 2>&1 ); echo "[o25s] box_spectra $r $reg rc=$? $(date -u +%FT%TZ)"
    fi
    [[ -f $O/bins.csv ]] || fail "no box_spectra $reg for $r (see $O.log)" 5
  done
done
P=$OUT/pairs/L$L; mkdir -p $P
for reg in box europe; do
  python $KIT/scripts/spectra_table.py $lead $P $reg $(for r in $ROWS; do echo -n "$r=$OUT/rows/${r}_L$L/box_spectra_$reg/bins.csv "; done) > $P/spectra_table_$reg.log 2>&1
  grep -q SPECTRA_TABLE_DONE $P/spectra_table_$reg.log || fail "spectra_table $reg" 5
done
python $KIT/scripts/paired_scores_o2560.py $lead $P $(for r in $ROWS; do echo -n "$r=$OUT/pred/${r}_L$L "; done) > $P/paired_scores.log 2>&1
grep -q PAIRED_SCORES_O2560_DONE $P/paired_scores.log || fail "paired_scores" 5
echo "[o25s] O2560_SCORE_DONE L$L $(date -u +%FT%TZ)"
