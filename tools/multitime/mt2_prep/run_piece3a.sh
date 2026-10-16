#!/bin/bash
# Piece 3(a): materialise the donor with the do0 68 -> 70 extension in the do0 sandbox venv (nothing modified there).
S=/home/ecm5702/scratch/sandbox/20261016-multitime-e2
source /ec/res4/scratch/ecm5702/sandbox/20261007-pl25-ac/activate.sh || exit 3
export OUTPUT=$S/runs/materialise_do0 HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=8
mkdir -p $OUTPUT $S/checkpoints/warm
cd $OUTPUT
date -u
/usr/bin/time -v python $S/scripts/materialise_do0_donor.py --out $S/checkpoints/warm/stage_a_do0_donor70
echo rc=$?
date -u
