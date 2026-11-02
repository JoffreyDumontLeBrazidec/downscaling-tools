#!/bin/bash
# o2560 Europe-box read (2 Nov 2026): submit the read. NOT RUN before the review and Joffrey's go.
# 18 prediction jobs (3 rows x 6 leads, 1 GH200 each), 6 scoring jobs (afterany on the 3 predictions of their lead) and one
# final job (afterany on the 6 scorings). Every job checks its own inputs. Ids go to state/jobs.txt. A noclobber marker
# state/SUBMITTED stops a second submission. Usage: bash submit_all.sh   (or LEADS="..." bash submit_all.sh)
set -eo pipefail
KIT=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read
LEADS=${LEADS:-"12 24 36 48 60 72"}
mkdir -p $KIT/state $KIT/logs
( set -o noclobber; echo "$(date -u +%FT%TZ) LEADS=$LEADS" > $KIT/state/SUBMITTED ) 2>/dev/null || { echo "ALREADY_SUBMITTED: $(cat $KIT/state/SUBMITTED)"; exit 3; }
scores=""
for lead in $LEADS; do
  L=$(printf %03d $lead); preds=""
  for row in A100 B24 B50; do
    j=$(sbatch --parsable --job-name=o25p_${row}_L$L $KIT/kit/predict.sbatch $row $lead); preds="$preds:$j"
    echo "o25p_${row}_L$L $j" | tee -a $KIT/state/jobs.txt
  done
  j=$(sbatch --parsable --job-name=o25s_L$L --dependency=afterany$preds $KIT/kit/score.sbatch $lead); scores="$scores:$j"
  echo "o25s_L$L $j afterany$preds" | tee -a $KIT/state/jobs.txt
done
j=$(sbatch --parsable --job-name=o25_final --dependency=afterany$scores $KIT/kit/final.sbatch $LEADS)
echo "o25_final $j afterany$scores" | tee -a $KIT/state/jobs.txt
echo "SUBMIT_ALL_DONE $(date -u +%FT%TZ)"
