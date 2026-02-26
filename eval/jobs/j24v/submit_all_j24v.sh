#!/bin/bash
set -euo pipefail
BASE="/home/ecm5702/dev/downscaling-tools/eval/jobs/j24v"
mkdir -p /home/ecm5702/perm/eval/j24v/logs

jid_eval=$(sbatch "${BASE}/eval_mars_j24v_lite.sbatch" | awk "{print \$NF}")
echo "eval_mars_j24v job: ${jid_eval}"

jid_quaver=$(sbatch "${BASE}/quaver_j24v_lite.sbatch" | awk "{print \$NF}")
echo "quaver_j24v job: ${jid_quaver}"

jid_spectra=$(sbatch "${BASE}/spectra_j24v_lite.sbatch" | awk "{print \$NF}")
echo "spectra_j24v job: ${jid_spectra}"

jid_tc_get=$(sbatch "${BASE}/tc_retrieve_missing_j24v.sbatch" | awk "{print \$NF}")
echo "tc_retrieve_missing_j24v job: ${jid_tc_get}"

jid_tc_plot=$(sbatch --dependency=afterok:${jid_tc_get} "${BASE}/tc_plot_j24v.sbatch" | awk "{print \$NF}")
echo "tc_plot_j24v job: ${jid_tc_plot}"

printf "\nSubmitted jobs:\n"
printf "  eval_mars_j24v      %s\n" "${jid_eval}"
printf "  quaver_j24v         %s\n" "${jid_quaver}"
printf "  spectra_j24v        %s\n" "${jid_spectra}"
printf "  tc_retrieve_j24v    %s\n" "${jid_tc_get}"
printf "  tc_plot_j24v        %s (afterok:%s)\n" "${jid_tc_plot}" "${jid_tc_get}"
