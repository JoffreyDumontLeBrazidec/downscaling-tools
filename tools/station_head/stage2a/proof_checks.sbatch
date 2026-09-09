#!/bin/bash
#SBATCH --job-name=sh2a_proof_checks
#SBATCH --qos=nf
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/ecm5702/agent-work/20260909-station-head-adapter/logs/%x_%j.out
# The checks that prove the stage 2a bundle route on the one case that has been
# predicted: the input the model saw against the arm A input store, and then the
# station gather with its nearest-index agreement count.
set -uo pipefail
S=/home/ecm5702/dev/downscaling-tools-station-head/tools/station_head/stage2a
PRED=/home/ecm5702/scratch/eval/station_head_2a/proof_case/predictions_20260512_step006.nc
module load python3 vtb ecmwf-toolbox
SI=$(python3 -c "
import csv
rows=list(csv.DictReader(open('/home/ecm5702/agent-work/20260909-station-head-adapter/notes/cases_firstcut.csv')))
print([r['synthetic_index'] for r in rows if r['case_id']=='2026051212_step006'][0])
")
echo "synthetic index of the proof case: $SI"
echo "=================== x and x_interp against the store ==================="
python3 -u $S/check_x_interp.py --prediction "$PRED" --synthetic-index "$SI"
echo "=================== the station gather ==================="
python3 -u $S/gather.py --case 2026051212_step006 --prediction "$PRED"
echo "PROOF_CHECKS_RC=$?"
