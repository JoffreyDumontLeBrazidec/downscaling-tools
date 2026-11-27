#!/bin/bash
# CorrDiff Europe on MN5: the lat-lon export on a GPP (CPU) node from the native E2-v4 stores, resumable: init (if the
# export does not exist yet), fill (CD_EXPORT_NP parallel parts, default 48; finished rows are skipped), stats (if
# missing), verify -> $CD/state/VERIFY_OK. The login node kills processes past 300 CPU-s, so the fill runs here.
# The time inputs are not exported: cd_dataset.py (time_inputs: true) computes them per row at its valid time.
#   sbatch [--qos=gp_debug --time=02:00:00] cd_export_mn5.sbatch        (default QOS gp_ehpc, 8 h)
#SBATCH --job-name=cd_m5_export
#SBATCH --account=ehpc01
#SBATCH --qos=gp_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/projects/ehpc01/ecm800825/sandbox/20261127-corrdiff-mn5/logs/%x_%j.out
set -uo pipefail
source /gpfs/projects/ehpc01/ecm800825/sandbox/20261127-corrdiff-mn5/kit/cd_env_mn5.sh
OUT=$E2V4/out_hres_o1280.zarr; IN=$E2V4/in_lres_o320_halo1p5.zarr; FORC=$E2/in_hres_o1280_forcings.zarr
NP=${CD_EXPORT_NP:-48}
echo "export job $SLURM_JOB_ID start $(date -u) host $(hostname) cpus $(nproc) -> $CD_DATA"
# can a compute node submit (the training chains submit their successor from the job)? --test-only submits nothing
sbatch --test-only -A ehpc01 -p acc -q acc_ehpc -N 1 --gres=gpu:4 -c 80 -t 00:10:00 --wrap hostname 2>&1 | sed 's/^/[compute-node sbatch test] /'
[[ -f $CD/state/VERIFY_OK ]] && { echo "VERIFY_OK already: $(cat $CD/state/VERIFY_OK)"; exit 0; }
mkdir -p $CDD $CD/logs/export
if [[ ! -f ${CD_DATA%.zarr}.rows.csv ]]; then
  OMP_NUM_THREADS=16 python $CD_KIT/cd_export.py init --out-store $OUT --in-store $IN --forcings $FORC --dest $CD_DATA || { echo INIT_FAIL; exit 1; }
fi
t0=$SECONDS; pids=()
for P in $(seq 0 $(( NP - 1 ))); do
  OMP_NUM_THREADS=2 python $CD_KIT/cd_export.py fill --out-store $OUT --in-store $IN --dest $CD_DATA --part $P --nparts $NP \
    > $CD/logs/export/fill_${SLURM_JOB_ID}_p$P.log 2>&1 & pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
echo "fill end $(date -u) after $(( SECONDS - t0 )) s: failed_parts=$fail, rows written $(grep -h FILL_DONE $CD/logs/export/fill_${SLURM_JOB_ID}_p*.log | awk '{s+=$4} END {print s+0}')"
(( fail == 0 )) || { echo "FILL_FAIL: $(grep -lE 'Error|Traceback' $CD/logs/export/fill_${SLURM_JOB_ID}_p*.log | head -3)"; exit 1; }
[[ -f ${CD_DATA%.zarr}.stats.json ]] || { OMP_NUM_THREADS=16 python $CD_KIT/cd_export.py stats --dest $CD_DATA || { echo STATS_FAIL; exit 1; }; }
python $CD_KIT/cd_export.py verify --out-store $OUT --in-store $IN --dest $CD_DATA 2>&1 | tee $CD/logs/export/verify_${SLURM_JOB_ID}.log
if grep -q VERIFY_OK $CD/logs/export/verify_${SLURM_JOB_ID}.log; then
  echo "$(grep VERIFY_OK $CD/logs/export/verify_${SLURM_JOB_ID}.log) job $SLURM_JOB_ID $(date -u)" > $CD/state/VERIFY_OK; cat $CD/state/VERIFY_OK
else echo "VERIFY_FAIL"; exit 1; fi
echo "EXPORT_JOB_END $(date -u)"
