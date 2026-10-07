#!/bin/bash
# Campaign T1d stage A (2026-10-07), gate G1: ONE draw of p12m_pw30_c0 (20230826, lead 24, member 1, base seed 756,
# ANEMOI_BASE_SEED unset) under the fresh patched sandbox AND under the certified venv, one A100 each on AC.
# Compare afterwards on the login node with g1_compare.py (LAUNCH.md step G1). Run on hpc-login, after the owner's go:
#   TEST=1 bash submit_gates_t1d.sh    # sbatch --test-only, nothing queued (gate G3)
#   bash submit_gates_t1d.sh           # submit
set -u
source "${T1D_CODE:-/home/ecm5702/work-t1d-20261007/code/downscaling-tools}/scripts/t1d_sampler_20261007/t1d_env.sh" || exit 2
TSV=$T1D_W/notes/jobs.tsv; mkdir -p "$T1D_W/notes" "$T1D_W/logs"
T=(); [[ "${TEST:-0}" == 1 ]] && T=(--test-only)
[[ -f $TSV || "${TEST:-0}" == 1 ]] || echo -e "kind\tarm\tlane\tseed\trun_root\tdepends\tjobid" > $TSV
rec() { [[ "${TEST:-0}" == 1 ]] || echo -e "$1" >> $TSV; echo -e "$1"; }
jid() { local out; out=$(sbatch --parsable "$@" 2>&1); local rc=$?; [[ "${TEST:-0}" == 1 ]] && { echo "TEST: $out" >&2; echo TEST; return 0; }; [[ $rc == 0 ]] && echo "${out%%;*}" || { echo "SUBMIT FAILED: $out" >&2; return 1; }; }
LANE=tc_o320_o1280_p12m_pw30_c0
for RT in sandbox certified; do
  RR=$T1D_E/o320_o1280_p12m_pw30_c0_g1_${RT}_$T1D_TAG
  [[ -e $RR ]] && { echo "REFUSED: $RR exists"; exit 1; }
done
for RT in sandbox certified; do
  RR=$T1D_E/o320_o1280_p12m_pw30_c0_g1_${RT}_$T1D_TAG
  j=$(jid "${T[@]}" --job-name=t1d_g1_$RT --output=$T1D_W/logs/%x_%j.out --time=00:45:00 \
      $T1D_S/se_tc_predict_t1d.sbatch p12m_pw30_c0 $LANE 756 $RR $RT 20230826 24 1) || exit 1
  rec "gate_g1\tp12m_pw30_c0\t$LANE\t756\t$RR\t-\t$j"
done
