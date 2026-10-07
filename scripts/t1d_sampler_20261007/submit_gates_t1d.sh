#!/bin/bash
# Campaign T1d stage A (2026-10-07), gate G1: ONE draw of p12m_pw30_c0 (20230826, lead 24, member 1, base seed 756,
# ANEMOI_BASE_SEED unset) under the fresh patched sandbox AND under the certified venv, one A100 each on AC.
# Compare afterwards on the login node with g1_compare.py (LAUNCH.md step G1). Run after the owner's go on hpc-login
# (T1D_HOST=ac, default: A100, sandbox venv vs ~/dev/.ds-260612) or on ag-login with T1D_HOST=ag (GH200, sandbox overlay
# on ~/dev/.ds-ag-260616 vs ~/dev/.ds-ag-260616): both gate jobs always run on the SAME cluster.
#   TEST=1 bash submit_gates_t1d.sh    # sbatch --test-only, nothing queued (gate G3)
#   bash submit_gates_t1d.sh           # submit
set -u
source "${T1D_CODE:-/home/ecm5702/work-t1d-20261007/code/downscaling-tools}/scripts/t1d_sampler_20261007/t1d_env.sh" || exit 2
TSV=$T1D_W/notes/jobs.tsv; mkdir -p "$T1D_W/notes" "$T1D_W/logs"
T=(); [[ "${TEST:-0}" == 1 ]] && T=(--test-only)
[[ -f $TSV || "${TEST:-0}" == 1 ]] || echo -e "kind\tarm\tlane\tseed\trun_root\tdepends\tjobid\tcluster" > $TSV
rec() { [[ "${TEST:-0}" == 1 ]] || echo -e "$1" >> $TSV; echo -e "$1"; }
# every sbatch call carries the campaign settings explicitly (Atos: SBATCH_EXPORT=NONE); TEST=1 prints the full line
EXP=$(t1d_export) || exit 2
jid() { local out; [[ "${TEST:-0}" == 1 ]] && echo "TEST CMD: sbatch --parsable $EXP $*" >&2; out=$(sbatch --parsable "$EXP" "$@" 2>&1); local rc=$?; [[ "${TEST:-0}" == 1 ]] && { echo "TEST: $out" >&2; echo TEST; return 0; }; [[ $rc == 0 ]] && echo "${out%%;*}" || { echo "SUBMIT FAILED: $out" >&2; return 1; }; }
LANE=tc_o320_o1280_p12m_pw30_c0
[[ "$T1D_HOST" == ag && "$(hostname)" != ag* ]] && echo "WARNING T1D_HOST=ag: submit from ag-login (this is $(hostname))" >&2
[[ "$T1D_HOST" == ac && "$(hostname)" == ag* ]] && echo "WARNING T1D_HOST=ac on $(hostname): AC jobs are submitted from hpc-login" >&2
RTS=(${ONLY:-sandbox certified})   # ONLY=certified (or sandbox) resubmits one gate job; its old root must be moved aside
for RT in "${RTS[@]}"; do
  [[ $RT == sandbox || $RT == certified ]] || { echo "ONLY must be sandbox or certified"; exit 2; }
  RR=$(t1d_g1_root $RT)
  [[ -e $RR ]] && { echo "REFUSED: $RR exists"; exit 1; }
done
for RT in "${RTS[@]}"; do
  RR=$(t1d_g1_root $RT)
  j=$(jid "${T[@]}" $(t1d_host_opts) --job-name=t1d_g1_$(t1d_jn)$RT --output=$T1D_W/logs/%x_%j.out --time=00:45:00 \
      $T1D_S/se_tc_predict_t1d.sbatch p12m_pw30_c0 $LANE 756 $RR $RT 20230826 24 1) || exit 1
  rec "gate_g1\tp12m_pw30_c0\t$LANE\t756\t$RR\t-\t$j\t$T1D_HOST"
done
