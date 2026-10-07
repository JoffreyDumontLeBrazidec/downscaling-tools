#!/bin/bash
# Campaign T1d stage A (2026-10-07): the box screen of the 1.2M parent. Five prediction runs (one A100 each on AC, qos ng,
# 100 draws: 5 dates x leads 24,120 x members 1-10), each followed (afterok) by its evaluation with all six stage-1
# evaluators; then (afterok on the five evaluations) the per-draw intensity table, the v3 box spectra and the paired read.
# Run on hpc-login ONLY after gates G1 and G2 passed and the owner typed his go:
#   TEST=1 bash submit_stageA_t1d.sh   # sbatch --test-only of every job, no dependencies, nothing queued (gate G3)
#   bash submit_stageA_t1d.sh          # submit; job ids go to $T1D_W/notes/jobs.tsv
# Refuses (before submitting anything) if any run root exists.
set -u
source "${T1D_CODE:-/home/ecm5702/work-t1d-20261007/code/downscaling-tools}/scripts/t1d_sampler_20261007/t1d_env.sh" || exit 2
TSV=$T1D_W/notes/jobs.tsv; mkdir -p "$T1D_W/notes" "$T1D_W/logs"
T=(); [[ "${TEST:-0}" == 1 ]] && T=(--test-only)
[[ -f $TSV || "${TEST:-0}" == 1 ]] || echo -e "kind\tarm\tlane\tseed\trun_root\tdepends\tjobid" > $TSV
rec() { [[ "${TEST:-0}" == 1 ]] || echo -e "$1" >> $TSV; echo -e "$1"; }
jid() { local out; out=$(sbatch --parsable "$@" 2>&1); local rc=$?; [[ "${TEST:-0}" == 1 ]] && { echo "TEST: $out" >&2; echo TEST; return 0; }; [[ $rc == 0 ]] && echo "${out%%;*}" || { echo "SUBMIT FAILED: $out" >&2; return 1; }; }
dep() { [[ "${TEST:-0}" == 1 ]] && return; echo "--dependency=$1"; }
L=--output=$T1D_W/logs/%x_%j.out
# arm seed walltime (pw30 ~100 min at 60 s/draw; the others in proportion to the calls; ~2x margin)
SPECS=("p12m_pw30_c0 756 04:00:00" "p12m_pw30_c0 757 04:00:00" "p12m_c0_pw16_s1k 756 02:30:00" "p12m_st2 756 02:30:00" "p12m_st4 756 03:00:00")
for spec in "${SPECS[@]}"; do read -r A SEED _ <<<"$spec"; RR=$(t1d_root $A $SEED); [[ -e $RR ]] && { echo "REFUSED: $RR exists"; exit 1; }; done
EV=()
for spec in "${SPECS[@]}"; do
  read -r A SEED WT <<<"$spec"; LANE=tc_o320_o1280_$A; RR=$(t1d_root $A $SEED); TAGN=$A$([[ $SEED == 757 ]] && echo _r757)
  j=$(jid "${T[@]}" --job-name=t1d_pred_$TAGN $L --time=$WT $T1D_S/se_tc_predict_t1d.sbatch $A $LANE $SEED $RR sandbox) || exit 1
  rec "predict_box\t$A\t$LANE\t$SEED\t$RR\t-\t$j"
  e=$(jid "${T[@]}" $(dep afterok:$j) --job-name=t1d_eval_$TAGN $L $T1D_S/se_tc_eval_t1d.sbatch $LANE $RR) || exit 1
  rec "eval_box\t$A\t$LANE\t$SEED\t$RR\tafterok:$j\t$e"; EV+=($e)
done
D=$(IFS=:; echo "${EV[*]}")
p=$(jid "${T[@]}" $(dep afterok:$D) $L $T1D_S/box_post_t1d.sbatch) || exit 1; rec "tc_intensity\tall\t-\t-\t-\tafterok:$D\t$p"
s=$(jid "${T[@]}" $(dep afterok:$D) $L $T1D_S/spectra_t1d.sbatch) || exit 1; rec "spectra_v3\tall\t-\t-\t-\tafterok:$D\t$s"
r=$(jid "${T[@]}" $(dep afterok:$D) $L $T1D_S/read_t1d.sbatch) || exit 1; rec "read_stage1\tall\t-\t-\t-\tafterok:$D\t$r"
