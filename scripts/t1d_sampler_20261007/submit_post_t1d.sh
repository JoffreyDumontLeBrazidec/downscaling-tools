#!/bin/bash
# Campaign T1d stage A, AG route (T1D_HOST=ag): the AC half. Run on hpc-login once the five AG prediction runs are
# COMPLETED (Slurm cannot chain afterok from AG to AC). Checks that every run root holds its 10 prediction files and a
# "SE_TC_T1D ... rc=0" log line, then submits on AC the five evaluations (no dependency: their inputs exist) and the
# intensity, spectra and read jobs (afterok on the five evaluations). Same jobs as the AC route.
#   TEST=1 bash submit_post_t1d.sh    # sbatch --test-only (skips the file checks only when TEST=1 and SKIPCHECK=1)
#   bash submit_post_t1d.sh
set -u
source "${T1D_CODE:-/home/ecm5702/work-t1d-20261007/code/downscaling-tools}/scripts/t1d_sampler_20261007/t1d_env.sh" || exit 2
TSV=$T1D_W/notes/jobs.tsv; mkdir -p "$T1D_W/notes" "$T1D_W/logs"
T=(); [[ "${TEST:-0}" == 1 ]] && T=(--test-only)
[[ -f $TSV || "${TEST:-0}" == 1 ]] || echo -e "kind\tarm\tlane\tseed\trun_root\tdepends\tjobid\tcluster" > $TSV
rec() { [[ "${TEST:-0}" == 1 ]] || echo -e "$1" >> $TSV; echo -e "$1"; }
jid() { local out; out=$(sbatch --parsable "$@" 2>&1); local rc=$?; [[ "${TEST:-0}" == 1 ]] && { echo "TEST: $out" >&2; echo TEST; return 0; }; [[ $rc == 0 ]] && echo "${out%%;*}" || { echo "SUBMIT FAILED: $out" >&2; return 1; }; }
dep() { [[ "${TEST:-0}" == 1 ]] && return; echo "--dependency=$1"; }
[[ "$(hostname)" == ag* ]] && echo "WARNING: this half runs on AC; submit from hpc-login (this is $(hostname))" >&2
L=--output=$T1D_W/logs/%x_%j.out
SPECS=("p12m_pw30_c0 756" "p12m_pw30_c0 757" "p12m_c0_pw16_s1k 756" "p12m_st2 756" "p12m_st4 756")
bad=0
for spec in "${SPECS[@]}"; do
  read -r A SEED <<<"$spec"; RR=$(t1d_root $A $SEED)
  n=$(ls "$RR"/predictions/predictions_*.nc 2>/dev/null | wc -l)
  ok=$(awk -F'\t' -v rr="$RR" '$1=="predict_box" && $5==rr {print $7}' "$TSV" 2>/dev/null | tail -1 | xargs -I{} sh -c "grep -l 'SE_TC_T1D .* rc=0 files=10' $T1D_W/logs/*_{}.out 2>/dev/null" | wc -l)
  [[ -e "$RR/eval_stageA" ]] && { echo "REFUSED: $RR/eval_stageA exists"; exit 1; }
  echo "$A seed $SEED: files=$n/10 rc0_log=$ok"
  [[ "$n" == 10 && "$ok" -ge 1 ]] || bad=1
done
[[ $bad == 0 || ( "${TEST:-0}" == 1 && "${SKIPCHECK:-0}" == 1 ) ]] || { echo "REFUSED: not every AG prediction run is complete"; exit 1; }
EV=()
for spec in "${SPECS[@]}"; do
  read -r A SEED <<<"$spec"; LANE=tc_o320_o1280_$A; RR=$(t1d_root $A $SEED); TAGN=$A$([[ $SEED == 757 ]] && echo _r757)
  e=$(jid "${T[@]}" --job-name=t1d_eval_$TAGN $L $T1D_S/se_tc_eval_t1d.sbatch $LANE $RR) || exit 1
  rec "eval_box\t$A\t$LANE\t$SEED\t$RR\t-\t$e\tac"; EV+=($e)
done
D=$(IFS=:; echo "${EV[*]}")
p=$(jid "${T[@]}" $(dep afterok:$D) $L $T1D_S/box_post_t1d.sbatch) || exit 1; rec "tc_intensity\tall\t-\t-\t-\tafterok:$D\t$p\tac"
s=$(jid "${T[@]}" $(dep afterok:$D) $L $T1D_S/spectra_t1d.sbatch) || exit 1; rec "spectra_v3\tall\t-\t-\t-\tafterok:$D\t$s\tac"
r=$(jid "${T[@]}" $(dep afterok:$D) $L $T1D_S/read_t1d.sbatch) || exit 1; rec "read_stage1\tall\t-\t-\t-\tafterok:$D\t$r\tac"
