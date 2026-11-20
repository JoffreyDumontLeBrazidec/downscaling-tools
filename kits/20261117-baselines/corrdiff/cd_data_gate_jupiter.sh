#!/bin/bash
# CorrDiff on Jupiter: data gate run at the start of every job (source it after cd_env_jupiter.sh).
# Waits for the login-node fill (state/FILL_RUNNING) up to 90 min, then completes the fill itself (32 parts at once,
# resumable: finished rows are skipped), writes stats if missing, runs verify and writes state/VERIFY_OK.
# Several jobs can pass through it: state/GATE.lock makes the second one wait for the first.
cd_data_gate() {
  local OUT IN k fail
  OUT=$(cat $CD/E2V4_OUT_STORE); IN=$(cat $CD/E2V4_IN_STORE)
  [[ -f $CD/state/VERIFY_OK ]] && { echo "[gate] VERIFY_OK present: $(cat $CD/state/VERIFY_OK)"; return 0; }
  for k in $(seq 1 90); do [[ -f $CD/state/FILL_RUNNING || -f $CD/state/GATE.lock ]] || break; [[ $k == 1 ]] && echo "[gate] waiting for login fill / other gate $(date -u)"; sleep 60; done
  [[ -f $CD/state/VERIFY_OK ]] && { echo "[gate] VERIFY_OK present after wait"; return 0; }
  echo "$SLURM_JOB_ID $(date -u)" > $CD/state/GATE.lock
  echo "[gate] fill pass start $(date -u)"
  local pids=() P
  for P in $(seq 0 31); do OMP_NUM_THREADS=2 python3 $CD_KIT/cd_export.py fill --out-store $OUT --in-store $IN --dest $CD_DATA --part $P --nparts 32 > $CD/logs/gate_fill_${SLURM_JOB_ID}_p$P.log 2>&1 & pids+=($!); done
  fail=0; for P in "${pids[@]}"; do wait $P || fail=$((fail+1)); done
  echo "[gate] fill pass end $(date -u) failed_parts=$fail; rows written: $(grep -h FILL_DONE $CD/logs/gate_fill_${SLURM_JOB_ID}_p*.log | awk '{s+=$5} END {print s}')"
  (( fail == 0 )) || { rm -f $CD/state/GATE.lock; echo "[gate] FATAL fill failed"; return 1; }
  [[ -f ${CD_DATA%.zarr}.stats.json ]] || python3 $CD_KIT/cd_export.py stats --dest $CD_DATA || { rm -f $CD/state/GATE.lock; return 1; }
  if python3 $CD_KIT/cd_export.py verify --out-store $OUT --in-store $IN --dest $CD_DATA 2>&1 | tee $CD/logs/verify_${SLURM_JOB_ID}.log | grep -q VERIFY_OK; then
    echo "$(grep VERIFY_OK $CD/logs/verify_${SLURM_JOB_ID}.log) job $SLURM_JOB_ID $(date -u)" > $CD/state/VERIFY_OK
    rm -f $CD/state/GATE.lock; echo "[gate] $(cat $CD/state/VERIFY_OK)"; return 0
  fi
  rm -f $CD/state/GATE.lock; echo "[gate] FATAL verify failed"; cat $CD/logs/verify_${SLURM_JOB_ID}.log | tail -5; return 1
}
