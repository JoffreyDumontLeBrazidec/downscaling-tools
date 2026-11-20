#!/bin/bash
# Login node, after login_fill.sh: wait for its end, check every group had failed_parts=0, then stats + verify under
# state/GATE.lock (jobs reaching the gate wait for it) and write state/VERIFY_OK in the gate's format.
set -uo pipefail
source /e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261120-corrdiff-jupiter/kit/cd_env_jupiter.sh
cd $CD; OUT=$(cat E2V4_OUT_STORE); IN=$(cat E2V4_IN_STORE)
while [[ -f state/FILL_RUNNING ]]; do sleep 60; done
echo "fill ended $(date -u)"; grep -E "group|LOGIN_FILL_END" logs/login_fill.log
[[ -f state/VERIFY_OK ]] && { echo "VERIFY_OK already: $(cat state/VERIFY_OK)"; exit 0; }
if [[ $(grep -c "failed_parts=0" logs/login_fill.log) != 4 ]]; then echo "LOGIN_VERIFY_SKIP: a fill group failed; the job gate will complete the fill"; exit 1; fi
[[ -f state/GATE.lock ]] && { echo "LOGIN_VERIFY_SKIP: a job gate holds the lock: $(cat state/GATE.lock)"; exit 0; }
echo "login $(date -u)" > state/GATE.lock
t0=$SECONDS
nice -n 10 python3 $CD_KIT/cd_export.py stats --dest $CD_DATA || { rm -f state/GATE.lock; echo STATS_FAIL; exit 1; }
echo "stats took $((SECONDS-t0)) s"; t0=$SECONDS
if nice -n 10 python3 $CD_KIT/cd_export.py verify --out-store $OUT --in-store $IN --dest $CD_DATA 2>&1 | tee logs/verify_login.log | grep -q VERIFY_OK; then
  echo "$(grep VERIFY_OK logs/verify_login.log) job login $(date -u)" > state/VERIFY_OK; echo "verify took $((SECONDS-t0)) s: $(cat state/VERIFY_OK)"
else echo "VERIFY_FAIL: $(tail -3 logs/verify_login.log)"; fi
rm -f state/GATE.lock
