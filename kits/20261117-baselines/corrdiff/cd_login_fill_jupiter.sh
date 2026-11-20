#!/bin/bash
# Export fill on the Jupiter login node (nice 19, 8 processes, 32 parts as cd_export.sbatch; resumable).
source /e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261120-corrdiff-jupiter/kit/cd_env_jupiter.sh
cd $CD; OUT=$(cat E2V4_OUT_STORE); IN=$(cat E2V4_IN_STORE)
date -u > state/FILL_RUNNING; echo "login fill start $(date -u) host $(hostname)"
NP=32
for T in 0 1 2 3; do
  pids=()
  for j in 0 1 2 3 4 5 6 7; do P=$((T*8+j)); OMP_NUM_THREADS=1 nice -n 19 python3 $CD_KIT/cd_export.py fill --out-store $OUT --in-store $IN --dest $CD_DATA --part $P --nparts $NP > logs/fill_part$P.log 2>&1 & pids+=($!); done
  fail=0; for p in "${pids[@]}"; do wait $p || fail=$((fail+1)); done
  echo "group $T done $(date -u) failed_parts=$fail"
done
rm -f state/FILL_RUNNING; echo "LOGIN_FILL_END $(date -u)"
