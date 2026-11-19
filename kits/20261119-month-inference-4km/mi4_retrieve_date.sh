#!/bin/bash
# MI4-S26 (19 Nov 2026): MARS -> GRIB files for one start date (rule 30: never FDB). od enfo pf 1-10, 00 UTC, grid=O1280 on every
# request (sfc and q are native O1280 gridpoints, an identity; pl t u v w z are archived spectral T1279 and MIR transforms them,
# exactly the s20 member store recipe kit/retrieve_storm.sh). Files in $MI4_A/grib, each with a .ok field count:
#   sfq_<d>  10u 10v 2d 2t msl, steps 12..240/6            1950 fields   KEPT after the build (native input for quaver)
#   sfx_<d>  skt tcw, same steps                             780 fields   deleted after the build
#   tp_<d>   tp cp, steps 6..240/6 (6 h windows)             800 fields   KEPT after the build
#   pl_gNN_<d>  q t u v w z at 500/700/850/925/1000, 3 steps each (13 groups: 12-24, 30-42, ..., 228-240)  900 fields, deleted
# Gate: waits (inside the job) while MI4_CAP other dates are on Atos between retrieval start and verified copy to Jupiter.
# Stop: touch $MI4_A/state/STOP and every later job exits 0 without work.
# Usage: mi4_retrieve_date.sh <yyyymmdd>
set -eo pipefail
ulimit -c 0
source "$(cd "$(dirname "$0")" && pwd)/mi4_set.sh"
d=$1; A=$MI4_A; G=$A/grib; Q=$A/state/req; mkdir -p $G $Q
[[ -f $A/state/STOP ]] && { echo "[ret] STOP file, nothing done for $d"; exit 0; }
if [[ -f $A/state/copied_$d.ok || -f $A/stores/check/in_$d.done ]]; then echo "[ret] $d already built"; echo "[ret] DATE_DONE $d"; exit 0; fi
# gate on Atos disk: dates started and not yet copied (other than this one)
while :; do
  n=0; for s in $A/state/started_*; do [[ -e $s ]] || continue; x=${s##*_}; [[ $x == $d || -f $A/state/copied_$x.ok ]] || n=$((n + 1)); done
  (( n < MI4_CAP )) && break
  [[ -f $A/state/STOP ]] && { echo "[ret] STOP while gated"; exit 0; }
  echo "[ret] $(date -u +%FT%TZ) gate: $n dates on Atos >= cap $MI4_CAP, waiting"; sleep 300
done
date -u +%FT%TZ > $A/state/started_$d
B="class=od,expver=0001,stream=enfo,type=pf,number=1/to/10,date=$d,time=00,grid=O1280"
ST="step=12/to/240/by/6"
req() {  # name want spec
  local t=$G/$1_$d.grib
  [[ -s $t && -f $t.ok ]] && { echo "[ret] have $1_$d"; return 0; }
  rm -f $t.part
  echo "retrieve,$B,$3,target=\"$t.part\"" > $Q/req_$1_$d.txt
  local t0; t0=$(date +%s); echo "[ret] $(date -u +%FT%TZ) start $1_$d"
  mars $Q/req_$1_$d.txt > $A/logs/mars_$1_$d.log 2>&1 || { echo "[ret] FAIL mars $1_$d (log logs/mars_$1_$d.log)"; return 1; }
  local n; n=$(grib_count $t.part)
  [[ "$n" == "$2" ]] || { echo "[ret] FAIL $1_$d: $n fields, want $2"; return 1; }
  mv $t.part $t; echo $n > $t.ok
  echo "[ret] $(date -u +%FT%TZ) done $1_$d fields=$n bytes=$(stat -c %s $t) seconds=$(( $(date +%s) - t0 ))"
}
T0=$(date +%s)
# pressure-level groups first (the slow part: tape + spectral transform), MI4_PAR at a time; a failed group is retried once
grp() { local g=$1; local s=$((12 + 18 * (g - 1))); printf -v nn "%02d" $g
  req pl_g${nn} 900 "levtype=pl,levelist=500/700/850/925/1000,param=q/t/u/v/w/z,step=$s/$((s + 6))/$((s + 12))" \
  || req pl_g${nn} 900 "levtype=pl,levelist=500/700/850/925/1000,param=q/t/u/v/w/z,step=$s/$((s + 6))/$((s + 12))"; }
pids=(); rc=0
for g in $(seq 1 13); do
  grp $g & pids+=($!)
  if (( ${#pids[@]} >= MI4_PAR )); then wait ${pids[0]} || rc=1; pids=("${pids[@]:1}"); fi
done
for p in "${pids[@]}"; do wait $p || rc=1; done
(( rc == 0 )) || { echo "[ret] FAIL pressure levels $d"; exit 1; }
req sfq 1950 "levtype=sfc,param=10u/10v/2d/2t/msl,$ST"
req sfx 780  "levtype=sfc,param=skt/tcw,$ST"
req tp  800  "levtype=sfc,param=tp/cp,step=6/to/240/by/6"
echo "[ret] DATE_DONE $d $(date -u +%FT%TZ) seconds=$(( $(date +%s) - T0 )) bytes=$(stat -c %s $G/*_$d.grib | awk '{s+=$1} END {print s}')"
