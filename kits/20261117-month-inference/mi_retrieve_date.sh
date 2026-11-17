#!/bin/bash
# Month inference set MI-S25 (17 Nov 2026): MARS -> GRIB files for one start date (rule 30: never FDB). Operational ENS, 00 UTC:
#   input stream eefo (36 km, O320) pf 1-10, the 70 input variables of GL512 (pressure levels on grid=O320, as the store recipe
#   and XSET-1), steps 12..240 every 12 h (12 h rain = acc(step) - acc(step-12); acc(0) = 0 needs no request).
#   The pressure-level request is split in two step halves (one MARS request per file; a failed half is retried alone).
#   in_sfc_pf is KEPT after the store is built: it is the native O320 input row for quaver (route B controls).
# Optional TRUTH=1: stream enfo (9 km, O1280) pf 1-10, 10u 10v 2t msl tp at the same steps (not needed for station scores).
# Usage: mi_retrieve_date.sh <yyyymmdd>    Atos, module load ecmwf-toolbox. Files land in $MI_WORK/grib, each with a .ok count file.
set -eo pipefail
ulimit -c 0
d=$1
A=${MI_WORK:-/ec/res4/scratch/ecm5702/agent-work/20261117-month-inference}; G=$A/grib; Q=$A/state/req; mkdir -p $G $Q
B="class=od,expver=0001,date=$d,time=00"
S1="step=12/24/36/48/60/72/84/96/108/120"
S2="step=132/144/156/168/180/192/204/216/228/240"
LV="levtype=pl,levelist=50/100/200/300/400/500/700/850/925/1000,param=q/t/u/v/w/z"
SF="levtype=sfc,param=10u/10v/2d/2t/tcw/sp/skt/msl/tp/cp,$S1/${S2#step=}"
TR="levtype=sfc,param=10u/10v/2t/msl/tp,$S1/${S2#step=}"
declare -A WANT=( [in_sfc_pf]=2000 [in_pl_pf_a]=6000 [in_pl_pf_b]=6000 [tr_sfc_pf]=1000 )
req() {  # name stream type spec
  local t=$G/$1_$d.grib
  [[ -s $t && -f $t.ok ]] && { echo "have $1_$d"; return 0; }
  local num=""; [[ $3 == pf ]] && num=",number=1/to/10"
  echo "retrieve,$B,stream=$2,type=$3$num,$4,target=\"$t.part\"" > $Q/req_$1_$d.txt
  echo "[ret] $(date -u +%FT%TZ) start $1_$d"
  local t0; t0=$(date +%s)
  mars $Q/req_$1_$d.txt
  local n; n=$(grib_count $t.part)
  [[ "$n" == "${WANT[$1]}" ]] || { echo "[ret] FAIL $1_$d: $n fields, want ${WANT[$1]}"; return 1; }
  mv $t.part $t; echo $n > $t.ok
  echo "[ret] $(date -u +%FT%TZ) done $1_$d fields=$n bytes=$(stat -c %s $t) seconds=$(( $(date +%s) - t0 ))"
}
req in_sfc_pf  eefo pf "$SF"
req in_pl_pf_a eefo pf "$LV,$S1,grid=O320"
req in_pl_pf_b eefo pf "$LV,$S2,grid=O320"
[[ "${TRUTH:-0}" == 1 ]] && req tr_sfc_pf enfo pf "$TR"
echo "[ret] DATE_DONE $d $(date -u +%FT%TZ)"
