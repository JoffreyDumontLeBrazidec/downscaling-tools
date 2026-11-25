#!/bin/bash
# S24 (25 Nov 2026): MARS -> GRIB of the 9 km ENS for one start date of Sep 2024 (rule 30: never FDB). od enfo 0001, 00 UTC,
# 10u 10v 2t msl tp, steps 12..240 every 12 h: tr_sfc_pf (pf 1-10, 1000 fields) and tr_sfc_cf (cf, 100 fields), into
# $XSET_WORK/grib with a .ok count file, the names xset_rows_date.py reads. The eefo input is NOT retrieved: the MI-S24 stores
# (month-inference stores/in, linked into $XSET_WORK/stores/in) and in_sfc_pf GRIB already hold it.
# Usage: s24_retrieve_truth.sh <yyyymmdd>    Atos, module load ecmwf-toolbox.
set -eo pipefail; ulimit -c 0
d=$1
A=${XSET_WORK:?source xset_set.env}; G=$A/grib; Q=$A/state/req; mkdir -p $G $Q
ST="step=12/24/36/48/60/72/84/96/108/120/132/144/156/168/180/192/204/216/228/240"
TR="levtype=sfc,param=10u/10v/2t/msl/tp,$ST"
declare -A WANT=( [tr_sfc_pf]=1000 [tr_sfc_cf]=100 )
req() {  # name type
  local t=$G/$1_$d.grib
  [[ -s $t && -f $t.ok ]] && { echo "have $1_$d"; return 0; }
  local num=""; [[ $2 == pf ]] && num=",number=1/to/10"
  echo "retrieve,class=od,expver=0001,date=$d,time=00,stream=enfo,type=$2$num,$TR,target=\"$t.part\"" > $Q/req_$1_$d.txt
  echo "[ret] $(date -u +%FT%TZ) start $1_$d"; local t0; t0=$(date +%s)
  mars $Q/req_$1_$d.txt
  local n; n=$(grib_count $t.part)
  [[ "$n" == "${WANT[$1]}" ]] || { echo "[ret] FAIL $1_$d: $n fields, want ${WANT[$1]}"; return 1; }
  mv $t.part $t; echo $n > $t.ok
  echo "[ret] $(date -u +%FT%TZ) done $1_$d fields=$n bytes=$(stat -c %s $t) seconds=$(( $(date +%s) - t0 ))"
}
req tr_sfc_cf cf
req tr_sfc_pf pf
echo "[ret] DATE_DONE $d $(date -u +%FT%TZ)"
