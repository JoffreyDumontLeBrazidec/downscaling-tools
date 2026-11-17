#!/bin/bash
# Extremes set XSET-1 (16 Nov 2026): MARS -> GRIB files for one start date (rule 30: never FDB). Operational ENS, 00 UTC:
#   input  stream eefo (36 km, O320) pf 1-10: the 70 input variables of GL512 (pressure levels on grid=O320, as the store recipe)
#   truth  stream enfo (9 km, O1280) pf 1-10 and cf: 10u 10v 2t msl tp
# Steps 12..72 by 12 (12 h rain = acc(step) - acc(step-12); acc(0) = 0 needs no request). One request per file = one tape read.
# Usage: xset_retrieve_date.sh <yyyymmdd>     Atos, module load ecmwf-toolbox. Files land in $A/grib, each with a .ok count file.
set -eo pipefail
ulimit -c 0
d=$1
A=${XSET_WORK:-/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set}; G=$A/grib; Q=$A/state/req; mkdir -p $G $Q
B="class=od,expver=0001,date=$d,time=00"
ST="step=12/24/36/48/60/72"
PL="levtype=pl,levelist=50/100/200/300/400/500/700/850/925/1000,param=q/t/u/v/w/z,$ST,grid=O320"
SF="levtype=sfc,param=10u/10v/2d/2t/tcw/sp/skt/msl/tp/cp,$ST"
TR="levtype=sfc,param=10u/10v/2t/msl/tp,$ST"
declare -A WANT=( [in_pl_pf]=3600 [in_sfc_pf]=600 [tr_sfc_pf]=300 [tr_sfc_cf]=30 )
req() {  # name stream type spec
  local t=$G/$1_$d.grib
  [[ -s $t && -f $t.ok ]] && { echo "have $1_$d"; return 0; }
  local num=""; [[ $3 == pf ]] && num=",number=1/to/10"
  echo "retrieve,$B,stream=$2,type=$3$num,$4,target=\"$t.part\"" > $Q/req_$1_$d.txt
  echo "[ret] $(date -u +%FT%TZ) start $1_$d"
  mars $Q/req_$1_$d.txt
  local n; n=$(grib_count $t.part)
  [[ "$n" == "${WANT[$1]}" ]] || { echo "[ret] FAIL $1_$d: $n fields, want ${WANT[$1]}"; return 1; }
  mv $t.part $t; echo $n > $t.ok
  echo "[ret] $(date -u +%FT%TZ) done $1_$d fields=$n bytes=$(stat -c %s $t)"
}
req in_sfc_pf eefo pf "$SF"
req in_pl_pf  eefo pf "$PL"
req tr_sfc_cf enfo cf "$TR"
req tr_sfc_pf enfo pf "$TR"
echo "[ret] DATE_DONE $d $(date -u +%FT%TZ)"
