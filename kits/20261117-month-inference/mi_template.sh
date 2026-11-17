#!/bin/bash
# MI-S25: the O1280 GRIB key template of mi_encode_grib.py = od enfo pf 1, 20250901 00 UTC, step 12, the 10 surface outputs.
set -eo pipefail
A=${MI_WORK:-/ec/res4/scratch/ecm5702/agent-work/20261117-month-inference}; mkdir -p $A/grib
t=$A/grib/tmpl_o1280.grib
[[ -s $t ]] && { echo "have $t"; exit 0; }
echo 'retrieve,class=od,expver=0001,stream=enfo,type=pf,number=1,date=20250901,time=00,step=12,levtype=sfc,param=10u/10v/2d/2t/msl/sp/skt/tcw/tp/cp,target="'$t'.part"' > $A/grib/tmpl_req.txt
mars $A/grib/tmpl_req.txt && mv $t.part $t && grib_ls -p shortName,gridType,N,numberOfValues,bitsPerValue,edition $t
