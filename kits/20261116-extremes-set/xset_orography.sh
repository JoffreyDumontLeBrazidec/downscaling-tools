#!/bin/bash
# Orography (surface geopotential z, step 0) of the operational 36 km ensemble eefo on O320, one date per model cycle of the
# set (48r1: 20240804; 49r1: 20250201), for the input row's 2t height correction, and their difference in metres.
# Usage: bash xset_orography.sh        Atos, module load ecmwf-toolbox python3
set -eo pipefail
A=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set; G=$A/grib; mkdir -p $G $A/state/req
for pair in 48r1:20240804 49r1:20250201; do
  c=${pair%%:*}; d=${pair##*:}; t=$G/orography_O320_eefo_$c.grib
  [[ -s $t ]] && { echo "have $t"; continue; }
  echo "retrieve,class=od,expver=0001,stream=eefo,type=cf,date=$d,time=00,step=0,levtype=sfc,param=z,target=\"$t.part\"" > $A/state/req/req_orog_$c.txt
  mars $A/state/req/req_orog_$c.txt
  mv $t.part $t; echo "orography $c: $(grib_count $t) field(s), $(stat -c %s $t) bytes"
done
python3 - <<'PY'
import eccodes, numpy as np
A = "/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set/grib/"
def z(p):
    with open(p, "rb") as f:
        h = eccodes.codes_grib_new_from_file(f); v = eccodes.codes_get_values(h) / 9.80665; n = eccodes.codes_get(h, "gridName"); eccodes.codes_release(h)
    return v, n
a, na = z(A + "orography_O320_eefo_48r1.grib"); b, nb = z(A + "orography_O320_eefo_49r1.grib")
print("grids", na, nb, a.size, b.size)
print("OROG_DIFF 49r1 - 48r1 [m]: max |diff| %.3f, rms %.3f" % (np.abs(a - b).max(), np.sqrt(((a - b) ** 2).mean())))
try:
    c, nc_ = z("/home/ecm5702/hpcperm/data/station_caches/orography_O320_eefh.grib")
    print("OROG_DIFF eefo 49r1 - eefh file [m]: max |diff| %.3f, rms %.3f" % (np.abs(b - c).max(), np.sqrt(((b - c) ** 2).mean())))
except Exception as e:
    print("eefh file not compared:", repr(e))
PY
