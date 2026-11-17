#!/bin/bash
# MI-S25 read-only probe on Atos: are eefo pf 1-10 sfc and pl at steps 12..240 archived for 1 and 30 Sep 2025 (MARS list,
# no data moved), how much space is used, and where a durable copy can go.
A=/ec/res4/scratch/ecm5702/agent-work/20261117-month-inference; mkdir -p $A/state
module load ecmwf-toolbox >/dev/null 2>&1
for d in 20250901 20250930; do
  for lt in "levtype=sfc,param=10u/10v/2d/2t/tcw/sp/skt/msl/tp/cp" "levtype=pl,levelist=50/100/200/300/400/500/700/850/925/1000,param=q/t/u/v/w/z"; do
    echo "list,class=od,expver=0001,stream=eefo,type=pf,number=1/to/10,date=$d,time=00,step=12/to/240/by/12,$lt,output=cost" > $A/state/list.req
    echo "== $d ${lt%%,*}"; mars $A/state/list.req 2>&1 | grep -iE "entries|size|online|tape|grand|error|warn" | head -8
  done
done
echo "== quota"; quota -s 2>/dev/null | tail -5; lfs quota -h -u ecm5702 /ec/res4/scratch 2>/dev/null | tail -3
echo "== /ec/ai writable?"; for p in /ec/ai/project/ai-ml/datasets /ec/ai/project/ai-ml; do [[ -w $p ]] && echo "writable $p" || echo "not writable $p"; done
df -h /ec/ai/project/ai-ml/datasets /ec/res4/hpcperm/ecm5702 2>/dev/null
echo "PROBE_DONE $(date -u +%FT%TZ)"
