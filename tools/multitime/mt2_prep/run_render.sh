#!/bin/bash
# Piece 4: render the effective (Hydra-composed, resolved) mt2_e2_20k config with the branch's own CLI, then check it.
source /home/ecm5702/scratch/sandbox/20261016-multitime-e2/scripts/vera_env.sh
export OUTPUT=$S/runs/mt2_e2_20k
mkdir -p /tmp/$USER/render; cd /tmp/$USER/render; date -u
R=$S/configs/mt2_e2_20k.rendered.yaml
[ -e $R ] && mv $R $R.bak-$(date -u +%Y%m%dT%H%M%S)
/home/ecm5702/hpcperm/sandbox/20260917-vera-spatial-downscaler/.venv-x86_64/bin/anemoi-training train --config-path=$S/configs --config-name=mt2_e2_20k --cfg job --resolve > $R 2> $S/logs/render.err
echo render_rc=$? lines=$(wc -l < $R)
head -3 $R
$PY $S/scripts/check_mt2_config.py --rendered $R --out $S/configs/CHECK.md --jobs $S/jobs/smoke_mt2.sbatch $S/jobs/mt2_e2_20k.sbatch $S/jobs/mt2_job.sh
echo check_rc=$?
date -u
