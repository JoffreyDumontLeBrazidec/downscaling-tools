#!/bin/bash
# Piece 3 forward checks on the login-node CPU (float32, pyg attention), one side at a time.
# usage: run_fwd.sh <side> <interface> <out-name> [venv: ours|vera]
S=/home/ecm5702/scratch/sandbox/20261016-multitime-e2; W=$S/checkpoints/warm
side=$1; iface=$2; name=$3; env=${4:-vera}
export OMP_NUM_THREADS=16 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ANEMOI_INFERENCE_NUM_CHUNKS=16
if [[ $env == ours ]]; then
  source /ec/res4/scratch/ecm5702/sandbox/20261007-pl25-ac/activate.sh || exit 3; PY=python
else
  source $S/scripts/vera_env.sh; export OMP_NUM_THREADS=16 ANEMOI_INFERENCE_NUM_CHUNKS=16
fi
cd /tmp; date -u
if [[ ! -f $W/fwd_input_seed20261016.pt ]]; then
  $PY $S/scripts/warm_start_fwdcheck.py make-input --grid 814810 --out $W/fwd_input_seed20261016.pt || exit 1
fi
/usr/bin/time -v $PY $S/scripts/warm_start_fwdcheck.py run --side $side --interface $iface --input $W/fwd_input_seed20261016.pt \
  --out $W/fwd_$name.pt --device cpu --backend pyg --shard-strategy edges
echo rc=$?; date -u
