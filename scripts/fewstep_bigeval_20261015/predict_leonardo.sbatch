#!/bin/bash
# Predict one arm of the few-step big eval (o96->o320) on one case set, on Leonardo (1 A100), then
# slim its sampler directories for the copy to Atos.
#
#   sbatch --export=ALL,ARM=meanflow,SET=A,CKPT=<by_step ckpt>,SAMPLERS=jumps1,jumps2 ...   # NO: commas split --export
#   Use the env form instead (Leonardo keeps the caller's environment):
#   ARM=meanflow SET=A CKPT=<ckpt> SAMPLERS="jumps1 jumps2 jumps4 vel_heun30" sbatch predict_leonardo.sbatch
#
# SAMPLERS is SPACE-separated (sbatch --export splits at commas); the job joins it with commas.
# Set A = the 36 validation samples at lead 60 h (indices in IDX_A below, note section 7);
# set B = the Idalia bundles, 2023-08-26..30 x 24..120 h, members 1-10.
# Code: W/anemoi-core (flow-map sandbox head ca55d5fd + the six drifting commits + the drift arms
# of predict_members.py); the sandbox venv's editable installs point at N/anemoi-core, so the
# worktree's src directories go first on PYTHONPATH and the job checks the imports resolve there.
#SBATCH --account=DestE_340_26
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --job-name=bigeval-pred
#SBATCH --output=/leonardo_work/DestE_340_26/users/jdumontl/agent-work/20261015-fewstep-bigeval/logs/%x_%j.out
set -euo pipefail

: "${ARM:?ARM}" "${SET:?SET (A or B)}" "${CKPT:?CKPT (by_step checkpoint path)}" "${SAMPLERS:?SAMPLERS (space-separated)}"
N=/leonardo_work/DestE_340_26/users/jdumontl/sandbox/20261002-flowmap-o96o320
W=/leonardo_work/DestE_340_26/users/jdumontl/agent-work/20261015-fewstep-bigeval
DT=${DT:-$W/downscaling-tools}          # checkout of downscaling-tools claude/project-thread-72mhur (slim script)
IDX_A=716,313,403,475,547,619,691,763,835,58,154,250,346,428,40,88,136,184,232,280,328,414,450,486,522,558,594,630,666,702,738,774,810,25,121,217

set +u
# shellcheck disable=SC1091
source "$N/env/activate.sh"
set -u
E=$W/anemoi-core/experiments/flowmap-o96o320
export PYTHONPATH="$W/anemoi-core/models/src:$W/anemoi-core/training/src:$W/anemoi-core/graphs/src:$E/jobs${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1 ANEMOI_BASE_SEED=756
python - <<EOF
import anemoi.models, anemoi.training
for m in (anemoi.models, anemoi.training):
    assert m.__file__.startswith("$W/anemoi-core/"), m.__file__
print("IMPORTS_OK", anemoi.models.__file__)
EOF
echo "JOB ARM=$ARM SET=$SET CKPT=$CKPT SAMPLERS=$SAMPLERS HEAD=$(git -C $W/anemoi-core rev-parse --short HEAD)"

specs=$(echo "$SAMPLERS" | tr ' ' ',')
case "$SET" in
  A) OUT=$W/pred/setA; extra=(--sample-indices "$IDX_A") ;;
  B) OUT=$W/pred/setB; extra=(--bundle-dir "$N/data/idalia" --dates 20230826,20230827,20230828,20230829,20230830
                             --steps 24,48,72,96,120 --pairs 20230828:48,20230829:24) ;;
  *) echo "SET must be A or B" >&2; exit 2 ;;
esac
mkdir -p "$OUT"
python "$E/predict/predict_members.py" --arm "$ARM" --ckpt "$CKPT" --samplers "$specs" \
    --out-root "$OUT" --weights ema "${extra[@]}"
echo "PREDICT_RC=0"

# Slim every sampler directory of this arm (the copy to Atos carries only the slim tree).
python "$DT/scripts/fewstep_bigeval_20261015/slim_predictions.py" --src "$OUT/$ARM" \
    --dst "$W/slim/set$SET/$ARM" --skip-existing
echo "SLIM_RC=0"
