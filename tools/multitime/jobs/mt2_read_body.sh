#!/bin/bash
# mt2_e2_20k read kit (2026-10-20), body of one prediction job: one hindcast date, one sampler stride, N members of the
# multi-time checkpoint on ONE GPU, then the conversion of the trajectories into eval.cli prediction files.
#   1. (MT_TRAIN_CKPT set and MT_CKPT missing) make_ema_inference.py writes the EMA copy MT_CKPT of the inference file;
#   2. predict_sequence.py --config MT_CONFIG --hdate MT_HDATE --stride MT_STRIDE (lane 30-step piecewise sampler,
#      churn off, fp32, member seeds MT_SEED + id - 1) -> MT_OUT/npz/<hdate>/mt2_<hdate>_m<id>.npz (+ _truth.npz);
#   3. npz_to_evalcli.py with the kit template folder MT_TEMPLATE -> MT_OUT/predictions/predictions_<hdate>_stepSSS.nc
#      (MT_CONVERT=check: read-only checks against the template, nothing written, and on success the smoke's pass marker
#      <dirname MT_OUT>/SMOKE_OK; MT_CONVERT=no: skipped). MT_GATE: a file that must exist before anything runs.
# Variables come from the generated job file (Atos sets SBATCH_EXPORT=NONE). Docs: epics/fine-scale-o320-o1280/
# in-progress/20261016_multi_time_downscaler_e2_and_window_sampler.md (reads A and B, section 6).
set -eo pipefail
ulimit -c 0
: "${MT_OUT:?}" "${MT_CKPT:?}" "${MT_CONFIG:?}" "${MT_HDATE:?}" "${MT_STRIDE:?}" "${MT_TEMPLATE:?}" "${MT_LABEL:?}"
MT_MEMBERS=${MT_MEMBERS:-10}; MT_MEMBER_START=${MT_MEMBER_START:-1}; MT_SEED=${MT_SEED:-1234}; MT_CONVERT=${MT_CONVERT:-yes}
S=/home/ecm5702/scratch/sandbox/20261016-multitime-e2
PY=/home/ecm5702/hpcperm/sandbox/20260917-vera-spatial-downscaler/.venv-x86_64/bin/python
C=$S/code/anemoi-core; DT=$S/code/downscaling-tools; ADS=/home/ecm5702/dev/anemoi-datasets
export PYTHONPATH=$C/models/src:$C/training/src:$C/graphs/src:$ADS/src
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp OMP_NUM_THREADS=8 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HYDRA_FULL_ERROR=1
set -u
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "[mt2read] start label=$MT_LABEL hdate=$MT_HDATE stride=$MT_STRIDE members=$MT_MEMBER_START+$MT_MEMBERS seed=$MT_SEED" \
     "ckpt=$MT_CKPT host=$(hostname -s) anemoi-core=$(git -C $C rev-parse --short HEAD)" \
     "downscaling-tools=$(git -C $DT rev-parse --short HEAD) anemoi-datasets=$(git -C $ADS rev-parse --short HEAD)" \
     "ads-diff-md5=$(git -C $ADS diff | md5sum | cut -c1-8) $(date -u +%FT%TZ)"
t0=$(date +%s)
if [[ -n ${MT_GATE:-} && ! -f $MT_GATE ]]; then
  echo "[mt2read] GATE MISSING: $MT_GATE (the smoke did not pass); nothing run" >&2; exit 3
fi
if [[ ! -f $MT_CKPT ]]; then
  : "${MT_TRAIN_CKPT:?MT_CKPT missing and no MT_TRAIN_CKPT to build it from}"
  mkdir -p "$(dirname "$MT_CKPT")"
  $PY $DT/tools/multitime/make_ema_inference.py --train-ckpt "$MT_TRAIN_CKPT" \
      --inference "$(dirname "$MT_TRAIN_CKPT")/inference-$(basename "$MT_TRAIN_CKPT")" --out "$MT_CKPT.partial" < /dev/null
  mv "$MT_CKPT.partial" "$MT_CKPT"
fi
NPZ=$MT_OUT/npz/$MT_HDATE; mkdir -p "$NPZ" "$MT_OUT/predictions"
$PY $DT/tools/multitime/predict_sequence.py --ckpt "$MT_CKPT" --config "$MT_CONFIG" --hdate "$MT_HDATE" \
    --members "$MT_MEMBERS" --member-start "$MT_MEMBER_START" --seed "$MT_SEED" --stride "$MT_STRIDE" \
    --out "$NPZ/mt2_$MT_HDATE" < /dev/null
echo "[mt2read] predicted seconds=$(( $(date +%s) - t0 )) $(date -u +%FT%TZ)"
case "$MT_CONVERT" in
  yes)   $PY $DT/tools/multitime/npz_to_evalcli.py --npz-prefix "$NPZ/mt2_$MT_HDATE" --hdate "$MT_HDATE" \
             --template-dir "$MT_TEMPLATE" --out-dir "$MT_OUT/predictions" --label "$MT_LABEL" --members "$MT_MEMBERS" ;;
  check) $PY $DT/tools/multitime/npz_to_evalcli.py --npz-prefix "$NPZ/mt2_$MT_HDATE" --hdate "$MT_HDATE" \
             --template-dir "$MT_TEMPLATE" --out-dir "$MT_OUT/predictions" --label "$MT_LABEL" --members "$MT_MEMBERS" \
             --check-only
         touch "$(dirname "$MT_OUT")/SMOKE_OK"; echo "[mt2read] smoke checks passed: wrote $(dirname "$MT_OUT")/SMOKE_OK" ;;
  no)    echo "[mt2read] conversion skipped" ;;
esac
echo "[mt2read] done label=$MT_LABEL hdate=$MT_HDATE stride=$MT_STRIDE seconds=$(( $(date +%s) - t0 )) $(date -u +%FT%TZ)"
