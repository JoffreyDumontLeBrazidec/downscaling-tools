#!/bin/bash
# End-to-end CPU test of train_cd.py (the example train.py + train_cd_v2.2.2.patch) on a synthetic export (cd_selftest.py
# build: 32 x 64 box, 38 inputs, 8 invariants, 5 time inputs), tiny U-Nets, fp32, one process: configs from
# make_cd_configs.py (--reg-crop 16 --ema --time-inputs ...), then (1) regression on 16 x 16 crops, checkpoint + EMA at 8
# samples, full-box validation; (2) the same run resumed to 16 samples (model, Adam, EMA reloaded); (3) patched
# diffusion (32 x 32 patches: the example needs square multiples of 32) with the step-(2) regression checkpoint, EMA, to 4 samples; (4) diffusion resumed to 8.
# Prints E2E_OK when every step shows its markers. Needs the venv (cd_env_mn5.sh) and train_cd.py (cd_apply_train_patch.sh).
#   bash cd_cpu_e2e.sh WORKDIR        (a new, empty folder)
set -uo pipefail
W=${1:?WORKDIR}; [[ -e $W ]] && { echo "$W exists: give a new folder"; exit 1; }
mkdir -p $W; W=$(cd $W && pwd)
K=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" HYDRA_FULL_ERROR=1 GIT_PYTHON_REFRESH=quiet
python - "$W" "$K" <<'EOF' || exit 1
import sys; sys.path.insert(0, sys.argv[2])
import cd_selftest as t, yaml
d = t.build(sys.argv[1])
yaml.safe_dump({"dataloader": {"training": {"datasets": {"in_lres": {"drop": []}}}}}, open(f"{sys.argv[1]}/gl.yaml", "w"))
print("synthetic export", d)
EOF
python $K/make_cd_configs.py --gl512-config $W/gl.yaml --data $W/syn.zarr --repo $CD_REPO --dataset-py $K/cd_dataset.py \
  --run-dir $W/run --out-conf-dir $W/conf --name-suffix cpu --time-inputs --val-spread --val-rows 4 --diff-lr-rampup 100 --diff-lr-decay-rate 50 \
  --reg-crop 16 --ema --ema-halflife-nimg 50 --reg-bpg 2 --reg-total-batch 2 --diff-bpg 2 --diff-total-batch 2 \
  --reg-ckpt-level 0 --diff-ckpt-level 0 --diff-max-patch 2 --dataloader-workers 1 > $W/configs.log 2>&1 || { cat $W/configs.log; exit 1; }
TINY="model.model_args.model_channels=16 model.model_args.channel_mult=[1,2] model.model_args.attn_resolutions=[] \
 training.perf.fp_optimizations=fp32 training.io.validation_steps=1"   # wandb: the configs' offline mode, as the chain
cd $CD_REPO/examples/weather/corrdiff
run() {  # mode dir log duration extra...
  local m=$1 d=$2 log=$3 n=$4; shift 4
  python train_cd.py --config-dir=$W/conf --config-name=config_training_e2v4cpu_$m.yaml training.io.checkpoint_dir=$d $TINY \
    training.hp.training_duration=$n training.io.print_progress_freq=2 training.io.save_checkpoint_freq=$n \
    training.io.validation_freq=$n "$@" > $log 2>&1
  echo "$m to $n samples: rc=$? $(grep -c sec_per_sample $log) ticks, $(grep -o 'Training Completed' $log | head -1)"
}
ok=1
chk() { grep -q "$2" $1 && echo "  [ok] $3" || { echo "  [MISSING] $3 ($2) in $1"; ok=0; }; }
R=$W/run/reg; run regression $R $W/reg1.log 8
chk $W/reg1.log "Regression on random 16x16 crops" "crop mode on"
chk $W/reg1.log "EMA initialised" "EMA initialised"
chk $W/reg1.log "EMA saved" "EMA saved"
chk $W/reg1.log "crop visits written" "visit map written"
chk $W/reg1.log "Training Completed" "regression part 1 completed"
run regression $R $W/reg2.log 16
chk $W/reg2.log "Loaded model state dictionary" "resume: model"
chk $W/reg2.log "Loaded optimizer state dictionary" "resume: Adam"
chk $W/reg2.log "EMA loaded from" "resume: EMA"
chk $W/reg2.log "Training Completed" "regression part 2 completed"
ls $R/checkpoints_regression/ $R/checkpoints_regression_ema/
export CD_REG_CKPT=$(ls $R/checkpoints_regression/*.mdlus | sort -V | tail -1)
D=$W/run/diff; P="training.hp.patch_shape_x=32 training.hp.patch_shape_y=32 training.hp.patch_num=2"
run diffusion $D $W/diff1.log 4 $P
chk $W/diff1.log "Loaded the pre-trained regression model" "diffusion: regression loaded"
chk $W/diff1.log "Patch-based training enabled" "diffusion: patches"
chk $W/diff1.log "EMA initialised" "diffusion: EMA"
chk $W/diff1.log "Training Completed" "diffusion part 1 completed"
run diffusion $D $W/diff2.log 8 $P
chk $W/diff2.log "EMA loaded from" "diffusion resume: EMA"
chk $W/diff2.log "Training Completed" "diffusion part 2 completed"
ls $D/checkpoints_patched_diffusion/ $D/checkpoints_patched_diffusion_ema/
python - $R/crop_visits_rank0.npz <<'EOF' || ok=0
import sys, numpy as np
z = np.load(sys.argv[1]); print("visit map:", {k: z[k].shape for k in z.files}, "crops", int(z["n_crops"]), "visits max", int(z["visits"].max()))
sys.exit(0 if int(z["n_crops"]) == 16 else 1)   # 8 crops per part, the resumed part continues the histogram
EOF
(( ok == 1 )) && echo E2E_OK || echo E2E_FAIL
