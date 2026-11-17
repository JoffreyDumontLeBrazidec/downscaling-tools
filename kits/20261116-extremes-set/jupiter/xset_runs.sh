# Extremes set XSET-1 (16 Nov 2026): the model rows of the global read on Jupiter. Source it:  source xset_runs.sh <ROW>
#   GL512_500k   db_global_512_300k step 500000, EMA-2000 (the reference row of the set)
#   GL512_450k   the same run at step 450000, EMA-2000 (the standard checkpoint since 17 Nov 2026)
#   GL512_237k   the same run at step 237000, EMA-2000 (only for the bit-identity test against the 4 Nov member read)
# A new arm = one more case line (run directory, config, code pin, step, EMA tag); nothing else changes.
CK=/e/data1/jureap-data/ecmwf/users/jdumont/outputs
DEVR=/e/home/jusers/dumontlebrazidec1/jupiter/dev
XK=${XSET_JWORK:-/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set}
XKIT=$XK/downscaling-tools/kits/20261116-extremes-set
XOUT=${XSET_JOUT:-$CK/xset1_20261116}
row=${1:?row}
unset RUN_DIR_ROW MODEL_LABEL_ROW; export SAMPLER_MODE=edm
case "$row" in
  GL512_500k) export CKPT_STEP=500000 ;;
  GL512_450k) export CKPT_STEP=450000 ;;
  GL512_237k) export CKPT_STEP=237000 ;;
  # regression baseline of the paper (17 Nov 2026, kits/20261117-baselines/reg): GL512 450k warm start, training_approach
  # deterministic (sigma 5e5), 30k steps on 4,000 km caps; read with ONE denoiser call (SAMPLER_MODE=det). Built from
  # GL512's config and code (same class and 10 outputs): only the weights differ.
  REG_GL512_30k) export CKPT_STEP=30000 SAMPLER_MODE=det MODEL_LABEL_ROW=reg_gl512_450k_det_ema2000 RUN_DIR_ROW=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261117-regression-baseline/chains/reg_gl512_450k_det ;;
  REG_GL512_20k) export CKPT_STEP=20000 SAMPLER_MODE=det MODEL_LABEL_ROW=reg_gl512_450k_det_ema2000 RUN_DIR_ROW=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261117-regression-baseline/chains/reg_gl512_450k_det ;;
  *) echo "FATAL unknown row '$row'"; return 1 ;;
esac
export RUN_DIR=${RUN_DIR_ROW:-$CK/checkpoint_nm50_global/db_global_512_300k} CFG_DIR=$DEVR/db_global_20261027/configs CFG_NAME=db_global_512_300k
export EMA_TAG=ema2000 MODEL_LABEL=${MODEL_LABEL_ROW:-db_global_512_300k_ema2000}
export EXPECT_CODE=db_global_20261027/code/anemoi-core CODE_ROOT=/e/data1/jureap-data/ecmwf/users/jdumont/db_global_20261027/code/anemoi-core
export CODE_SHA_FILE=$DEVR/db_global_20261027/CODE_PIN DATASETS_SHA_FILE=$DEVR/db_global_20261027/DATASETS_PIN
export ROW_LABEL=$row MEMBERS=${MEMBERS:-10}
export XSET_IN=$XK/stores/in XSET_TEMPLATE=$XOUT/rows/ENS9 XSET_HIST=$XOUT/hist OUTDIR=$XOUT/rows/$row
