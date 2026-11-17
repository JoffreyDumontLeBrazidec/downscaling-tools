# Extremes set XSET-1 (16 Nov 2026): the model rows of the global read on Jupiter. Source it:  source xset_runs.sh <ROW>
#   GL512_500k   db_global_512_300k step 500000, EMA-2000 (the reference row of the set)
#   GL512_450k   the same run at step 450000, EMA-2000 (the standard checkpoint since 17 Nov 2026)
#   GL512_237k   the same run at step 237000, EMA-2000 (only for the bit-identity test against the 4 Nov member read)
# A new arm = one more case line (run directory, config, code pin, step, EMA tag); nothing else changes.
CK=/e/data1/jureap-data/ecmwf/users/jdumont/outputs
DEVR=/e/home/jusers/dumontlebrazidec1/jupiter/dev
XK=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set
XKIT=$XK/downscaling-tools/kits/20261116-extremes-set
XOUT=$CK/xset1_20261116
row=${1:?row}
case "$row" in
  GL512_500k) export CKPT_STEP=500000 ;;
  GL512_450k) export CKPT_STEP=450000 ;;
  GL512_237k) export CKPT_STEP=237000 ;;
  *) echo "FATAL unknown row '$row'"; return 1 ;;
esac
export RUN_DIR=$CK/checkpoint_nm50_global/db_global_512_300k CFG_DIR=$DEVR/db_global_20261027/configs CFG_NAME=db_global_512_300k
export EMA_TAG=ema2000 MODEL_LABEL=db_global_512_300k_ema2000
export EXPECT_CODE=db_global_20261027/code/anemoi-core CODE_ROOT=/e/data1/jureap-data/ecmwf/users/jdumont/db_global_20261027/code/anemoi-core
export CODE_SHA_FILE=$DEVR/db_global_20261027/CODE_PIN DATASETS_SHA_FILE=$DEVR/db_global_20261027/DATASETS_PIN
export ROW_LABEL=$row MEMBERS=${MEMBERS:-10}
export XSET_IN=$XK/stores/in XSET_TEMPLATE=$XOUT/rows/ENS9 XSET_HIST=$XOUT/hist OUTDIR=$XOUT/rows/$row
