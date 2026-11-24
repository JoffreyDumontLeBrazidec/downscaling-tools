# Month inference set MI-S25 (17 Nov 2026): the model rows of the global month read on Jupiter. Source it:  source mi_runs.sh <ROW>
#   GL512_450k   db_global_512_300k step 450000, EMA-2000 (the standard checkpoint since 17 Nov 2026, rule 2 / 48)
#   GL512_500k   the same run at step 500000 (only for the xcheck gate against the XSET-1 500k row)
# A new arm (e.g. the FS-242 freeze export) = one more case line (run directory, config, code pin, step, EMA tag).
CK=/e/data1/jureap-data/ecmwf/users/jdumont/outputs
DEVR=/e/home/jusers/dumontlebrazidec1/jupiter/dev
MK=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261117-month-inference
MKIT=$MK/downscaling-tools/kits/20261117-month-inference
MI_SET=${MI_SET:-s25}
MOUT=$CK/month_inference_$MI_SET
row=${1:?row}
MI_MODELS=${MI_MODELS:-/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set/downscaling-tools/kits/20261116-extremes-set/jupiter/models}
case "$row" in
  GL512_450k|GL512_500k)
    [[ $row == GL512_450k ]] && export CKPT_STEP=450000 || export CKPT_STEP=500000
    export RUN_DIR=$CK/checkpoint_nm50_global/db_global_512_300k CFG_DIR=$DEVR/db_global_20261027/configs CFG_NAME=db_global_512_300k
    export EMA_TAG=ema2000 MODEL_LABEL=db_global_512_300k_ema2000
    export EXPECT_CODE=db_global_20261027/code/anemoi-core CODE_ROOT=/e/data1/jureap-data/ecmwf/users/jdumont/db_global_20261027/code/anemoi-core
    export CODE_SHA_FILE=$DEVR/db_global_20261027/CODE_PIN DATASETS_SHA_FILE=$DEVR/db_global_20261027/DATASETS_PIN ;;
  *_[0-9]*k)
    # Any XSET-1 model key (21 Nov 2026): <KEY>_<N>k reads the XSET-1 model file models/<KEY>.env (read-only, from the
    # live XSET checkout, so FS242 / FS242C resolve exactly as their XSET-1 rows do). EMA_FILE / RAW_FILE given in the
    # environment win over the RUN_DIR globs (needed for steps without a raw by-step copy under RUN_DIR).
    _key=${row%_*}; _n=${row##*_}; _n=${_n%k}
    [[ $_n =~ ^[0-9]+$ && -f $MI_MODELS/$_key.env ]] || { echo "FATAL no model file $MI_MODELS/$_key.env for row '$row'"; return 1; }
    set -a; source $MI_MODELS/$_key.env; set +a
    export CKPT_STEP=$(( _n * 1000 )) EMA_TAG=${EMA_TAG:-ema2000}
    export MODEL_LABEL=${MODEL_NAME}_${EMA_TAG} ;;
  *) echo "FATAL unknown row '$row'"; return 1 ;;
esac
export ROW_LABEL=$row MEMBERS=${MEMBERS:-10}
export MI_IN=/e/data1/jureap-data/ecmwf/users/jdumont/datasets/mi_${MI_SET}_eefo_o320_pf10_l012-240/in
export OUTDIR=$MOUT/rows/$row
