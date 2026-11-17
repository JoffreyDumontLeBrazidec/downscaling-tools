# Extremes set XSET-1 (16 Nov 2026): the model rows of the global read on Jupiter. Source it:  source xset_runs.sh <ROW>
# A row = one export of one model. Two kinds:
#   defined rows  XOUT/rowdefs/<ROW>.env, written once by xset_define_row.sh (model file of jupiter/models/, step, EMA tag,
#                 the two export files with the sha256 of the weights). This is what xset_row.sh uses for every new row.
#   first rows    GL512_500k, GL512_450k, GL512_237k (db_global_512_300k, EMA-2000) and the regression baseline rows
#                 REG_GL512_* of the baselines thread: kept as the case lines they were drawn with, so their jobs and
#                 files stay reproducible to the letter.
CK=/e/data1/jureap-data/ecmwf/users/jdumont/outputs
DEVR=/e/home/jusers/dumontlebrazidec1/jupiter/dev
XK=${XSET_JWORK:-/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set}
XKIT=$XK/downscaling-tools/kits/20261116-extremes-set
XOUT=${XSET_JOUT:-$CK/xset1_20261116}
row=${1:?row}
unset RUN_DIR_ROW MODEL_LABEL_ROW; export SAMPLER_MODE=edm
unset VENV DATASETS_REPO EXTRA_OVERRIDES ANCHOR XSET_SCHED_JSON XSET_SAMPLER_JSON EMA_SHA256
if [[ -f $XOUT/rowdefs/$row.env ]]; then
  source $XOUT/rowdefs/$row.env                    # MODEL CKPT_STEP EMA_TAG EMA_FILE RAW_FILE [XSET_SCHED_JSON XSET_SAMPLER_JSON]
  [[ -f $XKIT/jupiter/models/$MODEL.env ]] || { echo "FATAL row '$row': no model file jupiter/models/$MODEL.env"; return 1; }
  source $XKIT/jupiter/models/$MODEL.env
  for v in MODEL_NAME RUN_DIR CFG_DIR CFG_NAME EXPECT_CODE CODE_ROOT CODE_SHA_FILE DATASETS_SHA_FILE CKPT_STEP EMA_TAG EMA_FILE RAW_FILE; do
    [[ -n "${!v:-}" ]] || { echo "FATAL row '$row': $v is empty (model file $MODEL.env or row definition)"; return 1; }
  done
  export CKPT_STEP RUN_DIR CFG_DIR CFG_NAME EMA_TAG EMA_FILE RAW_FILE EXPECT_CODE CODE_ROOT CODE_SHA_FILE DATASETS_SHA_FILE
  export MODEL_LABEL=${MODEL_NAME}_${EMA_TAG} XSET_MODEL=$MODEL
  for v in VENV DATASETS_REPO EXTRA_OVERRIDES ANCHOR XSET_SCHED_JSON XSET_SAMPLER_JSON; do
    if [[ -n "${!v:-}" ]]; then export $v; fi
  done
else
  case "$row" in
    GL512_500k) export CKPT_STEP=500000 ;;
    GL512_450k) export CKPT_STEP=450000 ;;
    GL512_237k) export CKPT_STEP=237000 ;;
    # regression baseline of the paper (17 Nov 2026, kits/20261117-baselines/reg): GL512 450k warm start, training_approach
    # deterministic (sigma 5e5), 30k steps on 4,000 km caps; read with ONE denoiser call (SAMPLER_MODE=det). Built from
    # GL512's config and code (same class and 10 outputs): only the weights differ.
    REG_GL512_30k) export CKPT_STEP=30000 SAMPLER_MODE=det MODEL_LABEL_ROW=reg_gl512_450k_det_ema2000 RUN_DIR_ROW=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261117-regression-baseline/chains/reg_gl512_450k_det ;;
    REG_GL512_20k) export CKPT_STEP=20000 SAMPLER_MODE=det MODEL_LABEL_ROW=reg_gl512_450k_det_ema2000 RUN_DIR_ROW=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261117-regression-baseline/chains/reg_gl512_450k_det ;;
    *) echo "FATAL unknown row '$row': no $XOUT/rowdefs/$row.env (define it with xset_define_row.sh or xset_row.sh)"; return 1 ;;
  esac
  unset EMA_FILE RAW_FILE
  export RUN_DIR=${RUN_DIR_ROW:-$CK/checkpoint_nm50_global/db_global_512_300k} CFG_DIR=$DEVR/db_global_20261027/configs CFG_NAME=db_global_512_300k
  export EMA_TAG=ema2000 MODEL_LABEL=${MODEL_LABEL_ROW:-db_global_512_300k_ema2000} XSET_MODEL=GL512
  export EXPECT_CODE=db_global_20261027/code/anemoi-core CODE_ROOT=/e/data1/jureap-data/ecmwf/users/jdumont/db_global_20261027/code/anemoi-core
  export CODE_SHA_FILE=$DEVR/db_global_20261027/CODE_PIN DATASETS_SHA_FILE=$DEVR/db_global_20261027/DATASETS_PIN
fi
export ROW_LABEL=$row MEMBERS=${MEMBERS:-10}
export XSET_IN=$XK/stores/in XSET_TEMPLATE=$XOUT/rows/ENS9 XSET_HIST=$XOUT/hist OUTDIR=$XOUT/rows/$row
