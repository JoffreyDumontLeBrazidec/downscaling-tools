# o2560 Europe-box read (2 Nov 2026): one row = one checkpoint (EMA-10000 inference file). Source with a row name.
# A100  crop chain db_o2560_crop_100k (AG), step 100,000, 234.1 GH200-h, code 4ad32838c (its training code, crop off at inference)
# B24   full-box chain db_o2560_e2_50k_v2 (Jupiter), step 24,000, 229.9 GH200-h (equal GPU-hours with A100), code 20ba15241
# B50   the same chain at step 50,000, 477.0 GH200-h (its last step), code 20ba15241
# B24X  smoke only: B24 weights on A's code 4ad32838c (crop off), to show the crop commits leave full-box inference unchanged
KIT=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read
OUT=/e/data1/jureap-data/ecmwf/users/jdumont/outputs/o2560_box_read_20261102
BR=/e/data1/jureap-data/ecmwf/users/jdumont/db_o2560_e2_50k_v2_20261030/ckpt_root/168c43551e3a414ead0f39a25db7cf22
B_CODE=/e/data1/jureap-data/ecmwf/users/jdumont/db_o2560_e2_50k_v2_20261030/code/anemoi-core
A_CODE=$KIT/code/anemoi-core-4ad32838c
row=${1:?row}
case "$row" in
  A100) export CKPT_FILE=$KIT/ckpt/A100k/inference-ema10000-anemoi-by_step-epoch_198-step_100000.ckpt CKPT_STEP=100000
        export RAW_FILE=$KIT/ckpt/A100k/inference-anemoi-by_step-epoch_198-step_100000.ckpt
        export CODE_ROOT=$A_CODE CODE_PIN_SHA=4ad32838c0cee47d41cef97924dd32a93a56c3e3 ;;
  B24|B24X) export CKPT_FILE=$BR/inference-ema10000-anemoi-by_step-epoch_036-step_024000.ckpt CKPT_STEP=24000
        export RAW_FILE=$BR/inference-anemoi-by_step-epoch_036-step_024000.ckpt
        export CODE_ROOT=$B_CODE CODE_PIN_SHA=20ba15241ca68606dd8613461042bee407108326
        [[ $row == B24X ]] && export CODE_ROOT=$A_CODE CODE_PIN_SHA=4ad32838c0cee47d41cef97924dd32a93a56c3e3 ;;
  B50)  export CKPT_FILE=$BR/inference-ema10000-anemoi-by_step-epoch_076-step_050000.ckpt CKPT_STEP=50000
        export RAW_FILE=$BR/inference-anemoi-by_step-epoch_076-step_050000.ckpt
        export CODE_ROOT=$B_CODE CODE_PIN_SHA=20ba15241ca68606dd8613461042bee407108326 ;;
  *) echo "FATAL unknown row '$row'"; return 1 ;;
esac
[[ -f $CKPT_FILE ]] || { echo "FATAL missing $CKPT_FILE"; return 1; }
export ROW=$row MEMBERS=${MEMBERS:-10}
export DATES="${DATES:-20241101 20241102 20241103 20241104 20241105 20241106 20241107 20241108 20241109 20241110 20241111 20241112}"
cases_for() { local L=$1 c="" d; for d in $DATES; do c="$c${c:+,}$d:$((10#$L))"; done; echo "$c"; }
