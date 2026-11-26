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
unset SAMPLER_MODE MI_DET_SIGMA XSET_SCHED_JSON XSET_SAMPLER_JSON
MI_MODELS=${MI_MODELS:-/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set/downscaling-tools/kits/20261116-extremes-set/jupiter/models}
case "$row" in
  GL512_450k|GL512_500k)
    [[ $row == GL512_450k ]] && export CKPT_STEP=450000 || export CKPT_STEP=500000
    export RUN_DIR=$CK/checkpoint_nm50_global/db_global_512_300k CFG_DIR=$DEVR/db_global_20261027/configs CFG_NAME=db_global_512_300k
    export EMA_TAG=ema2000 MODEL_LABEL=db_global_512_300k_ema2000
    export EXPECT_CODE=db_global_20261027/code/anemoi-core CODE_ROOT=/e/data1/jureap-data/ecmwf/users/jdumont/db_global_20261027/code/anemoi-core
    export CODE_SHA_FILE=$DEVR/db_global_20261027/CODE_PIN DATASETS_SHA_FILE=$DEVR/db_global_20261027/DATASETS_PIN ;;
  GL512D1E5_450k|GL512D1E3_450k|GL512D1E2_450k)
    # One-call rows of the standard checkpoint (26 Nov 2026, thread "Add a one-call deterministic baseline to the paper"):
    # GL512 450k EMA-2000 weights unchanged, ONE denoiser call at sigma 1e5 / 1e3 / 1e2 (SAMPLER_MODE=det, MI_DET_SIGMA),
    # one draw per input member, rule-31 seeds. 1e5 = the standard sampler's first sigma; 1e3 / 1e2 = better-trained levels
    # (NM50: P(sigma > 1e5) = 6.5e-6, > 1e3 = 4.5e-3, > 1e2 = 3.7e-2).
    _s=${row#GL512D}; _s=${_s%%_*}
    export CKPT_STEP=450000 SAMPLER_MODE=det MI_DET_SIGMA=$(python3 -c "print(float('${_s}'))")
    export RUN_DIR=$CK/checkpoint_nm50_global/db_global_512_300k CFG_DIR=$DEVR/db_global_20261027/configs CFG_NAME=db_global_512_300k
    export EMA_TAG=ema2000 MODEL_LABEL=db_global_512_300k_ema2000_det${_s}
    export EXPECT_CODE=db_global_20261027/code/anemoi-core CODE_ROOT=/e/data1/jureap-data/ecmwf/users/jdumont/db_global_20261027/code/anemoi-core
    export CODE_SHA_FILE=$DEVR/db_global_20261027/CODE_PIN DATASETS_SHA_FILE=$DEVR/db_global_20261027/DATASETS_PIN ;;
  REG_GL512_30k)
    # Regression baseline (kits/20261117-baselines/reg; XSET-1 row of the same name in xset_runs.sh, to the letter): GL512
    # 450k warm start trained deterministic at sigma 5e5 for 30k steps, read with ONE call at 5e5. Same config and code.
    export CKPT_STEP=30000 SAMPLER_MODE=det MI_DET_SIGMA=500000
    export RUN_DIR=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261117-regression-baseline/chains/reg_gl512_450k_det_r2
    export CFG_DIR=$DEVR/db_global_20261027/configs CFG_NAME=db_global_512_300k EMA_TAG=ema2000 MODEL_LABEL=reg_gl512_450k_det_ema2000
    export EXPECT_CODE=db_global_20261027/code/anemoi-core CODE_ROOT=/e/data1/jureap-data/ecmwf/users/jdumont/db_global_20261027/code/anemoi-core
    export CODE_SHA_FILE=$DEVR/db_global_20261027/CODE_PIN DATASETS_SHA_FILE=$DEVR/db_global_20261027/DATASETS_PIN ;;
  GLAJ*K_J1|GLAJ*K_J2|GLAJ*K_J4|GLAJ*K_J10|GLAJ*K_G*)
    # MeanFlow rows (26 Nov 2026, thread "Score MeanFlow on Sep 2024 for the paper"): FG-199 glmf_alljump_50k at step n x 1000,
    # EMA-2000 export, drawn by SAMPLER_MODE=meanflow = the sampler route of the XSET-1 MeanFlow rows (kit
    # agent-work/20261110-meanflow-extremes/kit/xset: xset_runs_mf.sh sources kit/gl/runs_gl.sh with GLAJ<n>K_G<GRID>_ms_L024,
    # this case does the same, read-only). GLAJ<n>K_J<k> = k jumps from sigma 1e5, the grids of the FG-199 ladder (J1, J2S, JX4,
    # JX10 of runs_gl.sh): J1 = (1e5, 0); J2 = J2S (1e5, 80, 0); J4 = JX4 (1e5, 1000, 80, 0.97561, 0); J10 = JX10 (1e5, 1e4, 1000,
    # 283, 80, 4.65116, 1.92771, 0.97561, 0.490798, 0.197044, 0) = the XSET-1 GJX10 rows. GLAJ<n>K_G<GRID> = an XSET-1 row name
    # itself (the xcheck gate against outputs/xset1_20261116/rows/<that name>).
    [[ "$row" =~ ^GLAJ([0-9]+)K_(J1|J2|J4|J10|G([A-Z0-9]+))$ ]] || { echo "FATAL bad MeanFlow row '$row'"; return 1; }
    _mirow=$row; _n=${BASH_REMATCH[1]}
    case ${BASH_REMATCH[2]} in J1) _g=J1 ;; J2) _g=J2S ;; J4) _g=JX4 ;; J10) _g=JX10 ;; *) _g=${BASH_REMATCH[3]} ;; esac
    _mfk=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261110-meanflow-extremes/kit
    source $_mfk/gl/runs_gl.sh "GLAJ${_n}K_G${_g}_ms_L024" || { row=$_mirow; echo "FATAL runs_gl.sh refused GLAJ${_n}K_G${_g}_ms_L024"; return 1; }
    row=$_mirow   # runs_gl.sh sets its own row
    [[ "$KIND" == flowmap && "$CFG_NAME" == glmf_alljump_50k && "$EXPECT_CLASS" == AnemoiD2ModelMeanFlowX && "$EMA_TAG" == ema2000 ]] || {
      echo "FATAL $row: runs_gl.sh gave kind=$KIND cfg=$CFG_NAME class=$EXPECT_CLASS ema=$EMA_TAG"; return 1; }
    (( CKPT_STEP > 0 )) || { echo "FATAL $row: step 0 (the warm file) is not a month row"; return 1; }
    unset CASES PDATES OUTDIR NFILES V LEAD ARM MEMBER_MODE SEED_RULE PART MEMBERS MEMBER_IN_STORE ROW
    export SAMPLER_MODE=meanflow MF_GRID=$_g EXTRA_OVERRIDES="$CFG_EXTRA_OVERRIDES" RAW_FILE=$WARM_FILE
    export FM_SCHED_JSON FM_SAMPLER_JSON EXPECT_CALLS EXPECT_CLASS WARM_SHA ;;
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
