# Shared settings of campaign T1d stage A (2026-10-07). Sourced by every job and submit script.
# Override any value by exporting it before the call (e.g. T1D_SANDBOX after `exp new` on another date).
export T1D_W="${T1D_W:-/home/ecm5702/work-t1d-20261007}"                 # real directory in $HOME, never a scratch link
export T1D_CODE="${T1D_CODE:-$T1D_W/code/downscaling-tools}"            # worktree of the pushed T1d branch
export T1D_S="$T1D_CODE/scripts/t1d_sampler_20261007"
export T1D_SANDBOX="${T1D_SANDBOX:-/home/ecm5702/hpcperm/sandbox/20261007-t1d-stagea}"  # `exp new t1d-stagea`
export T1D_CORE_SHA="27391c1597e8090b6e0da9a9ce79352eaa6c6530"         # anemoi-core: samplers patch on hres-lead 00472fb6f
export T1D_HOST="${T1D_HOST:-ac}"                                        # cluster of the PREDICTIONS: ac (A100, x86) or ag (GH200, aarch64)
export T1D_CERT_VENV="${T1D_CERT_VENV:-/home/ecm5702/dev/.ds-260612}"   # certified x86 venv: AC predictions and every CPU job (always AC)
export T1D_CERT_VENV_AG="${T1D_CERT_VENV_AG:-/home/ecm5702/dev/.ds-ag-260616}"  # certified arm venv: AG predictions (G1 reference, overlay base)
export T1D_INPUT_ROOT="${T1D_INPUT_ROOT:-/ec/res4/scratch/ecm5702/eval/o320_o1280/manual_731d203a_pristine_20260818/bundles_with_y}"
export T1D_E="${T1D_E:-/home/ecm5702/scratch/eval}"
export T1D_TAG="20261007"
export T1D_DATES="20230826,20230827,20230828,20230829,20230830"
export T1D_MEMBERS="1,2,3,4,5,6,7,8,9,10"
export T1D_STEPS="24,120"
export T1D_ONLY="tc,texture,wind_extremes,probabilistic,surface,shape"
# sbatch options of one prediction job on the chosen cluster (AC: the script's own header)
t1d_host_opts() { [[ "$T1D_HOST" == ag ]] && echo "--partition=gpu --qos=ng --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 --mem=120G" || true; }
# job-name infix: empty on AC (names unchanged), "ag_" on AG
t1d_jn() { [[ "$T1D_HOST" == ag ]] && echo ag_ || true; }
# gate G1 run root (runtime sandbox|certified); AG gates carry _ag so they never collide with an AC gate
t1d_g1_root() { echo "$T1D_E/o320_o1280_p12m_pw30_c0_g1_${1}$([[ "$T1D_HOST" == ag ]] && echo _ag)_$T1D_TAG"; }
# run root of an arm (second argument: the seed, 756 or 757)
t1d_root() { local a="$1" s="${2:-756}"; [[ "$s" == 757 ]] && echo "$T1D_E/o320_o1280_${a}_r757_tc100_$T1D_TAG" || echo "$T1D_E/o320_o1280_${a}_tc100_$T1D_TAG"; }
