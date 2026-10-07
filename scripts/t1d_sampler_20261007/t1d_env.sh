# Shared settings of campaign T1d stage A (2026-10-07). Sourced by every job and submit script.
# Override a value by exporting it in the SUBMIT shell (e.g. T1D_HOST=ag). Atos sets SBATCH_EXPORT=NONE, so the job never
# inherits the submit shell's environment: every submit script passes the settings below explicitly with
# `$(t1d_export)` = --export=ALL,T1D_HOST=...,T1D_CODE=...,... on each sbatch line. Inside a job, the values that arrive
# through that list win over the defaults here; nothing a job needs may rely on anything else reaching it.
export T1D_W="${T1D_W:-/home/ecm5702/work-t1d-20261007}"                 # real directory in $HOME, never a scratch link
export T1D_CODE="${T1D_CODE:-$T1D_W/code/downscaling-tools}"            # worktree of the pushed T1d branch
export T1D_S="$T1D_CODE/scripts/t1d_sampler_20261007"
export T1D_SANDBOX="${T1D_SANDBOX:-/home/ecm5702/hpcperm/sandbox/20261007-t1d-stagea}"  # `exp new t1d-stagea`
export T1D_CORE_SHA="27391c1597e8090b6e0da9a9ce79352eaa6c6530"         # anemoi-core: samplers patch on hres-lead 00472fb6f
export T1D_HOST="${T1D_HOST:-ac}"                                        # cluster of the PREDICTIONS: ac (A100, x86) or ag (GH200, aarch64)
export T1D_CERT_VENV="${T1D_CERT_VENV:-/home/ecm5702/dev/.ds-260612}"   # certified x86 venv: AC predictions and every CPU job (always AC)
export T1D_CERT_VENV_AG="${T1D_CERT_VENV_AG:-/home/ecm5702/dev/.ds-ag-260616}"  # certified arm venv: the AG G1 reference
export T1D_RUNTIME="${T1D_RUNTIME:-venv}"   # how a "sandbox" job activates: venv (default) = the sandbox's own uv venv
                                            # .venv-$(uname -m) through its guarded activate.sh, on AC (.venv-x86_64) and AG
                                            # (.venv-aarch64, owner's option 1 of 2026-10-07); overlay = opt-in, AG only: the
                                            # certified arm venv + PYTHONPATH onto the sandbox code + import guard
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

# --export list of every sbatch call: the campaign settings plus any extra VAR names given (only those that are set).
# Values must not contain commas (sbatch splits --export on them): refused.
t1d_export() {
  local v out="--export=ALL" val
  for v in T1D_HOST T1D_W T1D_CODE T1D_E T1D_SANDBOX T1D_CERT_VENV T1D_CERT_VENV_AG T1D_INPUT_ROOT T1D_RUNTIME "$@"; do
    [[ -n "${!v+x}" ]] || continue
    val="${!v}"
    [[ "$val" == *,* ]] && { echo "t1d_export: $v contains a comma: $val" >&2; return 1; }
    out="$out,$v=$val"
  done
  echo "$out"
}
# HEAD commit of a git worktree. Plain `git` first; when it fails (on Atos the sandbox's git commands are known to work
# only from ag-login), read .git -> gitdir -> HEAD -> loose ref or packed-refs directly.
t1d_git_head() {
  local d="$1" g h c ref
  git -C "$d" rev-parse HEAD 2>/dev/null && return 0
  if [[ -f "$d/.git" ]]; then g=$(sed -n 's/^gitdir: //p' "$d/.git"); [[ "$g" == /* ]] || g="$d/$g"; else g="$d/.git"; fi
  h=$(cat "$g/HEAD" 2>/dev/null) || return 1
  [[ "$h" != ref:* ]] && { echo "$h"; return 0; }
  ref=${h#ref: }; c="$g"; [[ -f "$g/commondir" ]] && { c=$(cat "$g/commondir"); [[ "$c" == /* ]] || c="$g/$c"; }
  [[ -f "$c/$ref" ]] && { cat "$c/$ref"; return 0; }
  awk -v r="$ref" '$2 == r {print $1; f=1} END {exit !f}' "$c/packed-refs" 2>/dev/null
}
# Activate the "sandbox" runtime inside a job and prove it (sets CORE_DIR). Both clusters by default use the sandbox's
# own uv venv for this node's architecture through its guarded activate.sh (which refuses unless anemoi.models/training/
# graphs/inference/datasets resolve inside the sandbox); this adds: the venv is .venv-$(uname -m) of THIS sandbox, the
# samplers module comes from it and has the custom schedule, and anemoi-core HEAD is T1D_CORE_SHA.
t1d_activate_sandbox() {
  local arch; arch=$(uname -m)
  case "$T1D_RUNTIME" in
    venv)
      set +u; source "$T1D_SANDBOX/activate.sh" || { set -u; echo "FATAL sandbox activation failed ($T1D_SANDBOX, .venv-$arch)"; return 3; }; set -u
      [[ "$(readlink -f "${VIRTUAL_ENV:-none}")" == "$(readlink -f "$T1D_SANDBOX/.venv-$arch")" ]] \
        || { echo "FATAL active venv ${VIRTUAL_ENV:-none} is not $T1D_SANDBOX/.venv-$arch"; return 3; } ;;
    overlay)
      [[ "$arch" == aarch64 ]] || { echo "FATAL T1D_RUNTIME=overlay is the AG opt-in only"; return 3; }
      set +u; unset PYTHONPATH; source "$T1D_CERT_VENV_AG/bin/activate" || { set -u; return 3; }; set -u
      local c=$T1D_SANDBOX/code
      export PYTHONPATH="$c/anemoi-core/training/src:$c/anemoi-core/models/src:$c/anemoi-core/graphs/src:$c/anemoi-inference/src" ;;
    *) echo "FATAL T1D_RUNTIME=$T1D_RUNTIME (venv|overlay)"; return 3 ;;
  esac
  python - "$T1D_SANDBOX" <<'PYGUARD' || { echo "FATAL GUARD: anemoi does not load from the sandbox"; return 4; }
import importlib, os, sys
root = os.path.realpath(sys.argv[1]); bad = []
for name in ("anemoi.models", "anemoi.training", "anemoi.graphs", "anemoi.inference", "anemoi.models.samplers.diffusion_samplers"):
    p = os.path.realpath(importlib.import_module(name).__file__)
    (bad.append if not p.startswith(root + os.sep) else print)(f"  GUARD {name} -> {p}")
import anemoi.models.samplers.diffusion_samplers as d
if "custom" not in d.NOISE_SCHEDULERS: bad.append("  GUARD samplers module has no custom schedule (not 27391c1)")
import torch; print(f"  GUARD python {sys.version.split()[0]} torch {torch.__version__} venv {sys.prefix}")
if bad: print("\n".join(bad)); sys.exit(1)
PYGUARD
  CORE_DIR=$T1D_SANDBOX/code/anemoi-core
  local head; head=$(t1d_git_head "$CORE_DIR")
  [[ "$head" == "$T1D_CORE_SHA" ]] || { echo "FATAL sandbox anemoi-core HEAD '${head:-unreadable}' is not $T1D_CORE_SHA"; return 3; }
  echo "  GUARD anemoi-core HEAD $head"
}
