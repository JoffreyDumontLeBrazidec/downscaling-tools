#!/bin/bash --login
# ============================================================================
# Single SLURM wrapper for ALL interp tools.
#
# USAGE:
#   sbatch interp/slurm/run.sbatch <TOOL> <CKPT_PATH> <CKPT_ID> [extra args...]
#
#   TOOL: tier1 | permutation | ablation | activations | patching | ig | report
#     tier1 = permutation + ablation + activations in sequence.
#   Extra tool args: ECMWF's ecsbatch swallows --flags after the script, and
#   sbatch --export comma-splits values, so extras are passed as DASH-LESS
#   key[=value] positionals, converted to --key value here:
#     sbatch interp/slurm/run.sbatch ig <CKPT> <ID> \
#         functionals=global_mean,eye,tail,spectral tail-percentile=99
#   Keys taking a space-separated list (sigmas, modes, dates, members, steps,
#   run-dirs) accept comma-separated values: sigmas=0.1,5.0,80.0.
#   Repeatable single-value flags repeat the key: region=amazon region=tropics.
#
# Output convention: /home/ecm5702/perm/interp/<CKPT_ID>/<tool_subdir>/
#
# Environment: defaults to GA100 + env_ref.sh (.ds-dyn-dev + overlays).
# For GH200 (aarch64, .ds-ag):
#   sbatch --gres=gpu:gh200:1 --export=ALL,INTERP_ENV=env_gh200 \
#       interp/slurm/run.sbatch ig <CKPT> <ID>
#
# Examples:
#   sbatch interp/slurm/run.sbatch tier1 \
#     /home/ecm5702/scratch/aifs/checkpoint/cfec83a3.../anemoi-...step_200000.ckpt \
#     cfec_200k
#   sbatch interp/slurm/run.sbatch ig $CKPT cfec_200k --functionals global_mean,eye,tail,spectral
# ============================================================================
#SBATCH --job-name=interp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --gres=gpu:ga100:1
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/interp_%j.out
#SBATCH --error=/home/ecm5702/dev/jobscripts/outputs/interp_%j.err
set -euxo pipefail

TOOL="${1:?Usage: run.sbatch <TOOL> <CKPT_PATH> <CKPT_ID> [extra...]}"
CKPT_PATH="${2:?CKPT_PATH}"
CKPT_ID="${3:?CKPT_ID}"
shift 3
# Convert dash-less key[=value] extras into --key value tool flags.
# Comma-split values only for keys that argparse declares as nargs lists.
NARGS_KEYS=" sigmas modes dates members steps run-dirs "
EXTRA=()
for kv in "$@"; do
  if [[ "$kv" == *=* ]]; then
    k="${kv%%=*}"; v="${kv#*=}"
    if [[ "$NARGS_KEYS" == *" $k "* ]]; then
      EXTRA+=("--$k")
      IFS=, read -ra parts <<< "$v"
      EXTRA+=("${parts[@]}")
    else
      # --key=value as ONE token so values with a leading dash
      # (auto-window=-10,...) survive argparse.
      EXTRA+=("--$k=$v")
    fi
  else
    EXTRA+=("--$kv")
  fi
done
set -- ${EXTRA[@]+"${EXTRA[@]}"}

OUT_ROOT="/home/ecm5702/perm/interp/${CKPT_ID}"
EVENT="${INTERP_EVENT:-franklin_o96_o320_m4}"

source "/home/ecm5702/dev/downscaling-tools/interp/slurm/${INTERP_ENV:-env_ref}.sh"
cd /home/ecm5702/dev/downscaling-tools

run_tool () {
    local tool="$1" subdir="$2"; shift 2
    python -m interp "${tool}" --checkpoint "${CKPT_PATH}" \
        --output-dir "${OUT_ROOT}/${subdir}" "$@"
}

case "${TOOL}" in
  tier1)
    run_tool permutation feature_permutation --event "${EVENT}" "$@"
    run_tool ablation conditioning_ablation --event "${EVENT}" "$@"
    run_tool activations activations --event "${EVENT}" "$@"
    ;;
  permutation)
    run_tool permutation feature_permutation --event "${EVENT}" "$@"
    ;;
  ablation)
    run_tool ablation conditioning_ablation --event "${EVENT}" "$@"
    ;;
  activations)
    run_tool activations activations --event "${EVENT}" "$@"
    ;;
  patching)
    # patching uses a single peak bundle (default from EVENTS), not --event
    run_tool patching activation_patching "$@"
    ;;
  ig)
    # IG needs batch 1: single-member event
    run_tool ig integrated_gradients --event franklin_o96_o320 "$@"
    ;;
  report)
    python -m interp.viz --run-dir "${OUT_ROOT}" "$@"
    ;;
  *)
    echo "unknown tool: ${TOOL}" >&2; exit 2
    ;;
esac

echo "${TOOL} complete for ${CKPT_ID} -> ${OUT_ROOT}"
