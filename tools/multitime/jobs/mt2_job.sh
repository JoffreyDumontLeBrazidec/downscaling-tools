#!/bin/bash
# ============================================================================
# mt2_e2_20k job body (2026-10-16): multi-in multi-out diffusion downscaler, two time slots 12 h apart,
# o320 -> o1280 on the Europe box E2, on Vera Gahlen's branch (fork anemoi-core claude/project-thread-n1zaq3,
# 4a81ecb43), warm-started from our donor 12dcefea through the transplanted and tiled checkpoint
# checkpoints/warm/stage_c_vera_k2_tiled_average.ckpt. Modelled on the do0 sandbox's jobs/dropout_chain_job.sh.
# Called by the thin jobs/*.sbatch files, which set:
#
#   MT_KIND      chain: one segment of the 20k chain (downscaling-tools tools/chain: the start mode is read from
#                       the chain root on disk; chain_next queues the successor)
#                smoke: one standalone short run from the warm checkpoint (no chain, no successor)
#   MT_TAG       run tag: outputs in $S/runs/$MT_TAG; chain root $S/checkpoints/$MT_TAG for chain runs
#   MT_END_STEP  training.max_steps of this job
#   MT_EXTRA     optional extra Hydra overrides (space separated, no spaces inside one override)
#
# DRY_RUN=1 bash <sbatch file>: resolve and print the start (checkpoint, end step, max_steps) and the override
# list, run every fail-fast check, train nothing, submit nothing.
# Environment: the Vera x86 venv's interpreter with PYTHONPATH on the sandbox code and the lane fork
# anemoi-datasets (never the venv's activate.sh, which unsets PYTHONPATH).
# ============================================================================
set -eo pipefail
S=/home/ecm5702/scratch/sandbox/20261016-multitime-e2
VENV=/home/ecm5702/hpcperm/sandbox/20260917-vera-spatial-downscaler/.venv-x86_64
PY=$VENV/bin/python
CODE=$S/code/anemoi-core
CODE_PIN=4a81ecb43
DSFORK=/home/ecm5702/dev/anemoi-datasets
CHAIN_TOOLS=$S/code/downscaling-tools/tools/chain
WARM=${MT_WARM:-$S/checkpoints/warm/stage_c_vera_k2_tiled_average.ckpt}
CONFIG_DIR=$S/configs
CONFIG_NAME=mt2_e2_20k
: "${MT_KIND:?}" "${MT_TAG:?}" "${MT_END_STEP:?}"

export PYTHONPATH=$CODE/models/src:$CODE/training/src:$CODE/graphs/src:$DSFORK/src
export PATH=$VENV/bin:$PATH
export ANEMOI_BASE_SEED=756
export HYDRA_FULL_ERROR=1
export NCCL_IB_TIMEOUT=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/tmp
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONFAULTHANDLER=1
JOBID="${SLURM_JOB_ID:-local}"
export TRITON_CACHE_DIR=/tmp/$USER/triton_$JOBID TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/inductor_$JOBID
ulimit -c 0

export OUTPUT=$S/runs/$MT_TAG
LOG_DIR=$S/logs
if [[ "$MT_KIND" == chain ]]; then CKPT_ROOT=$S/checkpoints/$MT_TAG; else CKPT_ROOT=$OUTPUT/checkpoints; fi
RUN_NAME=${MT_RUN_NAME:-$MT_TAG}

# ─── Fail-fast checks (refuse rather than guess) ────────────────────────────
test -f "$CONFIG_DIR/$CONFIG_NAME.yaml" || { echo "FATAL: no config $CONFIG_DIR/$CONFIG_NAME.yaml" >&2; exit 4; }
test -x "$PY" || { echo "FATAL: no interpreter $PY" >&2; exit 4; }
head_sha="$(git -C "$CODE" rev-parse HEAD)"
[[ "$head_sha" == "$CODE_PIN"* ]] || { echo "FATAL: $CODE HEAD $head_sha is not the pin $CODE_PIN" >&2; exit 46; }
[[ -z "$(git -C "$CODE" status --porcelain --untracked-files=no)" ]] || { echo "FATAL: uncommitted changes in $CODE" >&2; exit 44; }
for f in $(grep -E '^ +(residual_statistics|truncation|truncation_inv|graph|dataset|dataset_lres): /' "$CONFIG_DIR/$CONFIG_NAME.yaml" | awk '{print $2}' | sort -u); do
  test -e "$f" || { echo "FATAL: input missing: $f" >&2; exit 5; }
done
"$PY" - "$CODE" "$DSFORK" <<'PY' || { echo "FATAL: import resolution or venv pins are not as intended" >&2; exit 47; }
import importlib, importlib.metadata as md, os, sys
code, dsfork = (os.path.realpath(p) for p in sys.argv[1:3])
want = {"anemoi.models": code, "anemoi.training": code, "anemoi.graphs": code, "anemoi.datasets": dsfork}
bad = []
for name, root in want.items():
    f = os.path.realpath(importlib.import_module(name).__file__)
    print(f"[guard] {name} -> {f}")
    if not f.startswith(root + os.sep):
        bad.append(f"{name} resolves outside {root}")
mv = md.version("mlflow")
print(f"[guard] mlflow {mv}")
if mv != "3.11.1":
    bad.append(f"mlflow {mv} != 3.11.1")
if bad:
    print("\n".join(bad)); sys.exit(1)
PY

# ─── Start: chain state on disk (chain) or the warm checkpoint (smoke) ──────
warm_start=("system.input.warm_start=$WARM" "training.load_weights_only=True" "training.transfer_learning=False"
            "training.run_id=null" "training.fork_run_id=null")
if [[ "$MT_KIND" == chain ]]; then
  CHAIN_PY=$PY
  source "$CHAIN_TOOLS/chain_segment.sh"
  if [[ "${DRY_RUN:-0}" == "1" && ! -f "$CKPT_ROOT/CHAIN.json" ]]; then
    # Read-only preview before the chain is declared: a real segment refuses (exit 3) until
    # `chain_state.py init --root $CKPT_ROOT` has been run; after init the first segment starts fresh.
    echo "[dry-run] no $CKPT_ROOT/CHAIN.json: a real segment would refuse now; after chain_state.py init it starts fresh"
    CHAIN_ACTION=fresh; CHAIN_EXPECT_STEP=0; CHAIN_RUN_ID=""
  else
    chain_resolve "$CKPT_ROOT" "$MT_END_STEP"        # exits 0 if done or stopped, 3 if the state is ambiguous
  fi
  if [[ "${DRY_RUN:-0}" != "1" ]]; then chain_next "$CKPT_ROOT"; fi
  if [[ "$CHAIN_ACTION" == fresh ]]; then
    start=("${warm_start[@]}")
    resolved_ckpt="$WARM (weights only, fresh chain start)"
  else
    mapfile -t start < <(chain_resume_overrides)
    resolved_ckpt="${CHAIN_LAST_CKPT:-<chain_state did not name it>} (full resume of run $CHAIN_RUN_ID)"
  fi
elif [[ "$MT_KIND" == smoke ]]; then
  CHAIN_ACTION=fresh; CHAIN_EXPECT_STEP=0
  start=("${warm_start[@]}")
  resolved_ckpt="$WARM (weights only, smoke)"
else
  echo "FATAL: MT_KIND must be chain or smoke, not '$MT_KIND'" >&2; exit 2
fi
if [[ "$CHAIN_ACTION" == fresh ]]; then
  test -f "$WARM" || { echo "FATAL: warm checkpoint missing: $WARM" >&2; exit 4; }
fi

# ─── Overrides ──────────────────────────────────────────────────────────────
read -r -a extra <<< "${MT_EXTRA:-}"
overrides=(
  "--config-path=$CONFIG_DIR"
  "--config-name=$CONFIG_NAME"
  "hydra.run.dir=/tmp/$USER/hydra/${JOBID}_\${oc.env:SLURM_PROCID,0}"
  "hydra.output_subdir=.hydra"
  "diagnostics.log.mlflow.run_name=$RUN_NAME"
  "training.max_steps=$MT_END_STEP"
  "training.max_epochs=null"
  "system.output.checkpoints.root=$CKPT_ROOT/"
  "${extra[@]}"
  "${start[@]}"
)

echo "[start] kind=$MT_KIND tag=$MT_TAG action=$CHAIN_ACTION expect_step=$CHAIN_EXPECT_STEP"
echo "[start] resolved checkpoint: $resolved_ckpt"
echo "[start] end step: $MT_END_STEP"
echo "[start] training.max_steps override: $(printf '%s\n' "${overrides[@]}" | grep '^training.max_steps=' | cut -d= -f2)"
echo "[start] checkpoint root: $CKPT_ROOT"
printf 'overrides:\n'; printf '  %s\n' "${overrides[@]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "DRY_RUN=1: nothing trained, nothing submitted"; exit 0; fi

# ─── Provenance ─────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT" "$CKPT_ROOT" "$LOG_DIR" "/tmp/$USER/hydra" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
{
  echo "kind=$MT_KIND tag=$MT_TAG action=$CHAIN_ACTION run_id=${CHAIN_RUN_ID:-new} expect_step=$CHAIN_EXPECT_STEP end_step=$MT_END_STEP host=$(hostname -s) job=$JOBID date=$(date -u +%FT%TZ)"
  for r in "$CODE" "$S/code/downscaling-tools" "$DSFORK"; do printf '%-70s %s\n' "$r" "$(git -C "$r" --no-pager show -s --format='%H %d' HEAD)"; done
  echo "config sha256 $(sha256sum "$CONFIG_DIR/$CONFIG_NAME.yaml" | awk '{print $1}')"
  echo "model group sha256 $(sha256sum "$CONFIG_DIR/model/graphtransformer_transport_residual_edm_mt2.yaml" | awk '{print $1}')"
  echo "start: ${start[*]}"
  echo "tier: experimental (sandbox run)"
} | tee "$OUTPUT/provenance_${JOBID}.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

cd "$OUTPUT"
t0=$(date +%s)
set +e
srun --label --kill-on-bad-exit=1 "$PY" "$CHAIN_TOOLS/chain_guard_launch.py" --expect-step "$CHAIN_EXPECT_STEP" --root "$CKPT_ROOT" \
  -- "$S/jobs/train_with_stats.py" "${overrides[@]}"
rc=$?
set -e
echo "[summary] kind=$MT_KIND tag=$MT_TAG rc=$rc wall_seconds=$(( $(date +%s) - t0 ))"
ls -la "$CKPT_ROOT"/*/ 2>/dev/null || echo "[summary] no checkpoint folder under $CKPT_ROOT"
if [[ "$MT_KIND" == chain ]]; then chain_record "$CKPT_ROOT"; fi
exit $rc
