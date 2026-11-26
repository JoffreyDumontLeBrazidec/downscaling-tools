#!/bin/bash
# ============================================================================
# gs-global-crop on JUPITER (11 Nov 2026; docs epics/paper-o320-o1280/in-progress/20261102_global_from_scratch_o320_o1280_plan.md
# s10.2): FROM-SCRATCH global o320 -> o1280, recipe A of s9.3 = GL512's model and data (parent class
# AnemoiD2ModelEncProcDec, 512 x 16 x 4 heads, EDM, NM50 noise, NL35, AdamW wd 0, freeze_qk_gains on 18 blocks from
# step 0, fixed Triton kernel, global v4 stores, global r8 knn13data graph, global batch 8, seed 756) with NO warm start,
# lr 1k warm-up to 5e-4 then constant, 12 outputs (rule 29), EMA [2000, 5000] with slim exports.
#   LL_MODE=full : arm F (gs_global_512), one model sharded over 4 GH200 (num_gpus_per_model 4), as GL512.
#   LL_MODE=crop : arm X (gs_global_512_crop), dynamic random caps (training.crop.mode dynamic, loader read), one GPU per model.
# Body = GL512's jobs/db_global_job.sh (~/dev/db_global_20261027) with: sandbox on /e/data1 (nothing under Jupiter HOME),
# warm-start checks removed (from scratch), recipe checks for the values above, crop checks for LL_MODE=crop, the soft stop
# of arm F (LL_SOFT_END: the chain ends at that step unless $EXP/state/GO_ON exists; the schedule horizon stays
# LL_END_STEP), GPU utilisation in the nvidia-smi log, and smoke acceptance for a from-scratch start.
#   LL_TAG run tag (run name)   LL_CONFIG config name   LL_END_STEP = config max_steps   LL_NODES   LL_CHAIN_NEXT
#   LL_SEED 756   LL_ACCEPT 1 = smoke acceptance   LL_EXTRA extra Hydra overrides (smokes)   LL_COMPILE 0/1
#   LL_CKPT_ROOT chain root (chain_state.py init before the first submission)   LL_RADIUS crop radius km (crop mode)
# Login-node checks: CFG_ONLY=1 bash jobs/<file>.sbatch ; DRY_RUN=1 bash jobs/<file>.sbatch
# ============================================================================

# REGRESSION BASELINE (17 Nov 2026, downscaling-tools kits/20261117-baselines/reg, made by make_reg_job.py from FS-242's
# gs_job.sh): GL512 450k EMA-2000 warm start, training_approach deterministic, 10 outputs, lr 1e-4, 4,000 km caps.
set -eo pipefail
EXP=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261117-regression-baseline
DATA_ROOT=/e/data1/jureap-data/ecmwf/users/jdumont
RUN_ROOT=$EXP
DATASETS_REPO=/e/home/jusers/dumontlebrazidec1/jupiter/dev/anemoi-datasets-hres-precip-ru   # as GL512
: "${LL_MODE:?}" "${LL_CONFIG:?}" "${LL_TAG:?}" "${LL_END_STEP:?}" "${LL_NODES:?}" "${LL_CHAIN_NEXT:?}" "${LL_CKPT_ROOT:?}"
LL_SEED=${LL_SEED:-756}
LL_ACCEPT=${LL_ACCEPT:-0}
LL_SOFT_END=${LL_SOFT_END:-}
CODE_DIR=${LL_CODE_DIR:-$EXP/code}            # K-reuse smokes: $EXP/code_k (branch exp/gs-global-crop-k-20261111)
PIN_FILE=${LL_CODE_PIN_FILE:-$EXP/CODE_PIN}
LL_DRAWS=${LL_DRAWS:-1}                       # training.crop.dynamic.draws_per_sample expected in the config (1 = key absent)
[[ "$LL_MODE" == full || "$LL_MODE" == crop ]] || { echo "FATAL: LL_MODE must be full or crop, got '$LL_MODE'" >&2; exit 2; }
[[ "$LL_MODE" == full ]] && GPM=4 || GPM=1
if [[ "$LL_MODE" == crop ]]; then : "${LL_RADIUS:?LL_RADIUS (km) is required in crop mode}"; fi
LL_INCHAIN_SMOKE=${LL_INCHAIN_SMOKE:-0}   # 1: a fresh segment first trains 200 steps, runs the smoke gate, then resumes (chains)
GO_ON_FILE="${LL_CKPT_ROOT%/}/state/GO_ON"
# <chain root>/state/SOFT_END (one integer) overrides LL_SOFT_END: the sbatch copy Slurm keeps for a queued job cannot be
# edited, but this body is read at run time, and chain_next successors read the same file. Cloud thread 11 Nov 19:25 UTC:
# crop chains soft end 20,000 -> 100,000 (amendment of plan s10.2 before any read); F stays at 20,000.
SOFT_END_FILE="${LL_CKPT_ROOT%/}/state/SOFT_END"
if [[ -f "$SOFT_END_FILE" ]]; then
  se="$(tr -d ' \n' < "$SOFT_END_FILE")"
  [[ "$se" =~ ^[0-9]+$ ]] || { echo "FATAL: $SOFT_END_FILE holds '$se', not a step" >&2; exit 2; }
  LL_SOFT_END=$se
fi
EFF_END=$LL_END_STEP
if [[ -n "$LL_SOFT_END" && ! -f "$GO_ON_FILE" ]]; then EFF_END=$LL_SOFT_END; fi
FLAG_LINE="[anemoi.models.models.diffusiondownscaler_encoder_processor_decoder][INFO] - Direct-only prognostic channels (out_hres not in in_lres): ['cp', 'tp']"
EDM_LINE="[weight] preconditioning=edm sigma_data=1.0 loss weight FLAT: sigma=5e+05"
NOISE_LINE='Training noise: mixture 0.50 x LogNormal(0.693, 0.750) + 0.50 x LogNormal(1.000, 2.500)'
LR=${LL_LR:-5.0e-4}   # constant rate = min (16 Nov 2026: per-arm value for the batch-32 lr probe)
FREEZE=${LL_FREEZE:-true}   # freeze_qk_gains on all 18 blocks; false for the batch-32 runs (owner 16 Nov 09:07 UTC)
EMA_HL=${LL_EMA:-2000,5000}   # EMA half-lives expected in the config
export ACC_LR=$LR ACC_FREEZE=$FREEZE

# ─── Runtime: Jupiter only (GL512's runtime) ───
ARCH="$(uname -m)"
[[ "$ARCH" == aarch64 && -d /e/data1/jureap-data ]] || { echo "FATAL: this body runs on JUPITER (aarch64, /e/data1) only, this is $ARCH on $(hostname -s)" >&2; exit 3; }
CLUSTER=jupiter
VENV_JUP=$DATA_ROOT/venvs/.ds
set +e +u
source /e/software/default/configs/modules.sh 2>/dev/null
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3 git/2.45.1
mod_rc=$?
set -e
[[ "$mod_rc" == 0 ]] || { echo "FATAL: module load failed (rc $mod_rc)" >&2; exit 3; }
source "$VENV_JUP/bin/activate" || { echo "FATAL: venv activation failed: $VENV_JUP" >&2; exit 3; }
ulimit -c 0
export PYTHONPATH="$DATASETS_REPO/src:$CODE_DIR/anemoi-core/training/src:$CODE_DIR/anemoi-core/models/src:$CODE_DIR/anemoi-core/graphs/src"
python - "$CODE_DIR" "$DATASETS_REPO/src" <<'PYGUARD' || { echo "FATAL: GUARD FAILED: the sandbox source overlay did not win" >&2; exit 3; }
import importlib, os, sys
root = os.path.realpath(sys.argv[1]); ds_root = os.path.realpath(sys.argv[2]); bad = []
for name, want in (("anemoi.models", root), ("anemoi.training", root), ("anemoi.graphs", root),
                   ("anemoi.models.models.diffusiondownscaler_encoder_processor_decoder", root),
                   ("anemoi.training.schedulers.piecewise", root), ("anemoi.models.layers.crop_dynamic", root),
                   ("anemoi.training.data.crop_read", root), ("anemoi.datasets", ds_root)):
    try:
        p = os.path.realpath(importlib.import_module(name).__file__)
    except Exception as exc:                                    # noqa: BLE001
        bad.append(f"  {name}: IMPORT FAILED ({type(exc).__name__}: {exc})"); continue
    (bad.append if not p.startswith(want + os.sep) else print)(f"  {name} -> {p}")
import torch; print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()}")
if bad: print("\n".join(bad)); sys.exit(1)
PYGUARD
RUNTIME="Jupiter Stages/2025 + venv $VENV_JUP + anemoi-datasets fork overlay + sandbox source overlay (guarded)"
set -u

CONFIG_DIR="$EXP/configs"
CFG_FILE="$CONFIG_DIR/$LL_CONFIG.yaml"
JOBID="${SLURM_JOB_ID:-local}"
RUN_NAME="${LL_TAG}"
CACHE_ROOT=$DATA_ROOT/.cache
export OUTPUT="$RUN_ROOT/outputs/$LL_TAG"
CKPT_ROOT="${LL_CKPT_ROOT%/}"
LOG_DIR="$EXP/logs"
export ANEMOI_BASE_SEED=$LL_SEED
export HYDRA_FULL_ERROR=1
export NCCL_IB_TIMEOUT=30 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600 TORCH_NCCL_TIMEOUT=2700
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LL_GPN=${LL_GPN:-4}
export SLURM_GPUS_PER_NODE=$LL_GPN
export LL_CVD=$([[ "$LL_GPN" == 1 ]] && echo 0 || echo 0,1,2,3)
export ANEMOI_ENCODER_TRAP_LOG=${ANEMOI_ENCODER_TRAP_LOG:-0}
export TMPDIR=/tmp
LL_COMPILE=${LL_COMPILE:-0}
[[ "$LL_COMPILE" == 0 || "$LL_COMPILE" == 1 ]] || { echo "FATAL: LL_COMPILE must be 0 or 1, got '$LL_COMPILE'" >&2; exit 2; }
export TORCH_COMPILE_DISABLE=$((1 - LL_COMPILE))
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONFAULTHANDLER=1
export TRITON_CACHE_DIR="$CACHE_ROOT/triton" TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor" CUDA_CACHE_PATH="$CACHE_ROOT/nv_compute"

# ─── Fail-fast checks on the code and the config file ─
C=$CODE_DIR/anemoi-core
test -f "$CFG_FILE" || { echo "FATAL: no config $CFG_FILE" >&2; exit 4; }
grep -q log_normal_mixture "$C/training/src/anemoi/training/train/tasks/diffusiondownscaler.py" || { echo "FATAL: NM code missing" >&2; exit 43; }
test -f "$C/models/src/anemoi/models/models/diffusiondownscaler_encoder_processor_decoder.py" || { echo "FATAL: parent class missing" >&2; exit 43; }
grep -q 'freeze_qk_gains: bool = False' "$C/models/src/anemoi/models/layers/processor.py" || { echo "FATAL: processor freeze_qk_gains plumbing missing" >&2; exit 43; }
[[ "$(grep -c 'freeze_qk_gains: bool = False' "$C/models/src/anemoi/models/layers/mapper.py")" -ge 2 ]] || { echo "FATAL: mapper freeze_qk_gains plumbing missing" >&2; exit 43; }
grep -q 'QK-norm parameters frozen' "$C/models/src/anemoi/models/layers/block.py" || { echo "FATAL: frozen-gain log line missing" >&2; exit 43; }
grep -q 'export_keep_every' "$C/training/src/anemoi/training/diagnostics/callbacks/ema.py" && grep -q 'skip_raw_export' "$C/training/src/anemoi/training/diagnostics/callbacks/checkpoint.py" || { echo "FATAL: slim-export code (6c88bb7be) missing" >&2; exit 43; }
if [[ "$LL_MODE" == crop ]]; then
  grep -qF 'loader_read: bool = False' "$C/models/src/anemoi/models/layers/crop_dynamic.py" \
    && grep -qF 'validation_radius_km: Optional[float] = None' "$C/models/src/anemoi/models/layers/crop_dynamic.py" \
    && grep -q 'def _next_dynamic_crop' "$C/training/src/anemoi/training/train/tasks/diffusiondownscaler.py" \
    || { echo "FATAL: dynamic-crop code with loader read and validation_radius_km missing" >&2; exit 43; }
fi
GT="$C/models/src/anemoi/models/triton/gt.py"
grep -qF 'out_saved = torch.empty((N_dst, H, C), device=q.device, dtype=torch.float32)' "$GT" \
  && grep -qF 'D = torch.empty((N_dst, H), device=q.device, dtype=torch.float32)' "$GT" \
  || { echo "FATAL: $GT does not carry the bbca1748 fp32 saved output and fp32 D (fixed kernel)" >&2; exit 47; }
gt_blob="$(git -C "$C" rev-parse HEAD:models/src/anemoi/models/triton/gt.py)"
[[ "$gt_blob" == 88b03616* ]] || { echo "FATAL: gt.py blob $gt_blob is not the fixed kernel 88b03616" >&2; exit 47; }
echo "[gtfix] $GT carries bbca1748: fp32 saved output, fp32 D; blob $gt_blob; md5 $(md5sum "$GT" | cut -c1-32)"
for key in dataset_in_lres dataset_in_hres dataset_out_hres residual_statistics truncation truncation_inv; do
  p="$(grep -E "^    ${key}: " "$CFG_FILE" | awk '{print $2}')"
  test -e "$p" || { echo "FATAL: $key missing: $p" >&2; exit 5; }
done
: "${LL_WARM_FILE:?}" "${LL_WARM_SHA:?}"
grep -qxF "    warm_start: $LL_WARM_FILE" "$CFG_FILE" || { echo "FATAL: system.input.warm_start must be $LL_WARM_FILE" >&2; exit 42; }
[[ -f "$LL_WARM_FILE.sha256ok" && "$(cat "$LL_WARM_FILE.sha256ok")" == "$LL_WARM_SHA" ]] || {
  [[ "$(sha256sum "$LL_WARM_FILE" | awk '{print $1}')" == "$LL_WARM_SHA" ]] || { echo "FATAL: warm file sha256 is not $LL_WARM_SHA" >&2; exit 42; }
  echo "$LL_WARM_SHA" > "$LL_WARM_FILE.sha256ok"; }
echo "[warm] $LL_WARM_FILE sha256 $LL_WARM_SHA"
grep -qxF "  training_approach: deterministic" "$CFG_FILE" || { echo "FATAL: training_approach must be deterministic" >&2; exit 42; }
export LL_WARM_FILE
graph="$(grep -E '^    graph: ' "$CFG_FILE" | awk '{print $2}')"
[[ "$graph" == /e/data1/jureap-data/ecmwf/users/jdumont/outputs/graphs/o320_o1280_icosahedral_r8_multiscale_h1_s8-1encoder_halo_true_r8_knn13data.pt ]] || { echo "FATAL: the arm uses GL512's global r8 knn13data graph, the config names $graph" >&2; exit 43; }
for ds in in_lres in_hres out_hres; do
  want="$(awk -v d="$ds" '$2 == d {print $1}' "$EXP/GRAPH_SHA256")"
  have="$(sha256sum "$(readlink -f "${graph%.pt}_${ds}.pt")" | awk '{print $1}')"
  [[ -n "$want" && "$have" == "$want" ]] || { echo "FATAL: graph file ${graph%.pt}_${ds}.pt sha256 $have is not GRAPH_SHA256's '$want'" >&2; exit 43; }
done
echo "[graph] global r8 knn13data graph files match GRAPH_SHA256 (GL512's, used as is)"
grep -qE '^ +nodes_attribute_name: area_weight$' "$CFG_FILE" || { echo "FATAL: global node weights (area_weight) not selected" >&2; exit 42; }
grep -q "^  max_steps: ${LL_END_STEP}\$" "$CFG_FILE" || { echo "FATAL: max_steps in the config != $LL_END_STEP" >&2; exit 42; }
grep -q '^  accum_grad_batches: 1$' "$CFG_FILE" && grep -qE "^    num_nodes: ${LL_NODES}\$" "$CFG_FILE" && grep -qE "^    num_gpus_per_node: ${LL_GPN}\$" "$CFG_FILE" || { echo "FATAL: the config is not the ${LL_NODES}-node layout" >&2; exit 42; }
python - "$CFG_FILE" "$LL_MODE" "${LL_RADIUS:-0}" "$LL_DRAWS" "$LR" "$FREEZE" "$EMA_HL" <<'PYCHK' || { echo "FATAL: the config fails the recipe check above" >&2; exit 50; }
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])); mode = sys.argv[2]; radius = float(sys.argv[3]); draws = int(sys.argv[4]); lr = float(sys.argv[5]); freeze = sys.argv[6] == "true"; ema = [int(x) for x in sys.argv[7].split(",")]; t = c["training"]; mm = c["model"]; m = mm["model"]
out12 = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "cp", "tp"]   # GL512's 10 outputs (name kept)
gpm = 4 if mode == "full" else 1
d = m["diffusion"]
checks = {
    "optimizer.lr = ${training.lr.rate}": t["optimizer"].get("lr") == "${training.lr.rate}",
    "AdamW betas 0.9/0.95, weight_decay 0.0": t["optimizer"]["_target_"] == "torch.optim.AdamW" and t["optimizer"].get("weight_decay") == 0.0 and t["optimizer"].get("betas") == [0.9, 0.95],
    f"EMA {ema}, slim exports (export_keep_every, skip_raw_export)": bool((t.get("ema") or {}).get("enabled")) and t["ema"].get("halflife_steps") == ema
            and int(t["ema"].get("export_keep_every") or 0) > 0 and t["ema"].get("skip_raw_export") is True,
    f"lr constant, rate = min = {lr:g}, warm-up 1000": t["lr"].get("schedule") == "constant" and t["lr"].get("min") == lr
            and t["lr"].get("rate") == lr and t["lr"].get("warmup") == 1000,
    "lr.iterations = max_steps": t["lr"].get("iterations") == t.get("max_steps"),
    "WARM START: transfer_learning true, load_weights_only true, warm_start set": t.get("transfer_learning") is True and t.get("load_weights_only") is True
            and bool(c["system"]["input"].get("warm_start")),
    "training_approach deterministic": t.get("training_approach") == "deterministic",
    "no optimizer_param_groups (one lr everywhere)": "optimizer_param_groups" not in t,
    "nothing frozen by submodule": t.get("submodules_to_freeze") in ([], None),
    f"num_gpus_per_model {gpm}": c["system"]["hardware"]["num_gpus_per_model"] == gpm,
    "gpus per node 4": c["system"]["hardware"]["num_gpus_per_node"] == 4,
    "keep_batch_sharded false": mm.get("keep_batch_sharded") is False,
    "read_group_size = num_gpus_per_model": c["dataloader"].get("read_group_size") == "${system.hardware.num_gpus_per_model}",
    "batch 1 per model": c["dataloader"]["batch_size"]["training"] == 1,
    "parent class": m["_target_"] == "anemoi.models.models.diffusiondownscaler_encoder_processor_decoder.AnemoiD2ModelEncProcDec",
    "num_channels 512": mm.get("num_channels") == 512,
    "no lossless block": "lossless" not in m,
    "edm preconditioning": d.get("preconditioning") == "edm" and "x_prediction_weight" not in d,
    "NM50 noise": abs(d["log_normal_mean"] - 0.6931471805599453) < 1e-12 and d["log_normal_std"] == 0.75 and d.get("log_normal_mixture") == {"weight": 0.5, "mean": 1.0, "std": 2.5},
    "uncertainty weighting off": d["uncertainty_weighting"]["enabled"] is False,
    f"qk_norm on, freeze_qk_gains {freeze} on encoder, processor, decoder": all(mm[p].get("qk_norm") is True and mm[p].get("freeze_qk_gains") is freeze for p in ("encoder", "processor", "decoder")),
    "processor 16 x 4 heads, encoder and decoder 4 heads": mm["processor"]["num_layers"] == 16 and all(mm[p]["num_heads"] == 4 for p in ("encoder", "processor", "decoder")),
    "graph attention triton everywhere": all(mm[p].get("graph_attention_backend") == "triton" for p in ("encoder", "processor", "decoder")),
    "mapper sharding heads (never edges)": all(mm[p].get("shard_strategy") == "heads" for p in ("encoder", "decoder")),
    "tp, cp direct prediction": c["data"]["datasets"]["out_hres"].get("direct_prediction") == ["tp", "cp"],
    "GL512's 10 outputs in training/validation/test": all(c["dataloader"][s]["datasets"]["out_hres"].get("reorder") == out12
            and {"z_500", "t_850"} <= set(c["dataloader"][s]["datasets"]["out_hres"]["drop"]) for s in ("training", "validation", "test")),
    "NL35: general_variable z 0.175, t 0.35, by param": t["scalers"]["datasets"]["out_hres"]["general_variable"]["weights"].get("z") == 0.175
            and t["scalers"]["datasets"]["out_hres"]["general_variable"]["weights"].get("t") == 0.35
            and "z_500" not in t["scalers"]["datasets"]["out_hres"]["general_variable"]["weights"]
            and "t_850" not in t["scalers"]["datasets"]["out_hres"]["general_variable"]["weights"],
    "no pressure_level scaler": "pressure_level" not in t["scalers"]["datasets"]["out_hres"],
    "precipitation in (tp, cp outputs, std normaliser)": "tp" in out12 and c["data"]["datasets"]["out_hres"]["processors"]["normalizer"]["config"]["std"] == ["tp", "cp"],
    "leads 12-72 h": c["dataloader"].get("steps") == [12, 24, 36, 48, 60, 72],
    "dates: train <= 2021, valid 2022, test 2023": c["dataloader"].get("training_date") == 2021 and c["dataloader"].get("valid_date") == 2022 and c["dataloader"].get("test_date") == 2023,
    "checkpoint every 1000 steps, latest by-step kept": c["diagnostics"]["checkpoint"]["every_n_train_steps"] == {"save_frequency": 1000, "num_models_saved": 1},
    "gradient clip 1.0 norm": t["gradient_clip"] == {"val": 1.0, "algorithm": "norm"},
    "bf16-mixed": t["precision"] == "bf16-mixed",
}
if mode == "crop":
    checks[f"training.crop dynamic, fixed {radius:.0f} km + halo 800, loader read, 16 validation caps of 3000 km"] = t.get("crop") == {
        "enabled": True, "mode": "dynamic", "seed": t.get("crop", {}).get("seed") if t.get("crop", {}).get("seed") in (None, 756) else "BAD",
        "dynamic": {"radius_mode": "fixed", "radius_km": radius, "halo_km": 800.0, "centre_spacing_km": 25.0, "capacity": "fixed",
                    "capacity_probes": 256, "pad_multiple": 8, "loader_read": True, "loader_margin_km": 300.0,
                    "validation_caps": 16, "validation_radius_km": 3000.0, **({"draws_per_sample": draws} if draws > 1 else {})}}
    checks["compile dynamic, recompile_limit 128"] = all((x.get("options") or {}).get("dynamic") is True for x in mm["compile"]) and t.get("recompile_limit") == 128
else:
    checks["no crop (full globe)"] = not (t.get("crop") or {}).get("enabled", False)
for k, ok in checks.items():
    print(f"  [cfg] {'ok  ' if ok else 'FAIL'} {k}")
sys.exit(0 if all(checks.values()) else 1)
PYCHK
test -f "$C/training/src/anemoi/training/diagnostics/callbacks/ema.py" || { echo "FATAL: no EMA callback in this code" >&2; exit 45; }
test -f "$C/training/src/anemoi/training/schedulers/piecewise.py" || { echo "FATAL: no PiecewiseLRScheduler in this code" >&2; exit 51; }
[[ -z "$(git -C "$C" status --porcelain --untracked-files=no)" ]] || { echo "FATAL: uncommitted changes in $C" >&2; exit 44; }
[[ -z "$(git -C "$DATASETS_REPO" status --porcelain --untracked-files=no)" ]] || { echo "FATAL: uncommitted changes in $DATASETS_REPO" >&2; exit 44; }
V4=/e/data1/jureap-data/ecmwf/ai-ml/datasets
for z in downscaling-od-cf-eefh-0001-mars-o320-2003-2023-24h-12h-v4.zarr downscaling-od-cf-enfh-0001-mars-o1280-2003-2023-24h-12h-v4.zarr; do
  n=$(ls "$V4/$z/data" 2>/dev/null | grep -c '^[0-9]' || true)
  [[ "$n" -ge 86400 ]] || { echo "FATAL: global v4 store $z incomplete ($n of 86400 chunk files)" >&2; exit 49; }
done
pin="$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "$PIN_FILE" 2>/dev/null)"
head_sha="$(git -C "$C" rev-parse HEAD)"
[[ -n "$pin" && "$head_sha" == "$pin"* ]] || { echo "FATAL: code HEAD $head_sha is not the pin '$pin' in $PIN_FILE" >&2; exit 46; }
git -C "$C" branch -r --contains "$head_sha" 2>/dev/null | grep -q 'origin/' || { echo "FATAL: code HEAD $head_sha is not on any origin branch (push before sbatch)" >&2; exit 46; }
ds_pin="$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "$EXP/DATASETS_PIN" 2>/dev/null)"
ds_sha="$(git -C "$DATASETS_REPO" rev-parse HEAD)"
[[ -n "$ds_pin" && "$ds_sha" == "$ds_pin"* ]] || { echo "FATAL: $DATASETS_REPO HEAD $ds_sha is not the pin '$ds_pin'" >&2; exit 46; }

# ─── Start mode: from the run's state on disk (downscaling-tools tools/chain; standing rule 15) ───
source "$EXP/jobs/chain/chain_segment.sh"
# Step-0 gate on the unproven SHA (cloud thread 14:12 UTC 11 Nov, arm F): before anything else, read the verdict of
# gs_trunk_step0 against GL512's pin (tools/step0_verdict.py); wait up to LL_STEP0_WAIT s for it. Not PASS: write why,
# exit 0 with no successor and no training.
if [[ "${LL_STEP0_GATE:-0}" == 1 && "${CFG_ONLY:-0}" != "1" && "${DRY_RUN:-0}" != "1" ]]; then
  w=0; v=2
  while :; do
    set +e; python3 "$EXP/tools/step0_verdict.py"; v=$?; set -e
    [[ "$v" != 2 ]] && break
    (( w >= ${LL_STEP0_WAIT:-0} )) && break
    sleep 60; w=$((w + 60))
  done
  if [[ "$v" != 0 ]]; then
    mkdir -p "$CKPT_ROOT"
    echo "$(date -u +%FT%TZ) job=${SLURM_JOB_ID:-local} tag=$LL_TAG: step-0 gate not passed (verdict code $v: 1 FAIL, 2 not ready after ${w}s); no training, no successor. See $EXP/step0/state/VERDICT.json" | tee -a "$CKPT_ROOT/STEP0_BLOCKED.txt"
    exit 0
  fi
  echo "[step0] gate PASS: $(python3 -c "import json;print(json.load(open('$EXP/step0/state/VERDICT.json'))['why'])")"
fi
if [[ "${CFG_ONLY:-0}" == "1" || "${DRY_RUN:-0}" == "1" ]]; then
  CHAIN_ACTION=fresh; CHAIN_EXPECT_STEP=0
else
  # one hook pass before resolving (14 Nov 2026): a "done" successor exits inside chain_resolve, so the reads of the last
  # exports of the previous segment are submitted here (12 Nov: the 100k reads waited two days)
  [[ "$LL_CHAIN_NEXT" == 1 ]] && { bash "$EXP/tools/read_hook.sh" "$LL_TAG" >> "$LOG_DIR/read_hook_${JOBID}.log" 2>&1 || true; }
  chain_resolve "$CKPT_ROOT" "$EFF_END"
  if [[ "$LL_CHAIN_NEXT" == 1 && ! ( "$LL_INCHAIN_SMOKE" == 1 && "$CHAIN_ACTION" == fresh ) ]]; then chain_next "$CKPT_ROOT"; fi
fi
if [[ "$CHAIN_ACTION" == fresh && -n "${LL_RESUME_FROM:-}" ]]; then
  # arm H (12 Nov 2026): a NEW run that resumes the full training state (weights, optimizer, lr scheduler, EMA shadows,
  # global_step) of another run's training checkpoint: warm_start = that file, load_weights_only false, no run_id
  # (train.py: ckpt_path = warm_start when load_weights_only is false). The file is a copy outside the source chain root.
  : "${LL_RESUME_STEP:?LL_RESUME_STEP is required with LL_RESUME_FROM}"
  [[ -f "$LL_RESUME_FROM" ]] || { echo "FATAL: LL_RESUME_FROM $LL_RESUME_FROM missing" >&2; exit 42; }
  start=("training.run_id=null" "training.fork_run_id=null" "system.input.warm_start=$LL_RESUME_FROM"
         "training.load_weights_only=False" "training.transfer_learning=False")
  CHAIN_EXPECT_STEP=$LL_RESUME_STEP
elif [[ "$CHAIN_ACTION" == fresh ]]; then
  start=("training.run_id=null" "training.fork_run_id=null")   # from scratch: no transfer learning, step 0
  CHAIN_EXPECT_STEP=0
else
  mapfile -t start < <(chain_resume_overrides)
fi

# Restart seed (plan s11.3, 14 Nov 2026): a segment that starts from a checkpoint draws new data, noise and crop streams:
# base seed = LL_SEED x 1000 + start step (anemoi multiplies a seed < 1000 by 1000, so a fresh start keeps 756 -> 756000).
# LL_SEED_OVERRIDE (tests only) forces a value.
set_start_seed() {
  if (( ${CHAIN_EXPECT_STEP:-0} > 0 )); then export ANEMOI_BASE_SEED=$(( LL_SEED * 1000 + CHAIN_EXPECT_STEP )); else export ANEMOI_BASE_SEED=$LL_SEED; fi
  [[ -n "${LL_SEED_OVERRIDE:-}" ]] && export ANEMOI_BASE_SEED=$LL_SEED_OVERRIDE
  echo "[seed] start step ${CHAIN_EXPECT_STEP:-0}: ANEMOI_BASE_SEED=$ANEMOI_BASE_SEED"
}
set_start_seed

# ─── Overrides ──────────────────────────────────────────────────────────────
overrides=(
  "--config-path=$CONFIG_DIR"
  "--config-name=$LL_CONFIG"
  "config_validation=False"
  "hydra.run.dir=/tmp/$USER/hydra/${JOBID}_\${oc.env:SLURM_PROCID,0}"
  "hydra.output_subdir=.hydra"
  "diagnostics.log.mlflow.run_name=$RUN_NAME"
  "training.max_steps=$EFF_END"
  "training.max_epochs=null"
  "system.output.checkpoints.root=$CKPT_ROOT/"
  "${start[@]}"
)
if [[ -n "${LL_EXTRA:-}" ]]; then read -r -a extra <<< "$LL_EXTRA"; overrides+=("${extra[@]}"); fi

if [[ "${CFG_ONLY:-0}" == "1" ]]; then
  RES="$EXP/jobs/rendered/${LL_TAG}.resolved.yaml"
  mkdir -p "$EXP/jobs/rendered"
  anemoi-training train "${overrides[@]}" --cfg job --resolve > "$RES"
  python - "$RES" "$EFF_END" "$LL_MODE" "${LL_RESUME_FROM:-}" "$LR" "$FREEZE" "$EMA_HL" <<'PYRES' || { echo "FATAL: the resolved config fails the check above" >&2; exit 52; }
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])); end = int(sys.argv[2]); mode = sys.argv[3]; resume = sys.argv[4] or None; lr = float(sys.argv[5]); freeze = sys.argv[6] == "true"; ema = [int(x) for x in sys.argv[7].split(",")]; t = c["training"]; mm = c["model"]; d = mm["model"]["diffusion"]
gpm = 4 if mode == "full" else 1
checks = {
    ("full resume of " + resume if resume else "warm start (transfer_learning true, load_weights_only true)"): (t["transfer_learning"] is False and t["load_weights_only"] is False and c["system"]["input"].get("warm_start") == resume) if resume
            else (t["transfer_learning"] is True and t["load_weights_only"] is True and c["system"]["input"].get("warm_start") == __import__("os").environ["LL_WARM_FILE"]),
    "training_approach deterministic": t.get("training_approach") == "deterministic",
    "edm, NM50": d.get("preconditioning") == "edm" and abs(d["log_normal_mean"] - 0.6931471805599453) < 1e-12 and d["log_normal_std"] == 0.75
            and d.get("log_normal_mixture") == {"weight": 0.5, "mean": 1.0, "std": 2.5},
    f"optimizer lr {lr:g} resolved, rate = min = {lr:g}": t["optimizer"].get("lr") == lr and t["lr"]["rate"] == lr and t["lr"]["min"] == lr,
    "weight_decay 0.0": t["optimizer"].get("weight_decay") == 0.0,
    f"EMA {ema}": t["ema"].get("halflife_steps") == ema and t["ema"].get("enabled") is True,
    f"num_gpus_per_model {gpm}, read_group_size {gpm}": c["system"]["hardware"]["num_gpus_per_model"] == gpm and c["dataloader"]["read_group_size"] == gpm,
    f"max_steps {end}": t["max_steps"] == end,
    f"freeze_qk_gains {freeze} x3": all(mm[p].get("freeze_qk_gains") is freeze for p in ("encoder", "processor", "decoder")),
    "10 outputs": c["dataloader"]["training"]["datasets"]["out_hres"]["reorder"] == ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "cp", "tp"],
}
if mode == "crop":
    checks["crop resolved (dynamic, loader read, 16 caps of 3000 km)"] = t["crop"]["mode"] == "dynamic" and t["crop"]["dynamic"]["loader_read"] is True \
        and t["crop"]["dynamic"]["validation_caps"] == 16 and t["crop"]["dynamic"]["validation_radius_km"] == 3000.0
for k, ok in checks.items():
    print(f"  [resolved] {'ok  ' if ok else 'FAIL'} {k}")
sys.exit(0 if all(checks.values()) else 1)
PYRES
  echo "CFG_ONLY=1: resolved config $RES checked"
  exit 0
fi

PROV="$(
  echo "mode=$LL_MODE config=$LL_CONFIG tag=$LL_TAG cluster=$CLUSTER chain_action=$CHAIN_ACTION run_id=${CHAIN_RUN_ID:-new} expect_step=$CHAIN_EXPECT_STEP end_step=$LL_END_STEP effective_end=$EFF_END soft_end=${LL_SOFT_END:-none} soft_end_file=$([[ -f $SOFT_END_FILE ]] && cat $SOFT_END_FILE || echo none) go_on=$([[ -f $GO_ON_FILE ]] && echo yes || echo no) inchain_smoke=$LL_INCHAIN_SMOKE seed=$ANEMOI_BASE_SEED host=$(hostname -s) job=$JOBID run_name=$RUN_NAME ckpt_root=$CKPT_ROOT date=$(date -u +%FT%TZ)"
  printf '%-22s %s\n' code/anemoi-core "$(git -C "$C" --no-pager show -s --format='%H %d' HEAD)"
  printf '%-22s %s\n' anemoi-datasets "$(git -C "$DATASETS_REPO" --no-pager show -s --format='%H %d' HEAD) ($DATASETS_REPO)"
  echo "runtime: $RUNTIME"
  echo "compile: LL_COMPILE=$LL_COMPILE TORCH_COMPILE_DISABLE=$TORCH_COMPILE_DISABLE"
  echo "config sha256 $(sha256sum "$CFG_FILE" | awk '{print $1}')"
  echo "graph sha256 $(sha256sum "${graph%.pt}_out_hres.pt" | awk '{print $1}')"
  echo "start: ${start[*]}"
  echo "extra: ${LL_EXTRA:-none}"
  echo "chain tools: $(cat "$EXP/jobs/chain/COMMIT" 2>/dev/null)"
  echo "tier: experimental (sandbox run)"
  echo "gt.py blob $gt_blob; gpus per node $LL_GPN, nodes $LL_NODES, gpus per model $GPM, radius ${LL_RADIUS:-none}, accept $LL_ACCEPT"
)"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "$PROV"
  printf 'overrides:\n'; printf '  %s\n' "${overrides[@]}"
  if [[ -f "$CKPT_ROOT/CHAIN.json" ]]; then echo "DRY_RUN: chain root $CKPT_ROOT already has a manifest"; else echo "DRY_RUN: chain root $CKPT_ROOT has no manifest: 'chain_state.py init' is still owed"; fi
  echo "DRY_RUN=1: every guard passed; not launching"; exit 0
fi

# ─── Provenance and diagnostics ─────────────────────────────────────────────
mkdir -p "$OUTPUT" "$CKPT_ROOT" "$LOG_DIR" "$EXP/state" "/tmp/$USER/hydra" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
echo "$PROV" | tee "$OUTPUT/provenance_${JOBID}.txt"
echo "VIRTUAL_ENV=$VIRTUAL_ENV  python=$(command -v python)  $(python -V 2>&1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
GPULOG="$LOG_DIR/${SLURM_JOB_NAME:-ll}_${JOBID}.gpumem.csv"
export H9DBG_RANKDIR="$LOG_DIR/ranks/${SLURM_JOB_NAME:-ll}-${JOBID}"
printf 'overrides:\n'; printf '  %s\n' "${overrides[@]}"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 5 > "$GPULOG" 2>&1 &
SMI_PID=$!
before="$(ls -d "$OUTPUT"/logs/mlflow/*/*/ 2>/dev/null | sort || true)"
(
  for _ in $(seq 120); do
    sleep 30
    new="$(comm -13 <(echo "$before") <(ls -d "$OUTPUT"/logs/mlflow/*/*/ 2>/dev/null | sort) | xargs -r -n1 basename | grep -E '^[0-9a-f]{32}$' || true)"
    if [[ -n "$new" ]]; then
      echo "$(date -u +%FT%TZ) job=$JOBID tag=$LL_TAG cluster=$CLUSTER action=$CHAIN_ACTION new_mlflow_run=$(echo $new | tr ' ' ',')" >> "$EXP/state/$LL_TAG.run_id"
      break
    fi
  done
) &
CAP_PID=$!
# Log guard (fixed 11 Nov 15:55 UTC: the crop pattern matched no line of 8c12d4dec, whose line is
# "training.crop dynamic (_crop_sampler, radius R km): N centres, capacities ..."; 7afa4a59 printed it without the
# parenthesis; the pattern now accepts both). Window: 90 x 60 s = 90 min counted from this point, i.e. just before srun
# (covers graph load, capacity probes, compile and sanity validation on a cold node; the crop line is printed at the first
# cropped training step). On a miss: if training losses are being logged, warn and write <chain root>/GUARD_MISS.txt
# (never kill a training job for a guard pattern); cancel only when there is also no training loss in the log.
JOBLOG="$LOG_DIR/${SLURM_JOB_NAME:-ll}_${JOBID}.out"
CROP_LINE_RE='training\.crop dynamic( \([^)]*\))?: [0-9]+ centres, capacities'
(
  miss=""
  for _ in $(seq 90); do
    sleep 60
    miss=""
    grep -qF "$NOISE_LINE" "$JOBLOG" 2>/dev/null || miss="$miss noise-mixture"
    grep -qF "$FLAG_LINE" "$JOBLOG" 2>/dev/null || miss="$miss parent-class"
    if [[ "$LL_MODE" == crop ]]; then
      grep -qsE 'crop loader_read on, [0-9]+ centres' "$JOBLOG" "${JOBLOG%.out}.err" || miss="$miss crop-loader"
      grep -qsE "$CROP_LINE_RE" "$JOBLOG" "${JOBLOG%.out}.err" || miss="$miss crop-sampler"
    fi
    [[ -z "$miss" ]] && { echo "[log guard] noise-mixture line, parent-class line$([[ "$LL_MODE" == crop ]] && echo ' and crop lines') found"; exit 0; }
  done
  if grep -qsE 'train_multi_dataset_loss_step=[0-9]' "$JOBLOG" "${JOBLOG%.out}.err"; then
    echo "WARNING: [log guard] after 90 min missing:$miss; training losses ARE being logged, so the job continues; see GUARD_MISS.txt" >&2
    echo "$(date -u +%FT%TZ) job=$JOBID tag=$LL_TAG log guard missing:$miss (training losses present; job not cancelled)" >> "$CKPT_ROOT/GUARD_MISS.txt"
    exit 0
  fi
  echo "FATAL: [log guard] after 90 min missing:$miss and no training loss in $JOBLOG; cancelling" >&2
  echo "$(date -u +%FT%TZ) job=$JOBID tag=$LL_TAG log guard missing:$miss and no training loss; job cancelled" >> "$CKPT_ROOT/GUARD_MISS.txt"
  scancel "$JOBID"
) &
LOGG_PID=$!
# Read hook (12 Nov 2026): every 5 min during training, submit the reads of this chain whose EMA-2000 export exists
# (tools/read_hook.sh: once per step, marker + lock), and once more after training (run_hook_now below).
HOOK_PID=""
run_hook_now() { [[ "$LL_CHAIN_NEXT" == 1 ]] && bash "$EXP/tools/read_hook.sh" "$LL_TAG" >> "$LOG_DIR/read_hook_${JOBID}.log" 2>&1 || true; }
if [[ "$LL_CHAIN_NEXT" == 1 ]]; then
  ( while sleep 300; do bash "$EXP/tools/read_hook.sh" "$LL_TAG" >> "$LOG_DIR/read_hook_${JOBID}.log" 2>&1; done ) &
  HOOK_PID=$!
fi
trap 'kill "$SMI_PID" "$CAP_PID" "$LOGG_PID" $HOOK_PID 2>/dev/null || true' EXIT

cd "$OUTPUT"
run_train() {   # run_train <expect_step> <override...>
  local expect="$1"; shift
  set +e
  srun --kill-on-bad-exit=1 --label bash -c 'export CUDA_VISIBLE_DEVICES="$LL_CVD"; exec python "$@"' ll-wrap \
    "$EXP/jobs/chain/chain_guard_launch.py" --expect-step "$expect" --root "$CKPT_ROOT" \
    -- "$EXP/jobs/stall_watchdog.py" "$EXP/jobs/train_with_peakmem_h9.py" "$@"
  rc=$?
  set -e
}
summary() {
  peak_mib="$(awk -F', ' 'NR>1 {gsub(/ MiB/,"",$3); if ($3+0>m) m=$3+0} END {print m+0}' "$GPULOG")"
  util="$(awk -F', ' 'NR>1 {gsub(/ %/,"",$4); s+=$4; n++} END {if (n) printf "%.1f", s/n}' "$GPULOG")"
  echo "[summary] $1 mode=$LL_MODE config=$LL_CONFIG rc=$rc wall_seconds=$(( $(date +%s) - t0 )) nvidia_smi_peak_memory_used_MiB=$peak_mib mean_gpu_util_pct=$util (first node, incl. start-up) gpumem_log=$GPULOG"
  du -sh "$CKPT_ROOT"/*/ 2>/dev/null || true
  ls -la "$CKPT_ROOT"/*/ 2>/dev/null | tail -12 || echo "[summary] no checkpoint folder under $CKPT_ROOT"
}
# ─── Smoke acceptance for a from-scratch start (ACC_KIND smoke: 200-step smoke with 50-step warm-up; inchain: the first
#     200 steps of a chain with the real 1000-step warm-up); sets acc=0 on pass ───
source "$EXP/jobs/reg_accept.sh"   # accept() (moved out 11 Nov 19:10 UTC)

t0=$(date +%s)
if [[ "$LL_INCHAIN_SMOKE" == 1 && "$CHAIN_ACTION" == fresh ]]; then
  # Phase A: the first 200 steps of the chain (real recipe, checkpoint every 100 so the step-200 state resumes), then the gate
  ovA=()
  for o in "${overrides[@]}"; do [[ "$o" == training.max_steps=* ]] && ovA+=("training.max_steps=200") || ovA+=("$o"); done
  ovA+=("diagnostics.checkpoint.every_n_train_steps.save_frequency=100")
  echo "[inchain] phase A: steps 0-200 then the smoke gate; successor queued only on a pass"
  run_train 0 "${ovA[@]}"
  summary "phase A (steps 0-200)"
  chain_record "$CKPT_ROOT"
  acc=1
  [[ "$rc" == 0 ]] && accept inchain
  if [[ "$acc" != 0 ]]; then
    { echo "$(date -u +%FT%TZ) job=$JOBID tag=$LL_TAG: in-chain smoke gate FAILED (rc=$rc); no successor queued, chain not continued. Log: $JOBLOG"; } | tee -a "$CKPT_ROOT/SMOKE_FAILED.txt"
    "$CHAIN_PY" "$CHAIN_TOOLS/chain_state.py" log --root "$CKPT_ROOT" "in-chain smoke gate FAILED at step 200 (job $JOBID); see SMOKE_FAILED.txt" || true
    exit 1
  fi
  "$CHAIN_PY" "$CHAIN_TOOLS/chain_state.py" log --root "$CKPT_ROOT" "in-chain smoke gate PASSED at step 200 (job $JOBID); resuming to $EFF_END" || true
  # Phase B: resume from the step-200 checkpoint on disk, queue the successor, train to the effective end
  chain_resolve "$CKPT_ROOT" "$EFF_END"
  if [[ "$LL_CHAIN_NEXT" == 1 ]]; then chain_next "$CKPT_ROOT"; fi
  mapfile -t start < <(chain_resume_overrides)
  set_start_seed
  ovB=()
  for o in "${overrides[@]}"; do [[ "$o" == training.run_id=* || "$o" == training.fork_run_id=* ]] || ovB+=("$o"); done
  ovB+=("${start[@]}")
  echo "[inchain] phase B: chain_action=$CHAIN_ACTION expect_step=$CHAIN_EXPECT_STEP start: ${start[*]}"
  run_train "$CHAIN_EXPECT_STEP" "${ovB[@]}"
  kill "$SMI_PID" 2>/dev/null || true
  summary "phase B"
  chain_record "$CKPT_ROOT"
  run_hook_now
  exit $rc
fi
run_train "$CHAIN_EXPECT_STEP" "${overrides[@]}"
kill "$SMI_PID" 2>/dev/null || true
summary "segment"
chain_record "$CKPT_ROOT"
run_hook_now
if [[ "$rc" == 0 && "$LL_ACCEPT" == 1 ]]; then
  accept smoke
  [[ $acc == 0 ]] || { echo "[accept] FAIL (exit 1: an afterok successor will not start)"; exit 1; }
fi
exit $rc
