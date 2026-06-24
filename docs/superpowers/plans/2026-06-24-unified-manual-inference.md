# Unified Manual Inference Backend Implementation Plan

> **For Codex:** Execute this plan inline and in order. Keep the legacy backend unchanged, run only the two 4-GPU smoke jobs, and do not start the full parity campaign or publish any scoreboard.

**Goal:** Repair b785bf12 o320→o1280 manual bundle inference by using the proven `downscaling_unified` runner topology, while retaining canonical local NetCDF outputs (`y`, `x_interp`, `y_pred`) and the existing legacy direct backend.

**Architecture:** Add an explicit `unified` inference backend selected by a generated Anemoi runner configuration. It applies that configuration's environment before importing the runner, registers `downscaling_unified`, creates the runner through `anemoi.inference.runners.create_runner`, and passes the runner's model interface plus distributed metadata into the existing bundle loop and NetCDF writer. The existing direct path remains the default `legacy` backend and is loaded lazily so it cannot pre-initialize model modules for the unified path.

**Tech Stack:** Python 3.12, pytest, PyTorch distributed, anemoi-inference, PrepML `model-config`, Slurm on 4×A100-40GB.

---

### Task 1: Add a red unit-test contract for the unified loader

**Files:**

- Create: `eval/predict/tests/test_unified_runner.py`
- Modify: `eval/predict/tests/test_types.py`

**Step 1: Write the failing tests**

Cover a missing runner config, runner configuration environment application before the Anemoi imports, `downscaling_unified_runner` registration, runner creation, model-interface and `data_indices` handoff, and model parallel rank/group metadata. Extend `PredictionConfig` defaults coverage for the new backend fields.

**Step 2: Run the focused tests to confirm red**

Run: `source /home/ecm5702/dev/runtimes/ds-260612/activate.sh && pytest -q eval/predict/tests/test_unified_runner.py eval/predict/tests/test_types.py`

Expected: unified-loader tests fail because the loader and configuration fields do not exist; existing type tests still pass.

### Task 2: Implement a lazy unified-runner loader and backend dispatch

**Files:**

- Create: `eval/predict/unified_runner.py`
- Modify: `eval/predict/model_loader.py`
- Modify: `eval/predict/types.py`

**Step 1: Implement configuration validation and initialization**

Add `inference_backend` (`legacy` by default) and optional `runner_config` to `PredictionConfig`. Implement `load_unified_runner(config)` so it requires a readable Anemoi config, parses its mapping, applies its `env` keys before Anemoi/model imports, imports the site registration module and `create_runner` lazily, and constructs `create_runner(DotDict(raw_config))`.

**Step 2: Preserve the existing bundle-loop interface**

Return the runner model interface as `inference_model`, a small metadata adapter containing `data_indices` as `datamodule`, parsed `development_hacks.extra_args`, runner device, model communication group, and rank metadata. Reject a runner that does not expose the required model interface/data indices with a clear error.

**Step 3: Dispatch without contaminating unified startup**

Keep the existing direct model loading in a legacy-only function with its current behavior. Make `load_inference_model` dispatch on `config.inference_backend`; do not import manual direct loader utilities at module import time.

**Step 4: Run focused tests**

Run: `source /home/ecm5702/dev/runtimes/ds-260612/activate.sh && pytest -q eval/predict/tests/test_unified_runner.py eval/predict/tests/test_types.py`

Expected: all focused tests pass.

### Task 3: Expose the backend safely in the CLI

**Files:**

- Modify: `eval/predict/main.py`

**Step 1: Add explicit CLI options**

Add `--inference-backend {legacy,unified}` and `--runner-config`. Require `--runner-config` for the unified backend. Keep legacy CLI defaults and checkpoint resolution semantics intact.

**Step 2: Avoid premature legacy imports**

Move direct manual-only checkpoint/parallel helpers behind legacy-only functions. Use lightweight environment rank discovery before output setup so unified startup can apply the runner environment before model initialization.

**Step 3: Verify parser/config behavior**

Run: `source /home/ecm5702/dev/runtimes/ds-260612/activate.sh && python -m eval.predict.main --help | rg 'inference-backend|runner-config' && pytest -q eval/predict/tests/test_unified_runner.py eval/predict/tests/test_types.py`

Expected: help exposes both options and the focused tests pass.

### Task 4: Run static and regression checks

**Files:**

- Verify only the files above plus this plan have changed.

**Step 1: Compile the changed prediction package**

Run: `source /home/ecm5702/dev/runtimes/ds-260612/activate.sh && python -m compileall -q eval/predict`

**Step 2: Run the modular prediction test suite**

Run: `source /home/ecm5702/dev/runtimes/ds-260612/activate.sh && pytest -q eval/predict/tests`

Expected: all modular tests pass. Record, but do not repair, the known unrelated legacy `manual_inference/tests/test_prediction.py` fixture/API failures from baseline.

### Task 5: Generate run-scoped smoke launch assets

**Files:**

- Create under `/home/ecm5702/scratch/eval/manual_unified_repair_b785bf12_e080_s374868_<timestamp>/`: generated Anemoi configs, run manifest, job logs, and output roots.
- Create under `/home/ecm5702/dev/jobscripts/submit/20260624/`: a disposable 4-GPU smoke submit script and two generated launch scripts.

**Step 1: Derive, do not alter, a runner config**

Use the existing successful PrepML BF16/FP32 smoke configurations only as immutable templates. Render per-precision Anemoi configs with `prepml model-config`, retain `downscaling_unified`, edge sharding, `ANEMOI_INFERENCE_NUM_{CHUNKS,PROCESSOR,MAPPER}=1`, one-thread CPU controls, expandable CUDA segments, and enabled NCCL watchdog.

**Step 2: Submit only the BF16 and FP32 4-GPU manual smoke jobs**

Use the repaired code worktree and a one-date/one-lead/two-member truth-aware scope. Do not submit PrepML, a full run, evaluation jobs, or a scoreboard update.

**Step 3: Record job IDs before monitoring**

Append an in-progress entry in the canonical runs monitor, including the code branch/commit, run root, settings, and the two job IDs.

### Task 6: Verify the repair gate and document result

**Files:**

- Modify after terminal results: `/home/ecm5702/dev/docs/docs/runs-monitoring/in-progress/runs.md`
- Modify after terminal results: `/home/ecm5702/dev/docs/epics/checkpoint-eval-pipeline/in-progress/20260624_b785bf12_manual_prepml_parity_repair.md`

**Step 1: Inspect Slurm, logs, and artifacts**

For each smoke, require a completed 4-GPU job, a NetCDF prediction with exactly two members and the ten requested weather states, and finite `y_pred`, `y`, and `x_interp`. Reject OOM, dataset-path, NCCL model-collective, or rank-0 write failures.

**Step 2: Compare speed and memory**

Record elapsed time and peak GPU memory for BF16 and FP32. BF16 must remain within the established 18.41-GiB-class unified topology rather than the direct path's 38.81-GiB OOM footprint.

**Step 3: Update evidence without broadening scope**

Record the terminal results in the relevant epics and runs monitor. Only declare a safe full-run configuration if both smoke jobs pass; otherwise leave the full campaign gated.

