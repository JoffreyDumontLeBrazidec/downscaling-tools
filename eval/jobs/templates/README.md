# Strict Manual-Inference And Eval Templates

This directory is the canonical home for repo-specific eval and manual-inference templates used by `downscaling-tools`.
If copies exist under `jobscripts/`, treat them as mirrors, not the source of truth.
This index lists only the templates that are actually present in this directory.

**Before submitting any job**, source the preflight script or run it standalone:
```bash
# Standalone check
bash eval/jobs/templates/preflight_eval_check.sh --predictions-dir /path --stack new --job-type predict75

# Or source from within a template
source preflight_eval_check.sh
preflight_cluster
preflight_venv "new"
preflight_predictions_dir "/path/to/predictions" 25
preflight_summary
```

## Template Matrix

### Inference
- `submit_aug26_30_scoreboard_flow.sh`
  - Canonical login-node helper for the full Aug 26-30 production chain.
  - Renders run-local copies of the inference, sigma, and post-writer templates under `/home/ecm5702/dev/jobscripts/submit/<YYYYMMDD>/`.
  - Those rendered copies are disposable submission artifacts, not the canonical templates.
  - Submits sigma eval, full25 inference, then the scoreboard post-writer with `afterok` dependencies.
- `strict_manual_predict_x_bundle.sbatch`
  - Batch generation of `predictions_YYYYMMDD_stepXXX.nc` from any chosen bundle-date set.
  - Works for `new` and `old` stack.
  - Enforces fresh run folder and explicit lane/input compatibility.
  - New stack path keeps `y` mandatory in output files.
  - Writes `${RUN_ROOT}/EXPERIMENT_CONFIG.yaml` so scoreboard reporting can recover exact scope and sampler settings.
- `strict_manual_predict_one_bundle.sbatch`
  - Single-bundle debug/proof launcher for the same prediction code path.
  - Uses the same repo-owned checkpoint profiling and sampler normalization as the x-bundle launcher.
  - Writes `${RUN_ROOT}/EXPERIMENT_CONFIG.yaml` for downstream reporting.
- `predict_recovery.sbatch`
  - **Recovery for walltime-killed prediction runs.**
  - Auto-detects missing prediction files and relaunches only remaining date/step combos.
  - Targets an existing run directory and reruns only the missing files.
  - Works for `new` and `old` stack, AC and AG.

### TC Evaluation
- `tc_eval_from_predictions.sbatch`
  - **Canonical TC evaluation template — works on both AC and AG.**
  - Generates normalized TC PDF plots from `predictions_*.nc`.
  - Uses `eval/tc/plot_pdf_tc_from_predictions.py` from this repo.
  - Supports `native` and `regridded` modes (`regridded` requires metview, AC-only).
  - Checks TC reference data exists; points to request script if missing.
  - Preferred submit helper:
    - `eval/jobs/templates/submit_tc_eval_from_predictions.sh`
    - default auto profile: `ac -> qos=nf with no GPU request`, `ag -> qos=ng with --gpus-per-node=0`
    - leaves template memory/time untouched unless `TC_SUBMIT_MEM_OVERRIDE` or `TC_SUBMIT_TIME_OVERRIDE` is set

### Spectra
- `scoreboard_write_from_predictions.sbatch`
  - Canonical full-scoreboard post-writer for Aug 26-30 runs.
  - Produces spectra, TC, weighted surface loss, scoreboard metrics JSON, and refreshes scoreboard markdown.
  - Full-route spectra now use the canonical proxy10 subset rather than the older step120-only five-date slice.
- `scoreboard_sigma_eval.sbatch`
  - Canonical sigma-evaluator launcher for the Aug 26-30 full scoreboard chain.
  - Uses the repo root dynamically instead of a hard-coded checkout path.
- `spectra_ecmwf_from_predictions.sbatch`
  - **Canonical ECMWF spectra template — AC-only (requires gptosp.ser).**
  - Follows the same 3-stage pipeline as `eval/spectra/grb_to_spectra.sh`:
    1. Stage predictions as nopoles GRIBs
    2. `gptosp.ser` → spectral harmonics
    3. `compute_spectra-3.py` → spectra amplitudes
  - Resumable: skips already-completed `gptosp` transforms on resubmission.
  - Uses short `$TMPDIR` symlinks to avoid `gptosp` path-length truncation.
- `predictions_dir_spectra.py`
  - Helper used by `launch_proxy_eval.sh` for proxy scoreboard spectra artifacts.
- `stage_prediction_spectra_gribs.py`
  - Repo-owned helper for staging prediction NetCDF files into ECMWF-style no-poles GRIBs.
  - Used by `scoreboard_write_from_predictions.sbatch` for the trusted full-scoreboard spectra route.

### Submission Helper
- `submit_tc_eval_from_predictions.sh`
  - Login-node helper for standalone TC reruns on an existing predictions tree.
  - Renders a host-safe copy of `tc_eval_from_predictions.sbatch` under `/home/ecm5702/dev/jobscripts/submit/<YYYYMMDD>/`.
  - Default `auto` profile chooses:
    - `ac_cpu_safe` on AC (`qos=nf`, drops the GPU request)
    - `ag_cpu_safe` on AG (`qos=ng`, sets `--gpus-per-node=0`)
  - Optional env overrides:
    - `TC_SUBMIT_MEM_OVERRIDE=128G`
    - `TC_SUBMIT_TIME_OVERRIDE=04:00:00`
    - `TC_SUBMIT_HOLD=1`
    - `TC_SUBMIT_NO_SUBMIT=1` to render only

### Preflight
- `preflight_eval_check.sh`
  - **Source this before any submission** to validate cluster, venv, data, QOS, and walltime.
  - Also works standalone: `bash preflight_eval_check.sh --help`.

## Canonical Upstream Tools

These are the authoritative eval tools in `downscaling-tools`. Templates wrap them;
do not reimplement their logic in ad-hoc scratch scripts.

| Tool | Path | Purpose |
|------|------|---------|
| TC plotting | `eval/tc/plot_pdf_tc_from_predictions.py` | TC normalized PDF plots from predictions |
| TC data request | `eval/tc/all_events_request.sh` | MARS request for TC reference GRIBs (edit EXPID) |
| Spectra pipeline | `eval/spectra/grb_to_spectra.sh` | Full MARS→gptosp→compute spectra pipeline |
| Spectra compute | `/home/ecm5702/dev/post_prepml/spectra/spectra_ml/individual_files/compute_spectra-3.py` | Spectral harmonics → amplitude spectra |

## Design Invariants
- Manual-inference templates keep host/env resolution explicit and validated:
  - `ac + new -> /home/ecm5702/dev/.ds-dyn/bin/activate`
  - `ag + new -> /home/ecm5702/dev/.ds-ag/bin/activate`
  - `ac + old -> /home/ecm5702/dev/.ds-old/bin/activate`
  - `ag + old -> /home/ecm5702/dev/.ds-ag-old/bin/activate`
- Eval templates detect the cluster automatically and activate the correct env.
- Output paths stay explicit:
  - users set `RUN_ROOT`, `RUN_ID`, and optional output tags directly
  - templates write into the chosen run tree, not into hidden temp locations

## Resource Baselines (Empirical)

These are based on observed job outcomes from the checkpoint-eval-pipeline epic.

| Job Type | Walltime | Memory | QOS | Notes |
|----------|----------|--------|-----|-------|
| predict25 (O320, GPU) | 12:00:00 | default | ng | 25 files, single GPU |
| predict75 (O320, GPU) | 48:00:00 | default | ng | 61/75 in 12h observed; use 48h |
| predict25 (O1280, GPU) | 24:00:00 | default | ng | ~2-3x slower than O320 |
| TC eval (O320) | 04:00:00 | 128G | ng | Cross-cluster default uses `gpus-per-node=1` for submission compatibility |
| TC eval (O1280) | 06:00:00 | 256G | ng | O1280 five-date ten-member |
| Spectra ECMWF (AC) | 48:00:00 | 128G | nf | 300 gptosp transforms; resumable |
| Spectra proxy (AG) | 08:00:00 | 128G | ng | healpy-based, CPU-only |

## Cluster / QOS Rules

| Cluster | Available QOS | Notes |
|---------|---------------|-------|
| AC | `nf` (CPU), `np`, `ef`, `ng` (GPU) | Use `nf` for CPU eval jobs; the TC submit helper defaults to this on AC |
| AG | `ng` only | Use `--gpus-per-node=0` for CPU-only; `nf` does NOT work on AG |

Common mistakes to avoid:
- Requesting `qos=nf` on AG → immediate rejection.
- Requesting `qos=ng` + `--gpus-per-node=0` on AC → can fail with `QOSMinGRES`; prefer `qos=nf` with no GPU request.
- Omitting `--gpus-per-node=0` on AG → Slurm auto-adds a GPU.
- Requesting `>128G` on AC `nf` → may hit `QOSMaxMemoryPerJob`.

## How To Use
1. Copy a template from this folder.
2. Edit only the `USER SETTINGS` block.
3. Optionally run `bash preflight_eval_check.sh` to validate before submitting.
4. Submit with `sbatch <template>.sbatch`.

For the canonical Aug 26-30 production bundle with minimal manual steps, prefer:
edit and submit `submit_aug26_30_scoreboard_flow.sh`.

For standalone TC reruns on an existing predictions tree, prefer:
`bash eval/jobs/templates/submit_tc_eval_from_predictions.sh /path/to/edited_copy.sbatch`

## Smooth Routes By Goal
- Quick checkpoint screen (`10` bundles):
  - `codex_eval_predictions --ckpt-id <ID>`
- Promote a passing run to full250:
  - `codex_eval_predictions --ckpt-id <ID> --run-id <same_run_id> --continue-full`
- Manual full250 fallback:
  - edit `submit_aug26_30_scoreboard_flow.sh`
- Recovery:
  - edit `predict_recovery.sbatch`
- Standalone TC eval on existing predictions:
  - edit `tc_eval_from_predictions.sbatch`, then run `bash eval/jobs/templates/submit_tc_eval_from_predictions.sh /path/to/edited_copy.sbatch`

## Notes
- Resolver tests for the strict inference templates live in:
  - `/home/ecm5702/dev/downscaling-tools/eval/jobs/tests/test_checkpoint_profile.py`
