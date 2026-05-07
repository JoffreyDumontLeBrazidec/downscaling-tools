# Eval

Evaluation utilities that consume `predictions_YYYYMMDD_stepNNN.nc` files produced
by `eval.predict.main` or by the unified CLI.

## Canonical CLI: `eval.cli`

The unified CLI is the primary interface for all evaluation operations:

```bash
python -m eval.cli <subcommand> [args]
```

### Subcommands

**Full pipeline** (predict + evaluate + scoreboard):
```bash
python -m eval.cli run \
    --lane o96_o320 \
    --checkpoint /path/to/checkpoint \
    --host atos_ac
```

**Evaluate existing predictions**:
```bash
python -m eval.cli evaluate \
    --predictions-dir /path/to/predictions/ \
    --lane o96_o320 \
    --checkpoint /path/to/checkpoint
```

**Run a single evaluator**:
```bash
python -m eval.cli evaluate \
    --predictions-dir /path/to/predictions/ \
    --lane o96_o320 --only surface
```

**Generate scoreboard from completed evaluation**:
```bash
python -m eval.cli scoreboard \
    --eval-dir /path/to/eval/output/ \
    --lane o96_o320
```

Use `--dry-run` on any subcommand to print the resolved config as JSON.

Use `--include-diagnostics` to run the diagnostics group (sigma, mechanistic, intermediate) in addition to defaults.

## Pipeline Generation

Generate HPC sbatch chains with SLURM dependency chaining:

```bash
python -m eval.jobs.pipeline \
    --lane o96_o320 --host atos_ac \
    --checkpoint /path/to/checkpoint \
    --output-dir /path/to/pipeline/scripts/
```

This produces `01_predict.sbatch`, `02_eval_*.sbatch`, `03_scoreboard.sbatch`, and a `submit_pipeline.sh` launcher with `--dependency=afterok` chaining.

## Evaluator Architecture

Evaluators live in `eval/evaluators/<name>/` and follow a wrapper pattern:
- **tc, surface**: Native Python implementations
- **spectra, sigma, region_plot**: Subprocess wrappers around legacy modules
- **mechanistic, intermediate**: Stubs (not yet implemented)

Each evaluator exports `EVALUATOR_SPEC`, `run()`, `score()`, and optionally `plot()`.

Lane configuration: `eval/config/lanes/<lane>.yaml`
Host configuration: `eval/config/hosts/<host>.yaml`

## Notebooks
- `eval/notebooks/00_eval_overview.ipynb`
- `eval/notebooks/01_unified_runner.ipynb`
- `eval/notebooks/02_intermediate_plots.ipynb`
- `eval/notebooks/03_region_plotting.ipynb`
- `eval/notebooks/04_sigma_evaluator.ipynb`
- `eval/notebooks/05_quaver.ipynb`
- `eval/notebooks/06_spectra.ipynb`
- `eval/notebooks/07_tc.ipynb`

## Prediction Generation (New)

The `eval/predict/` package provides modular prediction generation from input bundles,
replacing the monolithic `generate_predictions_25_files.py`:

```bash
python -m eval.predict.main \
  --input-root /path/to/bundles \
  --out-dir /path/to/output \
  --name-ckpt /path/to/checkpoint.ckpt \
  --dates 20230826,20230827,20230828,20230829,20230830 \
  --steps 24,48,72,96,120 \
  --members 1,2,3,4,5,6,7,8,9,10
```

**Key improvements over `generate_predictions_25_files.py`:**
- Correct date metadata in output files (fixes the `date=0` / "1970-01-01" bug)
- Modular architecture (bundle discovery, model loading, inference, output writing)
- Schema validation for output files
- CF-compliant time metadata

See [`eval/predict/README.md`](predict/README.md) for full documentation.

## Intermediate Diffusion Trajectory Plots
New wrapper for visualizing denoising/sampling intermediate states (outside `anemoi-core`):

From checkpoint (generate intermediates + plot):
```bash
python -m eval.plot_intermediate.plot_intermediate checkpoint \
  --name-ckpt <RUN_ID_or_ckpt_path> \
  --member 0 \
  --sample 0 \
  --idx 0 \
  --weather-state 2t \
  --out /tmp/intermediate_2t.png
```

From existing dataset with `inter_state` variable:
```bash
python -m eval.plot_intermediate.plot_intermediate dataset \
  --predictions-nc /path/to/predictions_with_intermediate.nc \
  --sample 0 \
  --weather-state 2t \
  --out /tmp/intermediate_2t.png
```

Preset region-style intermediate panel plotting (keeps the validated ecm5702 layout):
```bash
python -m eval.region_plotting.plot_intermediate_presets \
  --predictions-nc /home/ecm5702/scratch/eval/manual_o320r2/eval/intermediate_bundle_idalia_strong/eefo_o96_0001_date20230826_time0000_mem06_step048h__intermediate_cached.nc \
  --region idalia_center \
  --weather-states 10u,10v,2t,msl \
  --ordered-steps 16,14,13,12,11 \
  --include-sigma-labels \
  --style amazon-baseline \
  --out /home/ecm5702/scratch/eval/manual_o320r2/eval/intermediate_bundle_idalia_strong/readability_v2/idalia_center_baseline.pdf
```

Minimal frame-change variant (`pcolormesh + contour` while keeping the baseline panel structure):
```bash
python -m eval.region_plotting.plot_intermediate_presets \
  --predictions-nc /home/ecm5702/scratch/eval/manual_o320r2/eval/intermediate_bundle_idalia_strong/eefo_o96_0001_date20230826_time0000_mem06_step048h__intermediate_cached.nc \
  --region idalia_center \
  --weather-states 10u,10v,2t,msl \
  --ordered-steps 16,14,13,12,11 \
  --include-sigma-labels \
  --style minimal-pcolor-contour \
  --out /home/ecm5702/scratch/eval/manual_o320r2/eval/intermediate_bundle_idalia_strong/readability_v2/idalia_center_minimal_contour.pdf
```

## Evaluation Modules
- `eval/predict` (**new** — modular prediction generation with proper date handling)
- `eval/region_plotting` (local region plots — refactored with shared `plotting/` utilities)
- `eval/sigma_evaluator` (sigma sweeps / tables)
- `eval/quaver` (quaver workflows)
- `eval/spectra` (spectral analysis)
- `eval/tc` (tropical cyclone evaluation)

## Deprecated
- `eval/jobs/generate_predictions_25_files.py` — use `python -m eval.predict.main` instead

Canonical one-date non-TC local plots:
```bash
python -m eval.region_plotting.plot_one_date_local \
  --run-root /home/ecm5702/scratch/eval/<RUN_ID> \
  --date 20230826
```
