# Eval

Evaluation utilities that consume `predictions.nc` produced by `manual_inference`
or by prepml/anemoi-inference experiment outputs.

## Notebooks (Super Simple)
- `eval/notebooks/00_eval_overview.ipynb`
- `eval/notebooks/01_unified_runner.ipynb`
- `eval/notebooks/02_intermediate_plots.ipynb`
- `eval/notebooks/03_region_plotting.ipynb`
- `eval/notebooks/04_sigma_evaluator.ipynb`
- `eval/notebooks/05_quaver.ipynb`
- `eval/notebooks/06_spectra.ipynb`
- `eval/notebooks/07_tc.ipynb`

## Unified Runner
Use the orchestration CLI:
```bash
python -m eval.run <subcommand> [args]
```

For background full-suite orchestration with retries and auto-monitoring, use:
```bash
/home/ecm5702/dev/downscaling-tools/eval/jobs/codex_eval --expver <EXPVER>
```

Predictions + eval flow (from checkpoint):
```bash
/home/ecm5702/dev/downscaling-tools/eval/jobs/codex_eval --run-id <RUN_ID> --ckpt-id <CKPT_ID>
```

All artifacts are stored under:
`/home/ecm5702/perm/eval/<run_name>/`

Each run folder contains at least:
- `predictions.nc`
- `metadata.json`
- region plot outputs (when enabled)
- `sigma_eval_table.csv` (checkpoint mode, when enabled)

### Main Workflows

Evaluate from MARS expver (example requested):
```bash
python -m eval.run mars-expver --expver j24v
```

Evaluate from a checkpoint:
```bash
python -m eval.run checkpoint --name-ckpt <exp_or_ckpt>
```

Evaluate from an existing predictions file:
```bash
python -m eval.run predictions --predictions-nc /path/to/predictions.nc
```

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

## Legacy Modules
- `eval/region_plotting` (local region plots)
- `eval/sigma_evaluator` (sigma sweeps / tables)
- `eval/quaver` (quaver workflows)
- `eval/spectra` (spectral analysis)
- `eval/tc` (tropical cyclone evaluation)
