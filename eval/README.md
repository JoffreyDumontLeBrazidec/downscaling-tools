# Eval

Evaluation utilities that consume `predictions.nc` produced by `manual_inference`
or by prepml/anemoi-inference experiment outputs.

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

## Legacy Modules
- `eval/region_plotting` (local region plots)
- `eval/sigma_evaluator` (sigma sweeps / tables)
- `eval/quaver` (quaver workflows)
- `eval/spectra` (spectral analysis)
- `eval/tc` (tropical cyclone evaluation)
