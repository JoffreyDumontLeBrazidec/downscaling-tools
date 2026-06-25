# Eval

Evaluation utilities that consume `predictions_YYYYMMDD_stepNNN.nc` files produced
by `eval.predict.main` or by the unified CLI.

## Canonical CLI: `eval.cli`

The unified CLI is the **required** interface for all evaluation operations:

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

**Local spread/CRPS diagnostics**:
```bash
python -m eval.cli evaluate \
    --predictions-dir /path/to/predictions/ \
    --lane o96_o320 --only probabilistic
```

The `probabilistic` evaluator reads local `predictions_YYYYMMDD_stepNNN.nc` files directly and writes `scores_by_lead.csv`, `summary_by_lead.csv`, `metrics.json`, and `plots/probabilistic_scores.pdf` under `evaluators/probabilistic/`. It computes CRPS, fair CRPS, spread, and ensemble-mean RMSE by lead time, variable, and domain without writing forecasts or scores to FDB. Quaver comparison curves can be overlaid from an exported local CSV, but quaver is not part of the evaluator runtime.

To compare the local summary with a quaver-exported CSV using the same `step,weather_state,domain,metric` keys:

```bash
python -m eval.jobs.compare_probabilistic_reference \
    --local-summary /path/to/evaluators/probabilistic/summary_by_lead.csv \
    --reference-summary /path/to/quaver_reference.csv \
    --out-dir /path/to/comparison/
```

For quaver references that already exist in the score DB, export a CSV first:

```bash
module load quaver
export TMPDIR=/path/to/scratch/tmp
quaver eval/jobs/export_quaver_probabilistic_reference.py \
    --out-csv /path/to/quaver_reference.csv
```

Validation caveat: quaver surface scores are station-observation/FDB-backed, while the local evaluator scores gridded `y` embedded in prediction NetCDFs. Numeric parity therefore requires the same forecast, target, domain, variables, and member set. If quaver has no rows for the ML expver, the comparison can still overlay available ENFO/EEFO reference curves, but it is not model-vs-model parity.

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
- **tc, surface**: Native Python implementations wrapping legacy kernels
- **spectra, sigma, region_plot**: Subprocess wrappers around legacy modules
- **mechanistic, intermediate**: Stubs (not yet implemented)

Each evaluator exports `EVALUATOR_SPEC`, `run()`, `score()`, and optionally `plot()`.

Lane configuration: `eval/config/lanes/<lane>.yaml`
Host configuration: `eval/config/hosts/<host>.yaml`

### TC tail-extreme ratios (AN-anchored)

The TC evaluator emits four ratio metrics per event + four aggregates, all anchored
to the **embedded** OPER analysis row in each stats JSON (not the canonical YAML, so
support_mode / bbox / member-clip stay consistent with everything else in the run):

| metric | meaning |
|---|---|
| `tc_<event>_mslp_p001_ratio` | depth(model.mslp_p001) / depth(AN.mslp_p001) |
| `tc_<event>_mslp_min_ratio`  | depth(model.mslp_min)  / depth(AN.mslp_min)  |
| `tc_<event>_wind_p9999_ratio`| model.wind_p9999 / AN.wind_p9999             |
| `tc_<event>_wind_max_ratio`  | model.wind_max   / AN.wind_max               |
| `tc_mean_<key>_ratio`        | mean of the per-event values                 |

AN row = 1.0 by construction. `>1` means ML reaches deeper minima / stronger winds
than the analysis; `<1` means weaker.

The percentile fields `mslp_p001` (0.01 percentile) and `wind_p9999` (99.99 percentile)
were added to `extreme_tail_table` alongside the existing `mslp_p1`/`p01`/`min` and
`wind_p99`/`p999`/`max` — every TC eval run produces them by default. For stats JSONs
generated before the change, run the one-shot backfill:

```
python -m eval.jobs.backfill_tc_extreme_percentiles --lane o96_o320
# Use --check-only first to see what's missing.
# Idempotent and atomic per file.
```

The scoreboard generators (`eval.cli scoreboard` generic CSV, and the custom
`eval.jobs.generate_enfo_o320_scoreboard`) surface the new columns automatically once
the underlying stats JSONs have the fields.

## Backends (`eval/_backends/`)

Internal implementation details of the evaluator wrappers. **Never invoke directly.**

Contains: `tc/`, `spectra/`, `region_plotting/`, `sigma_evaluator/`, `weight_diagnostics/`,
`plot_intermediate/`, `quaver/`, and `scoreboard/{tc,spectra,surface,_surface_compute,_utils,canonical_data,row_matching}.py`.

These modules were moved here from their original top-level `eval/` locations as part of the
legacy quarantine. All imports have been updated. Old import paths (`eval.tc.*`, `eval.spectra.*`,
etc.) will fail immediately — this is intentional.

## Notebooks
- `eval/notebooks/00_eval_overview.ipynb`
- `eval/notebooks/01_unified_runner.ipynb`
- `eval/notebooks/02_intermediate_plots.ipynb`
- `eval/notebooks/03_region_plotting.ipynb`
- `eval/notebooks/04_sigma_evaluator.ipynb`
- `eval/notebooks/05_quaver.ipynb`
- `eval/notebooks/06_spectra.ipynb`
- `eval/notebooks/07_tc.ipynb`

## Prediction Generation

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

See [`eval/predict/README.md`](predict/README.md) for full documentation.

## Archive (`eval/archive/`)

Contains retired scripts and old templates. Not used in live workflows.
