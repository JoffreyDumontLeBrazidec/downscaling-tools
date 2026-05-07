# downscaling-tools

Evaluation, manual-inference, and HPC-orchestration layer for anemoi-core
diffusion downscaling experiments. Consumes checkpoints and predictions, produces
scoreboards, plots, and metrics across multiple evaluation pillars (TC extremes,
power spectra, surface loss, regional plots, sigma sweeps), and orchestrates
multi-stage HPC job chains.

## Canonical CLI

```bash
# Full pipeline: predict + evaluate + scoreboard
python -m eval.cli run --checkpoint <path> --lane o96_o320 [--host atos_ac] [--only tc,spectra]

# Predictions only
python -m eval.cli predict --checkpoint <path> --lane o96_o320

# Evaluate existing predictions
python -m eval.cli evaluate --predictions-dir <dir> --lane o96_o320 [--only tc,spectra,surface]

# Scoreboard from existing evaluation results
python -m eval.cli scoreboard --eval-dir <dir> --lane o96_o320

# Dry run
python -m eval.cli run --checkpoint <path> --lane o96_o320 --dry-run

# Manual inference (single checkpoint)
python -m manual_inference.prediction.predict {from-bundle,from-dataloader,build-bundle}

# Modular prediction generation
python -m eval.predict.main --input-root ... --out-dir ... --name-ckpt ... --dates ... --steps ... --members ...
```

## Subsystems

| Package | Purpose | README |
|---|---|---|
| `eval/` | Evaluation framework (evaluators, scoreboard, config, discovery) | `eval/README.md` |
| `eval/predict/` | Modular date-aware prediction generation | `eval/predict/README.md` |
| `eval/jobs/` | HPC orchestration, sbatch templates, pipeline rendering | `eval/jobs/templates/README.md` |
| `manual_inference/` | Single-checkpoint inference (bundle/dataloader modes) | `manual_inference/README.md` |
| `mlflow/` | MLflow loss-plotting utilities | -- |
| `distributed/` | Multi-GPU distributed helpers | -- |

## Architecture

For the layered design, evaluator architecture, output directory contract, and
HPC orchestration, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Testing

CPU suite: `pytest -m "not gpu"`. GPU suite: `pytest -m gpu --run-gpu`.
See [TESTING.md](TESTING.md).
