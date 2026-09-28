# eval/jobs/scripts

One-off scripts and experiment suites. Nothing in the framework imports them, so
they are safe to move or retire, but their names appear in old task notes in the
project docs, which is why they are kept here and not in `eval/_quarantine/`.
Run them by path from the repository root, for example
`python eval/jobs/scripts/ag_crps_probe.py --help`.

- `ag59e4_screen.py` runs a paired control-against-autoguidance screen for
  checkpoints of the 59e4 class on the `interp` machinery.
- `ag_crps_probe.py` measures CRPS, spread and ensemble-mean RMSE for autoguidance with
  one fixed input and several diffusion seeds.
- `build_tc_o320_o1280_regional_predictions_from_dataloader.py` builds
  `predictions_*.nc` files for the regional o320-to-o1280 tropical cyclone harness
  from the native regional dataloader.
- `export_quaver_probabilistic_reference.py` exports quaver probabilistic score curves
  to CSV. Run it through the `quaver` binary, as its docstring explains.
- `validation/` holds a five-experiment checkpoint validation suite
  (`run_validation_suite.sh`, the `val_exp*.sbatch` files and `validate_checkpoint.py`).
  The launcher script has the path of the live checkout written into it.
