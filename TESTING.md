# Testing guide

The command is `python -m pytest`, run from the repository root with the certified
interpreter (`env -u PYTHONPATH ~/dev/.ds-260612/bin/python`). It collects every test
folder of the repository, on CPU, and should end with no failing test. A test that is
expected not to pass is marked `xfail` with a reason, and the table below lists each one,
so a red test means something is broken.

Result on 2026-09-28 (branch `refactor/evalcli-structure-20260928`):
`798 passed, 76 skipped, 24 xfailed, 0 failed` in about three minutes (`python -m pytest -m "not gpu"`, batch job on the `nf` queue). The 76 skips are the GPU tests, the golden-data tests whose data is absent, and the legacy ds tests that run in a subprocess instead (see "The legacy tree").

## Commands

- Whole repository, CPU only:
  - `python -m pytest -m "not gpu"`
- One folder or file: give the path, for example `python -m pytest eval/tests/test_cli_discover.py`.
- GPU suite:
  - `python -m pytest -m gpu --run-gpu`
- GPU debug mode:
  - `python -m pytest -m gpu --run-gpu --gpu-debug`
- As a batch job, because the suite takes about three minutes and should not run on a login
  node. There is no `pytest-timeout` in the environment, so wrap the call in the shell
  `timeout`:

  ```bash
  #!/bin/bash
  #SBATCH --qos=nf
  #SBATCH --cpus-per-task=8
  #SBATCH --mem=48G
  #SBATCH --time=01:00:00
  cd <repository root>
  timeout 3000 env -u PYTHONPATH ~/dev/.ds-260612/bin/python -m pytest -q -p no:cacheprovider
  ```

## Test layout

`pytest.ini` lists the four top-level places that hold tests. Most tests sit next to the
code they test.

| Folder | What it tests |
|---|---|
| `eval/tests/` | The evaluation framework as a whole: command line, configuration, discovery, evaluator registry and contract, renderer, plotting helpers |
| `eval/evaluators/*/tests/` | One evaluator, including its `core/` |
| `eval/shared/tests/`, `eval/shared/*/tests/`, `eval/tools/*/tests/` | The shared helpers and each tool |
| `eval/predict/tests/`, `eval/scoreboard/tests/` | Prediction generation and the scoreboard aggregation |
| `eval/tests/test_backends_forwarders.py` | The forwarding modules left in `eval/_backends/` |
| `eval/jobs/tests/` | The job orchestration and the maintenance commands of `eval/jobs/` |
| `manual_inference/tests/` | Inference and input construction (CPU and GPU) |
| `manual_inference_legacy_ds/tests/` | The frozen legacy copy of the same package, see below |
| `tools/aifsens2_regen/datasets/` | The combined-view test of the AIFS ensemble regeneration toolkit |

`eval/_quarantine/` is never collected. Importlib import mode is set in `pytest.ini`
because several folders have test files with the same name and some have no `__init__.py`.

## Runtime guardrails

- A global pytest hook enforces a hard maximum of 30 minutes per test.
- GPU tests are tagged with `@pytest.mark.gpu` and are skipped unless `--run-gpu` is given.
- A few tests skip themselves when data they need is absent, for example the golden
  verification tests, which need a run directory under `/home/ecm5702/perm/eval`.

## The legacy tree

`manual_inference_legacy_ds/` is written to be installed under the name `manual_inference`;
its modules import each other by that name. In a normal session `manual_inference` is the
unified package, so the tests of the legacy tree are skipped there (`conftest.py` in that
folder). One test, `test_legacy_suite_as_manual_inference.py`, runs the whole folder in a
subprocess with the legacy tree aliased as `manual_inference`, and every one of them passes
there. To run them by hand, make a folder that contains a symbolic link `manual_inference`
pointing at `manual_inference_legacy_ds`, put that folder on `PYTHONPATH`, and run pytest on
`<folder>/manual_inference/tests`.

## What was red on 2026-09-28, and what happened to each test

On `origin/main` at `bf24cbe`, 74 tests failed and 6 test modules failed to import, 80 ids
in all. The table gives the disposition of each group. None is left failing.

| Ids | Where | What was wrong | Disposition |
|---|---|---|---|
| 7 | `manual_inference_legacy_ds/tests/test_prediction.py` | Ran against the unified package because the legacy tree was not installed as `manual_inference` | Fixed: run in a subprocess with the alias (see above) |
| 4 | `eval/tests/test_cli_dry_run.py` | The subprocess tests ran the live checkout (fixed path) and expected old behaviour: `--only quaver` is now accepted, `predict` on AC is allowed for `o320_o1280`, rank 0 of a one-task step may prepare bundles, truthless bundles only warn | Fixed: tests updated, path is now the tree under test |
| 1 | `eval/tests/test_pipeline.py` | The split evaluator jobs are chained one after another, not all after predict | Fixed |
| 1 | `eval/tests/test_prepare_builder.py` | A bundle without `target_hres_*` is kept with a warning (non-blocking policy) | Fixed |
| 1 | `eval/tests/test_renderer.py` | `atos_ac` has a new venv; the test named the old path | Fixed: reads the host file |
| 1 | `eval/jobs/tests/test_finalize_lean_eval_layout.py` | The template reads `RUN_ROOT` and `RUN_ID` from the environment; the test edited text that is no longer there | Fixed |
| 3 | `eval/jobs/tests/test_materialize_x_interp_reference.py` | The script moved to `eval/archive/jobs/` | Fixed: path repointed |
| 16 | `eval/jobs/tests/test_predictions_jobs.py` | The scripts moved to `eval/archive/jobs/` | Fixed: paths repointed |
| 1 + 1 | `eval/jobs/tests/test_checkpoint_profile.py`, `test_o1280_o2560_bundle_preflight.py` | Import archived jobs from `eval.jobs`; the archived preflight also imported a moved sibling | Fixed: imports repointed, and one stale import inside the archive repaired |
| 2 | `eval/tests/test_mlflow_loader.py`, `test_mlflow_plot.py` | Loaded `mlflow/loader.py` and `plot.py`, which moved into `eval/evaluators/mlflow/` | Fixed: path repointed |
| 1 | `eval/evaluators/zoom_maps/tests/test_member_wind_maps.py` (called `membermaps` until 2026-09-29) | The variable spec gained a `fine_vmax` key | Fixed |
| 4 | `eval/predict/tests/test_dataset_builder.py` | `eval.predict` now also imports `manual_inference.prediction.predict`; the stub did not provide it | Fixed: the stub provides it |
| 15 | `eval/jobs/tests/test_scoreboard_metrics.py` | Tests of the anchored TC scores (reach, ENFO match, tail ratios), removed by the raw-extremes contract of 2026-06-21 | Moved to `eval/_quarantine/20260928/jobs/tests/`; three live tests now assert the raw-extremes contract |
| 4 + 4 + 1 | `test_predictions_dir_spectra.py`, `test_spectra_plot_pdf.py`, one test in `test_predictions_jobs.py` | Test helpers that were deleted when the spectra templates were archived | Moved to the quarantine (recoverable from git history) |
| 7 | `manual_inference/tests/test_prediction.py` | The test doubles model the old tuple batches and `predict_step(x_l, x_h)`; predict now reads dict batches (`ds.data['in_lres']`) and `find_missing_explicit_hres_inputs` moved to `input_data_construction.bundle` | xfail. To fix: rewrite the doubles for the dict-batch API |
| 2 | `eval/tests/test_plotting_metadata.py` | Expects the old debug-style title; `to_title()` now gives a readable one (commit `4a835bc`) | xfail. Update on the plotting restyle branch |
| 2 | `eval/tests/test_region_plotting_custom_o1280.py` | One expects the old debug-style title; the other expects a `<region>.pdf` and `.png` per region, but only `all_regions_plots.pdf` is written now | xfail. Update on the plotting restyle branch |
| 1 module (3 tests) | `eval/tests/test_mechanistic_compare_v1.py` | The backend imports `_validate_bundle_hres_contract` from `manual_inference.prediction.predict`, which no longer has it. Its evaluator was retired; the backend is dead code | xfail. Candidate for the quarantine |
| 1 module (9 tests) | `eval/tests/test_plot_intermediate.py` | The venv's anemoi-models has no `apply_shard_shapes`, so the diffusion-sampler import fails | xfail. Environment: needs the newer anemoi-models |

Counts: 43 ids fixed, 24 moved to the quarantine, 13 ids marked xfail (23 individual xfail
results, plus one older xfail in `test_tc_structure.py`), 0 still failing.

## Adding or changing a test

- Put it next to the code, in a `tests/` folder of that package.
- Do not hard-code a checkout path. Use `Path(__file__).resolve().parents[N]` for the
  repository root, so the test exercises the tree it belongs to.
- If a test cannot pass for a reason you understand and will not fix now, mark it
  `@pytest.mark.xfail(reason="one line saying what is wrong", strict=False)` and add a row
  to the table above. Do not leave it red.
