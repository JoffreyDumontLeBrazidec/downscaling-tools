# Map of `eval/_backends/` before the fold (2026-09-29)

This note records who used each folder of `eval/_backends/` on `origin/main` at commit `dfc824e`, before the folder was dissolved on the branch `refactor/fold-backends-20260929`. It was written first, before any file moved, and the decisions table below follows from it.

## How the map was made

Every Python file of the repository outside `eval/_quarantine/` was parsed with the `ast` module, and each import that reaches `eval._backends` was resolved, including relative imports inside the backends themselves. In addition, `git grep` searched all tracked files for the string `_backends`, which finds three kinds of reference that an import scan cannot see: `python -m eval._backends...` command strings (in Python, shell and sbatch files), file paths written in docstrings and documentation, and paths composed at run time. The composed paths are the two `_BACKEND = ... / "_backends" / "quaver"` constants in `eval/evaluators/quaver/runner.py` (line 30) and `plotter.py` (line 44), and the `MODULE_PATH` of `eval/tests/test_intermediate_sigma_curve.py` (line 12).

The search found no lane YAML, ladder YAML or job template that names a backend module. The only YAML mention is a comment in `eval/config/toolchains.yaml`. The `eval/jobs/templates/` directory has two stale references that were already broken before this change, because they name the pre-quarantine paths `eval/tc/all_events_request.sh` and `eval/spectra/grb_to_spectra.sh` (`README.md` lines 60 and 61, and `preflight_eval_check.sh` line 24).

## What the map shows

Nine folders have exactly one evaluator as their user, or one evaluator and a few helpers of it: `tc`, `probabilistic`, `spread_proxy`, `storm_maps`, `quaver`, `tctracker`, `local_global_parity.py`, and, once the misleadingly named `scoreboard` folder is taken apart, the `surface`, `tc` and `spectra` scoreboard loaders. Five folders serve no live evaluator at all, because the evaluator they belonged to was retired on 2026-09-28 or because they are analysis tools: `sigma_evaluator`, `obs_crps`, `plot_intermediate`, `weight_diagnostics` and the spectra analysis scripts. `videogen` is used only by the `eval.cli videogen` command. Three folders are shared and were split: `precip` (the precipitation truth source is used by four evaluators and by `interp/`), `region_plotting` (three unrelated evaluators, plus two tools that borrow its region helpers) and `scoreboard` (loaders of three different evaluators plus a small shared module).

## Decisions

The convention chosen is the following. The three contract files of an evaluator (`runner.py`, `scorer.py`, `plotter.py`) and their private helpers stay at the top of `eval/evaluators/<name>/`. Everything that used to live in `eval/_backends/` for that evaluator goes into the subpackage `eval/evaluators/<name>/core/`. Its tests go into `eval/evaluators/<name>/tests/`. Code with no evaluator goes into `eval/tools/<name>/`, next to the existing `eval/tools/parity/`. Code used by several evaluators goes into `eval/shared/`, which already exists, so no second shared package is created.

| Old location | Users | New location | Reason |
|---|---|---|---|
| `_backends/tc/` | tc, tc_structure, lane_diagnostics, `eval/jobs/backfill_tc_extreme_percentiles.py`, four `scripts/tc_*.py` | `evaluators/tc/core/` | tc is the clear primary owner; the others import from it |
| `_backends/scoreboard/tc.py`, `row_matching.py`, `canonical_data.py`, `data/` | tc scorer, `eval/jobs/scoreboard_metrics.py` | `evaluators/tc/core/scoreboard.py`, `row_matching.py`, `canonical_data.py`, `data/` | the docstring of `row_matching.py` says it classifies TC stats rows; the canonical data is TC analysis extremes |
| `_backends/scoreboard/surface.py`, `_surface_compute.py` | surface; `_surface_compute` is also imported by probabilistic, spread_proxy and `eval/jobs/scoreboard_surface_loss.py` | `evaluators/surface/core/scoreboard.py`, `compute.py` | the file is named for the surface loss and is unchanged; probabilistic and spread_proxy import three helpers from it |
| `_backends/scoreboard/spectra.py`, `spectra/naming.py`, `spectra/harmonics.py` | spectra_ecmwf_v2, `eval/jobs/scoreboard_metrics.py` | `evaluators/spectra_ecmwf_v2/core/scoreboard.py`, `naming.py`, `harmonics.py` | only spectra_ecmwf_v2 and the scoreboard job use them |
| `_backends/scoreboard/_utils.py` | the three scoreboard loaders, `eval/jobs/scoreboard_metrics.py` | `shared/json_utils.py` | two small generic functions used by several owners |
| `_backends/precip/sources.py` | precip_scores, precip_events, precip_dist, lane_diagnostics, `interp/core/data.py` | `shared/precip/sources.py` | four evaluators and code outside `eval/` use it |
| `_backends/precip/metrics.py`, `score_gribs.py` | precip_scores | `evaluators/precip_scores/core/` | single owner |
| `_backends/precip/tp_histogram_comparison.py` | precip_dist (as a subprocess) | `evaluators/precip_dist/core/` | single owner |
| `_backends/region_plotting/` `local_plotting.py`, `plot_regions.py`, `plot_one_date_local.py`, `plot_intermediate_presets.py`, `plot_tc_contours_from_predictions.py`, `plotting/` | region_plot; the tools `weight_diagnostics` and `plot_intermediate` borrow `get_region_ds` and the region boxes | `evaluators/region_plot/core/` | region_plot is the primary owner |
| `_backends/region_plotting/precip_events.py`, `plot_precip_events.py` | precip_events | `evaluators/precip_events/core/` | single owner |
| `_backends/region_plotting/plot_member_wind_maps.py`, `plot_trajectory_wind_maps.py` | membermaps and `eval.cli membermaps` | `evaluators/membermaps/core/` | single owner; the two files do not import the other region_plotting files |
| `_backends/probabilistic/` | probabilistic, spread_proxy | `evaluators/probabilistic/core/` | spread_proxy is derived from it and imports four helpers |
| `_backends/spread_proxy/` | spread_proxy | `evaluators/spread_proxy/core/` | single owner |
| `_backends/storm_maps/` | storm_maps, `eval/jobs/ladder.py` (a `python -m` string) | `evaluators/storm_maps/core/` | single owner |
| `_backends/quaver/` | quaver (found by path, not by import) | `evaluators/quaver/core/` | single owner; the scripts run under the `quaver` binary |
| `_backends/tctracker/` | tctracks, `eval.cli tctracker` | `evaluators/tctracks/core/` | tctracks is the package that uses it; the CLI imports from it |
| `_backends/local_global_parity.py` | local_global | `evaluators/local_global/core/parity.py` | single owner |
| `_backends/env/` | spectra_ecmwf_v2 runner; `toolchain.sh` is meant to be sourced by job scripts | `shared/toolchain.py`, `shared/toolchain.sh` | infrastructure used by evaluators and by shell job scripts |
| `_backends/videogen/` | `eval.cli videogen` only | `tools/videogen/` | no evaluator |
| `_backends/sigma_evaluator/`, `checkpoint_utils.py` | scripts and tests only; the `sigma` evaluator is retired | `tools/sigma_evaluator/` | no live evaluator |
| `_backends/obs_crps/` | `tools/station_head` (in docstrings); the `obs_crps` evaluator is retired | `tools/obs_crps/` | no live evaluator |
| `_backends/plot_intermediate/`, `weight_diagnostics/` | tests and archived scripts; their evaluators are retired | `tools/plot_intermediate/`, `tools/weight_diagnostics/` | no live evaluator |
| `_backends/spectra/` (other files), `spectra_ecmwf/plot_ratio.py` | notebooks and docs only; the evaluators `spectra` and `spectra_ecmwf` are retired | `tools/spectra_analysis/` | no live evaluator |

One correction was made while moving: `region_plotting/plotting/manifest.py` (the function `write_manifest`) turned out to be used by the member wind maps as well, so it went to `eval/shared/manifest.py` instead of `evaluators/region_plot/core/plotting/`.

Tests move with the code they test. A test that exercises one moved module goes to the `tests/` folder of its new owner, and a test that spans several owners stays where it is with its imports updated (for example `eval/scoreboard/tests/test_integration.py`). Fifteen forwarding modules remain in `eval/_backends/` for callers outside the repository, and `eval/_backends/README.md` lists each with the file that still needs updating.

## Detail per folder

The sections below were generated by the scan described above from `origin/main` at `dfc824e`. A source of the form "test eval/tests/..." is a test file. Destination folders in the first line of each section are where the files went.

### `checkpoint_utils.py` (top-level module)
Files: 1 (checkpoint_utils.py)
Destination: `eval/tools/sigma_evaluator/`
Python importers outside the backend:
- backend sigma_evaluator: eval/_backends/sigma_evaluator/run_sigma_evaluator.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- none

### `env`
Files: 3 (__init__.py, toolchain.py, toolchain.sh)
Destination: `(deleted)/`, `eval/shared/`
Python importers outside the backend:
- evaluator spectra_ecmwf_v2: eval/evaluators/spectra_ecmwf_v2/runner.py
- test eval/tests/test_toolchain.py: eval/tests/test_toolchain.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- eval/_backends/env/toolchain.sh:5 (path)
- eval/config/toolchains.yaml:2 (path)
- eval/evaluators/spectra_ecmwf_v2/runner.py:24 (module name)
- eval/tests/test_toolchain.py:9 (module name)

### `local_global_parity.py` (top-level module)
Files: 1 (local_global_parity.py)
Destination: `eval/evaluators/local_global/core/`
Python importers outside the backend:
- evaluator local_global: eval/evaluators/local_global/runner.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- eval/evaluators/local_global/runner.py:7 (module name)

### `obs_crps`
Files: 3 (__init__.py, obs_crps_compute.py, plotting.py)
Destination: `eval/tools/obs_crps/`
Python importers outside the backend:
- none
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- tools/station_head/stage1/retrieve_stations.py:4 (path)
- tools/station_head/stage2a/dataset.py:18 (path)
- tools/station_head/stage2a/dataset.py:57 (path)

### `plot_intermediate`
Files: 3 (__init__.py, generate_from_bundle.py, plot_intermediate.py)
Destination: `eval/tools/plot_intermediate/`
Python importers outside the backend:
- eval/archive: eval/archive/jobs/generate_intermediate_from_bundle.py
- test eval/tests/test_plot_intermediate.py: eval/tests/test_plot_intermediate.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- eval/archive/jobs/generate_intermediate_from_bundle.py:10 (module name)
- eval/tests/test_plot_intermediate.py:11 (module name)
- eval/tests/test_plot_intermediate.py:25 (module name)

### `precip`
Files: 10 (__init__.py, metrics.py, score_gribs.py, sources.py, test_metrics.py, test_regional_truth_order.py, test_score_gribs.py, test_sources.py, test_tp_histogram_comparison.py, tp_histogram_comparison.py)
Destination: `eval/evaluators/precip_dist/core/`, `eval/evaluators/precip_dist/tests/`, `eval/evaluators/precip_scores/core/`, `eval/evaluators/precip_scores/tests/`, `eval/shared/precip/`, `eval/shared/precip/tests/`
Python importers outside the backend:
- backend region_plotting: eval/_backends/region_plotting/plot_precip_events.py, eval/_backends/region_plotting/precip_events.py
- evaluator lane_diagnostics: eval/evaluators/lane_diagnostics/compute.py
- evaluator precip_scores: eval/evaluators/precip_scores/runner.py
- interp/: interp/core/data.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- eval/_backends/precip/score_gribs.py:22 (python -m)
- eval/_backends/precip/tp_histogram_comparison.py:23 (python -m)
- eval/evaluators/lane_diagnostics/compute.py:141 (module name)
- eval/evaluators/precip_dist/__init__.py:3 (module name)
- eval/evaluators/precip_dist/runner.py:49 (python -m)
- eval/evaluators/precip_dist/tests/test_runner.py:26 (module name)
- eval/evaluators/precip_scores/runner.py:24 (module name)
- eval/evaluators/precip_scores/runner.py:25 (module name)
- eval/evaluators/precip_scores/runner.py:253 (module name)
- eval/evaluators/spectra_ecmwf_v2/_input_bundle_stager.py:104 (path)
- eval/evaluators/spectra_ecmwf_v2/_input_bundle_stager.py:199 (path)
- interp/core/data.py:269 (module name)

### `probabilistic`
Files: 3 (__init__.py, plotting.py, scoring.py)
Destination: `eval/evaluators/probabilistic/core/`
Python importers outside the backend:
- backend spread_proxy: eval/_backends/spread_proxy/scoring.py
- evaluator probabilistic: eval/evaluators/probabilistic/plotter.py, eval/evaluators/probabilistic/runner.py
- test eval/tests/test_probabilistic_evaluator.py: eval/tests/test_probabilistic_evaluator.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- eval/evaluators/probabilistic/plotter.py:7 (module name)
- eval/evaluators/probabilistic/runner.py:7 (module name)
- eval/tests/test_probabilistic_evaluator.py:9 (module name)

### `quaver`
Files: 8 (README.md, __init__.py, compute_quaver.sh, q_compute_probabilistic.py, q_compute_surface_only.py, q_plot_pl.py, q_plot_sfc.py, q_plot_sfc_pristine.py)
Destination: `eval/evaluators/quaver/core/`
Python importers outside the backend:
- none
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- none

### `region_plotting`
Files: 21 (__init__.py, local_plotting.py, plot_intermediate_presets.py, plot_member_wind_maps.py, plot_one_date_local.py, plot_precip_events.py, plot_regions.py, plot_tc_contours_from_predictions.py, plot_trajectory_wind_maps.py, __init__.py, config.py, coordinate_utils.py, datetime_utils.py, manifest.py, metadata.py, preprocessing.py, variable_utils.py, precip_events.py, __init__.py, test_member_wind_maps.py, test_precip_events.py)
Destination: `(deleted)/`, `eval/evaluators/membermaps/core/`, `eval/evaluators/membermaps/tests/`, `eval/evaluators/precip_events/core/`, `eval/evaluators/precip_events/tests/`, `eval/evaluators/region_plot/core/`, `eval/evaluators/region_plot/core/plotting/`
Python importers outside the backend:
- backend plot_intermediate: eval/_backends/plot_intermediate/plot_intermediate.py
- backend precip: eval/_backends/precip/tests/test_regional_truth_order.py
- backend weight_diagnostics: eval/_backends/weight_diagnostics/mechanistic_compare_v1.py
- eval.cli: eval/cli/membermaps.py
- eval/archive: eval/archive/jobs/select_tc_representative_case.py, eval/archive/run.py
- evaluator membermaps: eval/evaluators/membermaps/runner.py
- evaluator precip_events: eval/evaluators/precip_events/runner.py
- evaluator precip_events (test): eval/evaluators/precip_events/tests/test_runner.py
- evaluator region_plot: eval/evaluators/region_plot/plotter.py
- test eval/tests/test_plot_one_date_local.py: eval/tests/test_plot_one_date_local.py
- test eval/tests/test_plotting_config.py: eval/tests/test_plotting_config.py
- test eval/tests/test_plotting_coordinate_utils.py: eval/tests/test_plotting_coordinate_utils.py
- test eval/tests/test_plotting_datetime_utils.py: eval/tests/test_plotting_datetime_utils.py
- test eval/tests/test_plotting_metadata.py: eval/tests/test_plotting_metadata.py
- test eval/tests/test_region_plotting_custom_o1280.py: eval/tests/test_region_plotting_custom_o1280.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- TESTING.md:88 (path)
- eval/_backends/region_plotting/plot_member_wind_maps.py:29 (python -m)
- eval/archive/jobs/select_tc_representative_case.py:12 (module name)
- eval/archive/jobs/select_tc_representative_case.py:13 (module name)
- eval/archive/run.py:66 (module name)
- eval/archive/run.py:206 (module name)
- eval/cli/membermaps.py:4 (module name)
- eval/cli/membermaps.py:45 (module name)
- eval/cli/membermaps.py:57 (module name)
- eval/evaluators/membermaps/__init__.py:19 (path)
- eval/evaluators/membermaps/runner.py:70 (module name)
- eval/evaluators/precip_events/runner.py:7 (module name)
- eval/evaluators/precip_events/runner.py:19 (module name)
- eval/evaluators/precip_events/runner.py:69 (python -m)
- eval/evaluators/precip_events/tests/test_runner.py:10 (module name)
- eval/evaluators/precip_events/tests/test_runner.py:42 (module name)
- eval/evaluators/region_plot/__init__.py:3 (module name)
- eval/evaluators/region_plot/__init__.py:16 (module name)
- eval/evaluators/region_plot/plotter.py:3 (module name)
- eval/evaluators/region_plot/plotter.py:45 (module name)
- eval/evaluators/region_plot/runner.py:41 (python -m)
- eval/evaluators/region_plot/tests/test_runner.py:27 (module name)
- eval/tests/test_plot_one_date_local.py:8 (module name)
- eval/tests/test_plotting_config.py:5 (module name)
- eval/tests/test_plotting_coordinate_utils.py:7 (module name)
- eval/tests/test_plotting_datetime_utils.py:8 (module name)
- eval/tests/test_plotting_metadata.py:5 (module name)
- eval/tests/test_region_plotting_custom_o1280.py:11 (module name)
- eval/tests/test_region_plotting_custom_o1280.py:17 (module name)
- eval/tests/test_region_plotting_custom_o1280.py:400 (module name)

### `scoreboard`
Files: 10 (__init__.py, _surface_compute.py, _utils.py, canonical_data.py, canonical_analysis.yaml, canonical_eefo.yaml, row_matching.py, spectra.py, surface.py, tc.py)
Destination: `(deleted)/`, `eval/evaluators/spectra_ecmwf_v2/core/`, `eval/evaluators/surface/core/`, `eval/evaluators/tc/core/`, `eval/evaluators/tc/core/data/`, `eval/shared/`
Python importers outside the backend:
- backend probabilistic: eval/_backends/probabilistic/scoring.py
- backend spread_proxy: eval/_backends/spread_proxy/scoring.py
- eval/archive: eval/archive/scoreboard_old/__init__.py, eval/archive/scoreboard_old/cli.py
- eval/jobs: eval/jobs/scoreboard_metrics.py, eval/jobs/scoreboard_surface_loss.py
- evaluator spectra_ecmwf_v2: eval/evaluators/spectra_ecmwf_v2/_plotter.py, eval/evaluators/spectra_ecmwf_v2/scorer.py
- evaluator surface: eval/evaluators/surface/runner.py, eval/evaluators/surface/scorer.py
- evaluator tc: eval/evaluators/tc/scorer.py
- test eval/scoreboard/tests/test_integration.py: eval/scoreboard/tests/test_integration.py
- test eval/scoreboard/tests/test_row_matching.py: eval/scoreboard/tests/test_row_matching.py
- test eval/scoreboard/tests/test_spectra.py: eval/scoreboard/tests/test_spectra.py
- test eval/scoreboard/tests/test_surface.py: eval/scoreboard/tests/test_surface.py
- test eval/scoreboard/tests/test_tc.py: eval/scoreboard/tests/test_tc.py
- test eval/scoreboard/tests/test_utils.py: eval/scoreboard/tests/test_utils.py
- test eval/tests/test_spectra_naming.py: eval/tests/test_spectra_naming.py
- test eval/tests/test_tc_reference_dedup.py: eval/tests/test_tc_reference_dedup.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- eval/archive/scoreboard_old/__init__.py:8 (module name)
- eval/archive/scoreboard_old/__init__.py:9 (module name)
- eval/archive/scoreboard_old/__init__.py:10 (module name)
- eval/archive/scoreboard_old/__init__.py:15 (module name)
- eval/archive/scoreboard_old/__init__.py:22 (module name)
- eval/archive/scoreboard_old/__init__.py:28 (module name)
- eval/archive/scoreboard_old/cli.py:12 (module name)
- eval/archive/scoreboard_old/cli.py:13 (module name)
- eval/archive/scoreboard_old/cli.py:31 (module name)
- eval/archive/scoreboard_old/cli.py:40 (module name)
- eval/evaluators/spectra_ecmwf_v2/_plotter.py:378 (module name)
- eval/evaluators/spectra_ecmwf_v2/scorer.py:16 (module name)
- eval/evaluators/spectra_ecmwf_v2/scorer.py:41 (module name)
- eval/evaluators/surface/__init__.py:6 (module name)
- eval/evaluators/surface/runner.py:12 (module name)
- eval/evaluators/surface/scorer.py:13 (module name)
- eval/evaluators/surface/scorer.py:17 (module name)
- eval/evaluators/tc/scorer.py:3 (module name)
- eval/evaluators/tc/scorer.py:19 (module name)
- eval/evaluators/tc/scorer.py:20 (module name)
- eval/jobs/scoreboard_metrics.py:16 (module name)
- eval/jobs/scoreboard_metrics.py:24 (module name)
- eval/jobs/scoreboard_metrics.py:25 (module name)
- eval/jobs/scoreboard_metrics.py:35 (module name)
- eval/jobs/scoreboard_metrics.py:53 (module name)
- eval/jobs/scoreboard_metrics.py:56 (module name)
- eval/jobs/scoreboard_metrics.py:65 (module name)
- eval/jobs/scoreboard_metrics.py:69 (module name)
- eval/jobs/scoreboard_surface_loss.py:9 (module name)
- eval/scoreboard/tests/test_integration.py:21 (module name)
- eval/scoreboard/tests/test_integration.py:49 (module name)
- eval/scoreboard/tests/test_row_matching.py:5 (module name)
- eval/scoreboard/tests/test_spectra.py:10 (module name)
- eval/scoreboard/tests/test_spectra.py:85 (module name)
- eval/scoreboard/tests/test_surface.py:9 (module name)
- eval/scoreboard/tests/test_tc.py:15 (module name)
- eval/scoreboard/tests/test_utils.py:6 (module name)
- eval/tests/test_spectra_naming.py:10 (module name)
- eval/tests/test_tc_reference_dedup.py:56 (module name)
- eval/tests/test_tc_reference_dedup.py:57 (module name)

### `sigma_evaluator`
Files: 8 (__init__.py, plot_sigma_evaluations.py, run_sigma_evaluator.py, run_sigma_evaluator.sh, scheduler_study.py, sigma_evaluator.py, sigmas.py, sigma_eval.sbatch)
Destination: `eval/tools/sigma_evaluator/`, `eval/tools/sigma_evaluator/templates/`
Python importers outside the backend:
- eval/archive: eval/archive/run.py
- eval/jobs (test): eval/jobs/tests/test_sigma_evaluator.py
- test eval/tests/test_scheduler_study.py: eval/tests/test_scheduler_study.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- eval/_backends/sigma_evaluator/run_sigma_evaluator.sh:39 (python -m)
- eval/_backends/sigma_evaluator/templates/sigma_eval.sbatch:102 (python -m)
- eval/archive/run.py:87 (module name)
- eval/evaluators/sigma_loss/kernel/loader.py:131 (module name)
- eval/jobs/tests/test_run_sigma_evaluator.py:34 (module name)
- eval/jobs/tests/test_run_sigma_evaluator.py:114 (module name)
- eval/jobs/tests/test_run_sigma_evaluator.py:238 (module name)
- eval/jobs/tests/test_run_sigma_evaluator.py:251 (module name)
- eval/jobs/tests/test_run_sigma_evaluator.py:288 (module name)
- eval/jobs/tests/test_run_sigma_evaluator.py:344 (module name)
- eval/jobs/tests/test_run_sigma_evaluator.py:439 (module name)
- eval/jobs/tests/test_run_sigma_evaluator.py:532 (module name)
- eval/jobs/tests/test_run_sigma_evaluator.py:649 (module name)
- eval/jobs/tests/test_run_sigma_evaluator.py:683 (module name)
- eval/jobs/tests/test_run_sigma_evaluator.py:709 (module name)
- eval/jobs/tests/test_sigma_evaluator.py:8 (module name)
- eval/jobs/tests/test_sigma_evaluator.py:9 (module name)
- eval/jobs/tests/test_sigma_evaluator.py:10 (module name)
- eval/jobs/tests/test_sigma_evaluator.py:11 (module name)
- eval/jobs/tests/test_sigma_evaluator.py:134 (module name)
- eval/tests/test_scheduler_study.py:7 (module name)

### `spectra`
Files: 13 (__init__.py, calibrate_fast_spectra_proxy.py, grb_to_spectra.sh, harmonics.py, intermediate_sigma_curve.py, make_fullgrid_templates.py, naming.py, noise_residual_dual_spectra.py, plot_fast_spectra_proxy_overlay.py, plot_reference_models_compare.py, plot_spectra.py, plot_spectra_compare.py, restore_legacy_cache.py)
Destination: `eval/evaluators/spectra_ecmwf_v2/core/`, `eval/tools/spectra_analysis/`
Python importers outside the backend:
- backend scoreboard: eval/_backends/scoreboard/spectra.py
- evaluator spectra_ecmwf_v2: eval/evaluators/spectra_ecmwf_v2/_amplitude_computer.py, eval/evaluators/spectra_ecmwf_v2/_plotter.py, eval/evaluators/spectra_ecmwf_v2/scorer.py
- evaluator spectra_ecmwf_v2 (test): eval/evaluators/spectra_ecmwf_v2/tests/test_runner.py, eval/evaluators/spectra_ecmwf_v2/tests/test_scorer.py
- test eval/tests/test_spectra_naming.py: eval/tests/test_spectra_naming.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- eval/archive/jobs/launch_full_eval_suite.sh:218 (python -m)
- eval/evaluators/spectra_coherence/runner.py:85 (module name)
- eval/evaluators/spectra_ecmwf_v2/_amplitude_computer.py:21 (module name)
- eval/evaluators/spectra_ecmwf_v2/_plotter.py:62 (module name)
- eval/evaluators/spectra_ecmwf_v2/scorer.py:46 (module name)
- eval/evaluators/spectra_ecmwf_v2/tests/test_runner.py:187 (module name)
- eval/evaluators/spectra_ecmwf_v2/tests/test_runner.py:207 (module name)
- eval/evaluators/spectra_ecmwf_v2/tests/test_scorer.py:11 (module name)
- eval/tests/test_spectra_naming.py:11 (module name)

### `spectra_ecmwf`
Files: 1 (plot_ratio.py)
Destination: `eval/tools/spectra_analysis/`
Python importers outside the backend:
- none
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- none

### `spread_proxy`
Files: 3 (__init__.py, plotting.py, scoring.py)
Destination: `eval/evaluators/spread_proxy/core/`
Python importers outside the backend:
- evaluator spread_proxy: eval/evaluators/spread_proxy/plotter.py, eval/evaluators/spread_proxy/runner.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- eval/evaluators/spread_proxy/plotter.py:7 (module name)
- eval/evaluators/spread_proxy/runner.py:7 (module name)

### `storm_maps`
Files: 3 (__init__.py, overlay_spectra_multi_arm.py, render.py)
Destination: `eval/evaluators/storm_maps/core/`
Python importers outside the backend:
- evaluator storm_maps: eval/evaluators/storm_maps/runner.py
- evaluator storm_maps (test): eval/evaluators/storm_maps/tests/test_runner.py
- test eval/tests/test_diag_plot_helpers.py: eval/tests/test_diag_plot_helpers.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- eval/_backends/storm_maps/render.py:23 (python -m)
- eval/evaluators/storm_maps/__init__.py:7 (module name)
- eval/evaluators/storm_maps/runner.py:1 (module name)
- eval/evaluators/storm_maps/runner.py:14 (module name)
- eval/evaluators/storm_maps/tests/test_runner.py:6 (module name)
- eval/jobs/ladder.py:141 (python -m)
- eval/tests/test_diag_plot_helpers.py:2 (module name)

### `tc`
Files: 15 (__init__.py, __main__.py, all_events_request.sh, data_types.py, events.py, experiment_config.py, grid.py, loading_grib.py, loading_predictions.py, member_plot.py, pdf_plot.py, plot_config.py, stats.py, loading_data.py, workflows.py)
Destination: `eval/evaluators/tc/core/`, `eval/evaluators/tc/core/tools/`
Python importers outside the backend:
- eval/archive: eval/archive/jobs/select_tc_representative_case.py
- eval/jobs: eval/jobs/backfill_tc_extreme_percentiles.py
- evaluator lane_diagnostics: eval/evaluators/lane_diagnostics/standing_set.py
- evaluator tc: eval/evaluators/tc/comparison_contract.py, eval/evaluators/tc/plotter.py, eval/evaluators/tc/runner.py
- evaluator tc_structure: eval/evaluators/tc_structure/runner.py
- scripts/: scripts/tc_brainstorm_round1_pdfs.py, scripts/tc_cfec83a3_manual_vs_prepml_pdfs.py, scripts/tc_resolution_damping_pdfs.py, scripts/tc_top7_franklin_idalia_pdfs.py
- test eval/tests/test_tc_comparison_contract.py: eval/tests/test_tc_comparison_contract.py
- test eval/tests/test_tc_plot_members.py: eval/tests/test_tc_plot_members.py
- test eval/tests/test_tc_plotter.py: eval/tests/test_tc_plotter.py
- test eval/tests/test_tc_support_modes.py: eval/tests/test_tc_support_modes.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- ARCHITECTURE.md:51 (path)
- ARCHITECTURE.md:144 (path)
- eval/archive/jobs/launch_full_eval_suite.sh:264 (python -m)
- eval/archive/jobs/launch_full_eval_suite.sh:265 (python -m)
- eval/archive/jobs/launch_proxy_eval.sh:413 (python -m)
- eval/archive/jobs/select_tc_representative_case.py:14 (module name)
- eval/archive/templates/o48_o96_write_from_predictions.sbatch:126 (python -m)
- eval/archive/templates/o48_o96_write_from_predictions.sbatch:137 (python -m)
- eval/evaluators/lane_diagnostics/standing_set.py:84 (module name)
- eval/evaluators/lane_diagnostics/standing_set.py:85 (module name)
- eval/evaluators/tc/comparison_contract.py:9 (module name)
- eval/evaluators/tc/plotter.py:9 (module name)
- eval/evaluators/tc/plotter.py:10 (module name)
- eval/evaluators/tc/runner.py:17 (module name)
- eval/evaluators/tc/runner.py:18 (module name)
- eval/evaluators/tc/runner.py:19 (module name)
- eval/evaluators/tc/runner.py:20 (module name)
- eval/evaluators/tc/runner.py:26 (module name)
- eval/evaluators/tc/runner.py:27 (module name)
- eval/evaluators/tc/runner.py:33 (module name)
- eval/evaluators/tc_structure/runner.py:4 (module name)
- eval/evaluators/tc_structure/runner.py:51 (module name)
- eval/evaluators/tc_structure/runner.py:52 (module name)
- eval/evaluators/tc_structure/runner.py:53 (module name)
- eval/evaluators/tc_structure/runner.py:54 (module name)
- eval/evaluators/tctracks/plotter.py:473 (path)
- eval/jobs/backfill_tc_extreme_percentiles.py:41 (module name)
- eval/jobs/backfill_tc_extreme_percentiles.py:42 (module name)
- eval/jobs/backfill_tc_extreme_percentiles.py:43 (module name)
- eval/jobs/backfill_tc_extreme_percentiles.py:50 (module name)
- eval/tests/test_tc_comparison_contract.py:11 (module name)
- eval/tests/test_tc_plot_members.py:7 (module name)
- eval/tests/test_tc_plotter.py:10 (module name)
- eval/tests/test_tc_plotter.py:11 (module name)
- eval/tests/test_tc_support_modes.py:10 (module name)
- eval/tests/test_tc_support_modes.py:11 (module name)
- eval/tests/test_tc_support_modes.py:12 (module name)
- eval/tests/test_tc_support_modes.py:13 (module name)
- eval/tests/test_tc_support_modes.py:14 (module name)
- eval/tests/test_tc_support_modes.py:15 (module name)
- eval/tests/test_tc_support_modes.py:16 (module name)
- scripts/tc_brainstorm_round1_pdfs.py:26 (module name)
- scripts/tc_brainstorm_round1_pdfs.py:27 (module name)
- scripts/tc_brainstorm_round1_pdfs.py:28 (module name)
- scripts/tc_brainstorm_round1_pdfs.py:35 (module name)
- scripts/tc_brainstorm_round1_pdfs.py:36 (module name)
- scripts/tc_brainstorm_round1_pdfs.py:37 (module name)
- scripts/tc_cfec83a3_manual_vs_prepml_pdfs.py:25 (module name)
- scripts/tc_cfec83a3_manual_vs_prepml_pdfs.py:26 (module name)
- scripts/tc_cfec83a3_manual_vs_prepml_pdfs.py:27 (module name)
- scripts/tc_cfec83a3_manual_vs_prepml_pdfs.py:34 (module name)
- scripts/tc_cfec83a3_manual_vs_prepml_pdfs.py:35 (module name)
- scripts/tc_cfec83a3_manual_vs_prepml_pdfs.py:36 (module name)
- scripts/tc_resolution_damping_pdfs.py:10 (module name)
- scripts/tc_resolution_damping_pdfs.py:26 (module name)
- scripts/tc_resolution_damping_pdfs.py:27 (module name)
- scripts/tc_resolution_damping_pdfs.py:28 (module name)
- scripts/tc_resolution_damping_pdfs.py:32 (module name)
- scripts/tc_resolution_damping_pdfs.py:33 (module name)
- scripts/tc_top7_franklin_idalia_pdfs.py:26 (module name)
- scripts/tc_top7_franklin_idalia_pdfs.py:27 (module name)
- scripts/tc_top7_franklin_idalia_pdfs.py:28 (module name)
- scripts/tc_top7_franklin_idalia_pdfs.py:35 (module name)
- scripts/tc_top7_franklin_idalia_pdfs.py:36 (module name)
- scripts/tc_top7_franklin_idalia_pdfs.py:37 (module name)

### `tctracker`
Files: 8 (__init__.py, parsing.py, pipeline.py, sources.py, tables.py, test_parsing_tables.py, test_pipeline.py, test_sources.py)
Destination: `eval/evaluators/tctracks/core/`, `eval/evaluators/tctracks/tests/`
Python importers outside the backend:
- eval.cli: eval/cli/_session.py, eval/cli/tctracker.py
- evaluator tctracks: eval/evaluators/tctracks/runner.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- eval/cli/_session.py:61 (module name)
- eval/cli/_session.py:67 (module name)
- eval/cli/_session.py:195 (module name)
- eval/cli/tctracker.py:73 (module name)
- eval/cli/tctracker.py:79 (module name)
- eval/evaluators/tctracks/runner.py:19 (module name)
- eval/evaluators/tctracks/runner.py:20 (module name)
- eval/evaluators/tctracks/runner.py:25 (module name)

### `videogen`
Files: 8 (__init__.py, __main__.py, config.py, data.py, layouts.py, panels.py, pipeline.py, scenes.py)
Destination: `eval/tools/videogen/`
Python importers outside the backend:
- eval.cli: eval/cli/videogen.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- eval/_backends/videogen/__init__.py:11 (python -m)
- eval/_backends/videogen/__main__.py:1 (python -m)
- eval/_backends/videogen/__main__.py:9 (python -m)
- eval/_backends/videogen/__main__.py:14 (python -m)
- eval/_backends/videogen/__main__.py:18 (python -m)
- eval/_backends/videogen/__main__.py:39 (python -m)
- eval/cli/videogen.py:3 (module name)
- eval/cli/videogen.py:19 (module name)
- eval/cli/videogen.py:34 (module name)

### `weight_diagnostics`
Files: 4 (__init__.py, mechanistic_compare_v1.py, plot_checkpoint_weights.py, plot_mechanistic_compare_v1.py)
Destination: `eval/tools/weight_diagnostics/`
Python importers outside the backend:
- test eval/tests/test_mechanistic_compare_v1.py: eval/tests/test_mechanistic_compare_v1.py
- test eval/tests/test_weight_diagnostics.py: eval/tests/test_weight_diagnostics.py
Textual references (python -m strings, file paths, docstrings), outside the backend itself unless a command:
- ARCHITECTURE.md:111 (path)
- eval/README.md:239 (path)
- eval/_backends/weight_diagnostics/plot_checkpoint_weights.py:6 (python -m)
- eval/tests/test_mechanistic_compare_v1.py:10 (module name)
- eval/tests/test_mechanistic_compare_v1.py:11 (module name)
- eval/tests/test_mechanistic_compare_v1.py:22 (module name)
- eval/tests/test_weight_diagnostics.py:7 (module name)


## References outside the repository

The file `docs-proposals/backends-external-references-20260929.txt` lists every line of `/home/ecm5702/dev/docs`, `/home/ecm5702/dev/scripts` and `/home/ecm5702/dev/jobscripts` that names `_backends`, excluding job log files (`*.out`, `*.err`, `*.log`), the archive `docs/attic/`, generated graph output and scoreboard JSON snapshots. The lines fall into two groups. The first group is code or runbook text that a person may run again: `docs/scripts/tp_histogram_comparison.py` and `docs/scripts/generate_combined_tc_pdf.py` import old module paths, `docs/docs/instructions/evaluation-workflow.md` line 92 and `plotting-readability.md` line 26 give `python -m eval._backends...` commands, `docs/epics/spread-calibration/README.md` line 26 gives one, and five sbatch files and two Python files under `jobscripts/submit/` name old paths. The second group is historical prose in completed epics and task notes, which describes the code as it was and does not need an update. No crontab exists for the user (`crontab -l` answers "no crontab"), and no other checkout under `/home/ecm5702/dev/` that is not a worktree of this repository mentions `_backends`.
