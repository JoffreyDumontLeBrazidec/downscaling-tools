# Deprecated forwarding modules

This folder used to hold the computation behind the evaluators. On 2026-09-29 that code moved
to `eval/evaluators/<name>/core/`, `eval/tools/<name>/` and `eval/shared/`; `ARCHITECTURE.md`
(section 2) says which goes where and `docs-proposals/backends-map-20260929.md` records who
used each old folder.

What is left here is 15 small modules that only forward to the new location. Each one logs a
deprecation warning and then either becomes the new module (when imported) or runs it (when
started with `python -m`). They exist because code outside this repository still uses the old
paths. Nothing inside the repository imports them, and one test
(`eval/tests/test_backends_forwarders.py`) checks that they keep forwarding.

Delete a module, and the folder once it is empty, when the caller listed for it has been
updated. The callers are files of `/home/ecm5702/dev`, which this repository does not own.

| Old module | New module | Caller outside the repository |
|---|---|---|
| `eval._backends.precip.tp_histogram_comparison` | `eval.evaluators.precip_dist.core.tp_histogram_comparison` | `docs/scripts/tp_histogram_comparison.py` lines 7 and 10 |
| `eval._backends.tc.data_types`, `events`, `experiment_config`, `grid`, `loading_grib`, `loading_predictions`, `plot_config` | `eval.evaluators.tc.core.<same name>` | `docs/scripts/generate_combined_tc_pdf.py` lines 25 to 36; `jobscripts/submit/20260709/surface_z500_boxrms.py` lines 18 to 20; `jobscripts/submit/20260729/regional_sampler_parity/dump_tc_instance_arrays.py` lines 14 to 16 |
| `eval._backends.tc.workflows` | `eval.evaluators.tc.core.workflows` | `jobscripts/submit/2026-05-12/manual_811960e6_new_o1280_o2560_20260512_ep130_100k_tc_eval.sbatch` line 166 and `..._tc_member_maps.sbatch` line 81 |
| `eval._backends.region_plotting.plot_regions` | `eval.evaluators.region_plot.core.plot_regions` | `jobscripts/submit/2026-05-12/manual_811960e6_new_o1280_o2560_20260512_ep130_100k_local_plots.sbatch` line 117 |
| `eval._backends.region_plotting.plot_one_date_local` | `eval.evaluators.region_plot.core.plot_one_date_local` | `docs/docs/instructions/plotting-readability.md` line 26; the same `..._local_plots.sbatch` line 158 |
| `eval._backends.weight_diagnostics.plot_checkpoint_weights` | `eval.tools.weight_diagnostics.plot_checkpoint_weights` | `docs/docs/instructions/evaluation-workflow.md` line 92 |
| `eval._backends.spread_proxy.scoring` | `eval.evaluators.spread_proxy.core.scoring` | `docs/epics/spread-calibration/README.md` line 26 |
| `eval._backends.storm_maps.render` | `eval.evaluators.storm_maps.core.render` | `docs/epics/tc-o320-o1280/completed-tasks/20260711_testbed_validation_200k.md` line 205 |
| `eval._backends.sigma_evaluator.run_sigma_evaluator` | `eval.tools.sigma_evaluator.run_sigma_evaluator` | five sbatch files in `jobscripts/submit/20260623/` (`sigma_o96o320_cfec83a3.sbatch` line 27 and four `sigma_o320o1280_b785bf12*.sbatch` files, line 21) |
