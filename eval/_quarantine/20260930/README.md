# Quarantine, 2026-09-30

Code retired from the evaluation framework on 2026-09-30. It is kept, not deleted,
so its history and logic stay readable. None of it is importable: `20260930` is not a
valid Python package name and the folder has no `__init__.py`. Nothing here is
collected by pytest (`norecursedirs` in `pytest.ini`).

## `tc_member_maps/`: the per-member maps of the `tc` evaluator

**What it was.** When a lane's `tc.member_maps.enabled` was true, the `tc` evaluator
wrote, after its statistics, one PDF per event and forecast date,
`evaluators/tc/member_maps/tc_members_<event>_<run>_<date>.pdf`. Each page showed one
ensemble member at one lead time: a row of three maps (input, model, truth) of mean
sea level pressure and a row of 10 m wind speed over the event box. With
`combined_pdf: true` the pages were also merged into one PDF. The same pages could be
drawn directly with `python -m eval.evaluators.tc.core.workflows member-maps`.

**Why it was retired.** Owner decision of 2026-09-30, after seeing one of these
pages: "I don't like this kind of plots, remove them definitely from eval.cli".

**What replaces it.** `zoom_maps` (`python -m eval.cli zoom_maps`, or the evaluator of
the same name) for single-member maps of a storm next to the input and the truth.

**What else of `tc` changed.** Nothing. Its distributions figure, `stats.json`,
`metrics.json` and scoreboard rows are unchanged.

**Where each piece used to live** (commit `b83cca2`):

| File here | Where it was |
|---|---|
| `tc_member_maps/member_plot.py` | `eval/evaluators/tc/core/member_plot.py`, moved whole (`_plot_member_page`, which drew one page, and its helpers). Nothing else used it. |
| `tc_member_maps/run_member_maps.py`, section 1 | `run_member_maps` in `eval/evaluators/tc/core/workflows.py`, and its re-export in `eval/evaluators/tc/core/__init__.py` |
| `tc_member_maps/run_member_maps.py`, section 2 | `load_prediction_member_fields` in `eval/evaluators/tc/core/loading_predictions.py`, which read and regridded the fields for the pages. Nothing else used it. |
| `tc_member_maps/run_member_maps.py`, section 3 | The block at the end of `run()` in `eval/evaluators/tc/runner.py` that read `tc.member_maps` and called `run_member_maps` (and merged the PDFs) |
| `tc_member_maps/run_member_maps.py`, section 4 | The `member-maps` subcommand of `main()` in `eval/evaluators/tc/core/workflows.py` |

No test exercised this code, so no test was moved. (`eval/evaluators/tc/tests/test_tc_plot_members.py`
concerns the separate GRIB-based `legacy-members` subcommand, which stays.)

**What still exists on purpose.**

- The lane files keep their `tc.member_maps` blocks. On 2026-09-30, 39 tracked files
  carried one (25 lanes and 14 ladder profiles) and the live checkout held 56 in all,
  counting untracked and generated lanes; 22 tracked lanes and 8 untracked ones set
  `enabled: true`, and lanes that inherit from them through `base:` inherit it. None
  of these files was edited. The block is accepted and ignored; when it has
  `enabled: true`, `tc` logs one warning per run that names this retirement and
  `zoom_maps` (`_warn_if_member_maps_requested` in `eval/evaluators/tc/runner.py`). An
  inert block is harmless and may be removed from a lane whenever that lane is edited
  for another reason.
- `python -m eval.evaluators.tc.core.workflows member-maps` (also reached as
  `python -m eval.evaluators.tc.core member-maps` and through the forwarder
  `eval._backends.tc.workflows`) is a tombstone: it prints the replacement and exits
  with status 1, whatever flags follow it. The message is `MEMBER_MAPS_RETIRED` in
  `eval/evaluators/tc/core/workflows.py`.
- `member_maps` stays in the `plots` list of the `tc` deliverables
  (`eval/evaluators/tc/__init__.py`). The lean-layout projection (`eval/lean_layout.py`)
  rebuilds `<run>/plots/` from scratch each time it runs and links only the sub-folders
  that the evaluator declares, so dropping the entry would remove the
  `plots/tc_member_maps` view of an old run folder the next time that folder is
  re-projected. Since no new run writes `member_maps/`, keeping the entry only
  affects old folders.
- `TCPlotConfig.member_map_msl_range` and `member_map_wind_range` stay in
  `eval/evaluators/tc/core/plot_config.py`: nothing reads them any more, but a lane's
  `tc.plot_config` override that names them would otherwise fail to load.
- `eval/jobs/templates/finalize_lean_eval_layout.sbatch` still knows how to copy
  `tc_member_maps*/` folders of old runs; it only reads them.
