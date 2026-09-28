# eval/jobs

This directory holds the code that turns the evaluation framework into SLURM jobs
and the commands that operate on finished runs. Four kinds of thing live here, and
each kind has its own place.

## Orchestration modules (Python you import or run with `-m`)

- `pipeline.py` renders the chain of sbatch scripts for one lane (predict, then one
  job per evaluator, then the scoreboard) and a `submit_pipeline.sh` that chains
  them with `--dependency=afterok`. Run it as `python -m eval.jobs.pipeline --help`.
- `renderer.py` renders a single sbatch script from the host and lane YAML files.
- `resources.py` resolves the SLURM resources of each stage from the host defaults
  and the lane's `resource_profiles`.
- `slurm_jobs.py` reads SLURM job states so a pipeline can detect a failed dependency.
- `ladder.py` and `ladder_references.py` implement the ladder evaluation, the cheap
  per-checkpoint progress metrics scored against a frozen baseline.
- `evolution.py` draws the evolution grid that `python -m eval.cli evolution` calls.

## Maintenance commands that keep their documented module path

These are run with `python -m eval.jobs.<name>` and are named in the lane files, so
they stay at the top level.

- `backfill_tc_extreme_percentiles.py` adds the percentile fields to old TC stats files.
- `compare_probabilistic_reference.py` compares the local probabilistic summary with a
  reference exported from quaver.
- `scoreboard_metrics.py` and `scoreboard_surface_loss.py` are libraries that
  the scoreboard helpers and the sigma scheduler study import.

## `scripts/` (one-off scripts and experiment suites)

Scripts written for a single investigation. They have no importer in the framework
and are run by path. See `scripts/README.md`.

## `templates/` (sbatch and shell templates that the framework or people copy)

Maintained templates; `templates/README.md` lists them. `templates/preflight_eval_check.sh`
is sourced by the generated pipeline scripts.

## Other

- `tests/` holds the tests of this directory.
- `legacy_docs/` holds pre-refactor playbooks, kept for reference only.
- Retired jobs are under `eval/_quarantine/20260928/jobs/`.
