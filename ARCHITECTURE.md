# Architecture

Target architecture for the evaluation framework. Where the current codebase
diverges from this target, a **Current state** annotation marks the gap. The
codebase evolves to match this document.

## 1. Layered Overview

```
Input sources
  |-- checkpoint (research)  -> eval.predict -> predictions_*.nc
  |-- MARS expver (prepml)   -> eval.cli predict/run --mode prepml -> predictions
                                        |
                              predictions directory
                                        |
                              eval.cli evaluate
                                        |
                    eval.evaluators.<name>.run()    (computation)
                    eval.evaluators.<name>.score()  (metrics extraction)
                    eval.evaluators.<name>.plot()   (visualization)
                    -- the same three calls for every evaluator: eval/evaluators/base.py
                                        |
                    eval.scoreboard.aggregator      (collect metrics.json)
                    eval.scoreboard.formatter       (CSV / markdown)
                                        |
                    eval.jobs.pipeline / renderer    (HPC orchestration)
```

Supporting layers:
- `eval/config/` -- lane, host, and event YAML configuration
- `eval/discovery/` -- prediction file finding and checkpoint identification
- `eval/shared/` -- code that several evaluators use: grid and plotting helpers, the precipitation truth
  source (`precip/`), the manifest writer, the toolchain recipes and small JSON helpers
- `eval/tools/` -- analysis tools that have no live evaluator, one folder each
- `eval/paths.py` -- canonical path resolution

**Current state**: Both input paths go through `eval.cli`: the checkpoint path
(`--mode manual`, the default) and the MARS/FDB path (`--mode prepml`). Both produce
a predictions directory that the evaluator framework consumes. The legacy
`eval.run mars-expver` entry point lives in `eval/archive/run.py`.

## 2. Evaluator Architecture

Each evaluator is one package under `eval/evaluators/<name>/`. The three contract
files sit at the top of the package and the computation behind them sits in a
subpackage of the same package called `core/`:

```
eval/evaluators/tc/
|-- __init__.py       # exports run, score, plot and EVALUATOR_SPEC (the contract)
|-- runner.py         # run(): orchestration, calls core
|-- scorer.py         # score(): scoreboard rows from the results
|-- plotter.py        # plot(): figures from the results
|-- core/             # the computation: data loading, statistics, grid operations,
|                     # plot code, and the reader of the results for the scoreboard
|-- tests/            # the tests of this evaluator, including its core
```

### Where the code of an evaluator lives

Until 2026-09-29 the computation lived in a separate folder, `eval/_backends/`, and
several of its folders were shared by many evaluators, so the tree did not show which
evaluator owned what. There is now one convention, and it has three parts.

1. Code that belongs to one evaluator lives in `eval/evaluators/<name>/core/`, and its
   tests in `eval/evaluators/<name>/tests/`. Small evaluators may have no `core/`.
   The private helper modules an evaluator already had at the top of its package
   (for example `spectra_ecmwf_v2/_grib_stager.py`) stay there.
   Evaluators that never had a backend folder keep their helper modules where they are
   (`tc_structure/core.py`, `sigma_loss/kernel/`, the `mlflow`, `shape` and
   `spectra_coherence` modules); renaming them was outside this change.
2. Code that several evaluators need goes to `eval/shared/`, in a module or package
   named for what it does: `precip/` (the precipitation truth source), `manifest.py`,
   `toolchain.py` with `toolchain.sh`, `json_utils.py`, `grid.py`, `plotting.py`,
   `date_bootstrap.py`. When one evaluator is clearly the primary owner of some code
   and others only borrow it, the code stays in the owner's `core/` and the others import
   from there. Those borrowings are few, and this is the whole list:
   `tc_structure` and `lane_diagnostics` import TC events, grids, loaders and plot code from
   `tc/core`; `probabilistic` and `spread_proxy` import three helpers from
   `surface/core/compute.py`; `spread_proxy` imports four helpers from `probabilistic/core`.
   Anything else that two evaluators need belongs in `eval/shared/`.
3. Analysis code that has no live evaluator goes to `eval/tools/<name>/`, next to
   `eval/tools/parity/`. That covers `videogen` (used only by `eval.cli videogen`),
   `sigma_evaluator`, `obs_crps`, `plot_intermediate`, `weight_diagnostics` and
   `spectra_analysis`; the evaluators of the last five were retired on 2026-09-28. The tools
   `plot_intermediate` and `weight_diagnostics` borrow the region helpers of
   `region_plot/core`.

Code outside the evaluators may import an evaluator's `core/` when it needs that
evaluator's computation: `eval/jobs/scoreboard_metrics.py` reads the scoreboard files of the
tc, surface and spectra_ecmwf_v2 cores (`core/scoreboard.py`), `eval.cli membermaps` and
`eval.cli tctracker` call the cores of membermaps and tctracks, and the scripts in
`scripts/` reuse the tc core.

Importing anything below `eval.evaluators.<name>` first runs the `__init__.py` of that
evaluator, which imports its `runner.py`, `scorer.py` and `plotter.py`. A consequence is that
running a `core/` module with `python -m` while its own runner also imports it (as
`python -m eval.evaluators.storm_maps.core.render` does) only prints Python's harmless
"found in sys.modules" warning; the run itself is unaffected.

`eval/_backends/` is left with 15 small forwarding modules for callers outside this
repository, listed in `eval/_backends/README.md`, and is to be deleted when they are gone.

The full list of evaluators, with their group (scored, standard, diagnostic or
retired), the question each one answers and the host it is limited to, is
`eval/evaluators/registry.py`. To see it, and one evaluator in detail, run
`python -m eval.cli list` and `python -m eval.cli describe <name>`; both are
generated from the registry and the packages, so there is no second catalogue.

`eval/scoreboard/` contains only the canonical aggregation layer
(`aggregator.py`, `formatter.py`, `types.py`). Per-domain scoring math lives
inside the evaluator's `scorer.py` or its `core/`.

### Evaluator Contract

The contract is written down once, in `eval/evaluators/base.py` (typing Protocols
plus a checker). Every registered, non-retired evaluator package exports exactly:

```python
EVALUATOR_SPEC = {
    "name": "tc",                    # the package name
    "requires": ["predictions"],     # or ["checkpoint"] when the evaluator runs the model
    "outputs": ["stats.json: ...", ...],   # what run() writes; shown by `describe`
    "deliverables": {...},           # optional: files promoted to the run root
}

def run(predictions_dir, lane_config, eval_config, *, output_dir=None, overwrite=False,
        checkpoint=None, run_label="", **kwargs): ...
def score(results_dir, lane_config, eval_config, *, predictions_dir=None, **kwargs) -> list[dict]: ...
def plot(results_dir, lane_config, eval_config, *, output_dir=None, **kwargs) -> None: ...
```

An evaluator with nothing to score or nothing to plot uses the adapters `no_score`
and `no_plot` from `base.py`, so `score` and `plot` always exist and `eval.cli`
never has to test for them. `eval/tests/test_evaluator_contract.py` binds every
evaluator's functions against the exact call `eval/cli/evaluate.py` makes.

### Evaluator Rules

- Each evaluator writes only under its own results directory,
  `<run>/evaluators/<name>/`.
- Evaluators import from `eval.config`, `eval.discovery`, `eval.plotting`, `eval.shared`,
  their own `core/`, and stdlib. Never from `eval.jobs`.
- An evaluator imports from another evaluator only in the few cases listed under
  "Where the code of an evaluator lives"; everything else two evaluators need goes to `eval/shared/`.

**Current state**: the consolidation is done. `eval/cli/evaluate.py` dispatches by
name through `importlib.import_module(f"eval.evaluators.{name}")` over
`ALL_EVALUATORS`, which is derived from the one registry
`eval/evaluators/registry.py`, so an evaluator is reachable if and only if the
registry lists it and it is not retired. Whether an evaluator feeds the scoreboard
is also read from the registry, not from its spec. The old top-level paths
(`eval/tc/`, `eval/spectra/`, ...) no longer exist and their import paths fail
immediately, which is intentional. The old `eval/_backends/` paths fail too, except for the
15 forwarding modules described in `eval/_backends/README.md`.

Three further facts a reader needs:

- `eval/lean_layout.py` projects the lean run-root layout natively in the
  harness; `eval/cli/evaluate.py` delegates run-root resolution and plot
  consolidation to it.
- `eval/archive/` is frozen but **not** dead --
  `eval/tools/weight_diagnostics/mechanistic_compare_v1.py` and
  `eval/tests/test_eval_run.py` still import from it, and a few tests in
  `eval/jobs/tests/` exercise the archived jobs, so it cannot be removed
  without untangling those first.
- `manual_inference/` and `manual_inference_legacy_ds/` are a deliberate fork,
  not an accident: `eval/predict/_mi.py` routes between them on the
  `KEYSTONE_LEGACY_DS` environment variable so cfec83a3-era single-dataset
  checkpoints keep working. Both are load-bearing. The legacy tree is written to
  be installed under the name `manual_inference`, which is why its tests only run
  in a subprocess with that alias (see `TESTING.md`).

## 3. Configuration

Lane, host, and event configuration lives in `eval/config/` as YAML:

```
eval/config/
|-- lanes/          # One file per resolution lane
|   |-- o48_o96.yaml
|   |-- o96_o320.yaml
|   |-- o320_o1280.yaml
|   |-- o1280_o2560.yaml
|-- hosts/          # One file per HPC host
|   |-- atos_ac.yaml
|   |-- atos_ag.yaml
|-- events/         # TC event definitions
|   |-- idalia.yaml
|   |-- franklin.yaml
|   |-- ...
|-- loader.py       # Reads YAML, validates required keys, returns dict
```

**Event boxes/dates have a single source of truth: the `events/*.yaml` files.**
`eval/evaluators/tc/core/events.py` does not hardcode coordinates — it loads those
YAMLs into the `EVENTS` registry at import (so `from ...events import EVENTS`
keeps working). To add or change a TC event, edit its YAML, never `events.py`.
Scoring-event boxes must stay mutually non-overlapping (an overlap makes
per-storm extrema pick up a neighbour's low — see the dora/fernanda/idalia
shared-low degeneracy).

**Lane YAML** is the central config file. It contains: predict defaults (dates,
steps, members), per-evaluator parameters, evaluator groups, region definitions,
and reference data paths.

**Evaluator groups** control which evaluators run by default:

```yaml
evaluator_groups:
  default: [tc, spectra_ecmwf_v2, surface, region_plot]
  diagnostics: [mlflow]
```

`eval.cli evaluate` runs the `default` group unless `--only` overrides.
`--include-diagnostics` adds the diagnostics group. The `default` group holds only
scored and standard evaluators (see the registry). A retired evaluator named with
`--only` stops the CLI with exit status 1 and names its replacement; a retired name
left in a lane group is skipped with a warning.

**Config precedence**: CLI args > lane YAML > host YAML defaults. Every run emits
`effective_config.json` recording the resolved snapshot.

**`eval_config` boundary**: `eval_config` is `lane_config[evaluator_name]` -- the
evaluator-specific subsection. Evaluators read evaluator-specific values only from
`eval_config` and cross-cutting values only from `lane_config`.

## 4. Output Directory Contract

Every evaluation run produces a structured output directory with a clear
data/plots separation:

```
<scratch_eval_root>/<lane>/<run_id>/
|-- effective_config.json
|-- data/
|   |-- predictions/            # Prediction NetCDFs
|   |-- tc/                     # TC stats, per-event results
|   |   |-- metrics.json
|   |-- spectra/                # Spectral amplitudes, per-variable results
|   |   |-- metrics.json
|   |-- surface/                # Surface loss results
|   |   |-- metrics.json
|   |-- sigma/                  # Sigma sweep results
|   |   |-- metrics.json
|   |-- scoreboard/
|       |-- scores.csv
|       |-- scores.md
|-- plots/
|   |-- tc/                     # TC PDFs, member field maps
|   |-- spectra/                # Spectra comparison plots
|   |-- region_plot/            # Six-panel regional comparisons
|   |-- sigma/                  # Sigma sweep plots
|   |-- mechanistic/            # Weight diagnostic plots
|   |-- intermediate/           # Intermediate diffusion step plots
|-- logs/
```

**Rules:**
- Each evaluator writes data under `data/<name>/` and plots under
  `plots/<name>/`. No evaluator writes outside its own subdirectories.
- Scored evaluators produce `data/<name>/metrics.json` -- a list of
  `{"metric": str, "value": float, "unit": str}` records.
- `run()` refuses to overwrite existing output unless `overwrite=True` is passed.
- `effective_config.json` is emitted twice: after config resolution (before
  expensive work) and updated at completion with status and actual outputs.

**Current state**: Evaluators currently write data and plots together under
`evaluators/<name>/`. The data/plots separation is a target restructuring.

## 5. CLI

Single entry point: `python -m eval.cli <command>`. Never bare `eval`. The
package `eval/cli/` has one module per command; each publishes a `Command` record
(name, group, summary, `register`, `run`) and `eval/cli/__init__.py` collects
them. Commands that need a lane and a host go through `eval/cli/_session.py`, which
loads the configuration, exports the host environment, chooses the evaluators and
the output directory, writes `effective_config.json` (or prints it for `--dry-run`)
and afterwards writes the `--vs-baseline` diff.

| Group | Commands |
|---|---|
| discovery | `list`, `describe <evaluator>` |
| pipeline | `run`, `predict`, `prepare`, `evaluate`, `scoreboard`, `report` |
| comparison | `evolution` |
| tropical cyclone tracks | `tctracker`, `tccompare` |
| figures | `membermaps`, `videogen` |
| maintenance | `prepml-cleanup`, `config` |

```bash
# What can be evaluated, and how does one evaluator work?
python -m eval.cli list
python -m eval.cli describe tc

# Full pipeline: predict + evaluate + scoreboard
python -m eval.cli run --checkpoint <path> --lane o96_o320 [--host atos_ac] [--only tc,surface]

# Predictions only
python -m eval.cli predict --checkpoint <path> --lane o96_o320

# Evaluate existing predictions
python -m eval.cli evaluate --predictions-dir <dir> --lane o96_o320 [--only tc,surface]

# Include the lane's diagnostics group
python -m eval.cli evaluate --predictions-dir <dir> --lane o96_o320 --include-diagnostics

# Scoreboard from existing evaluation results, diffed against the lane baseline
python -m eval.cli scoreboard --eval-dir <dir> --lane o96_o320 --vs-baseline

# Dry run (print resolved config, don't execute)
python -m eval.cli run --checkpoint <path> --lane o96_o320 --dry-run
```

**Evaluator selection** follows three-step resolution (`eval/cli/_selection.py`):
1. `--only tc,surface` -- run exactly those evaluators
2. `--include-diagnostics` -- default + diagnostics groups from lane YAML
3. Neither -- default group only

**Overrides**: `--members`, `--steps`, `--dates` override lane YAML predict
defaults. CLI always wins over YAML.

**Current state**: `eval.cli` is operational for all commands, including the prepml
path (`--mode prepml`).

## 6. HPC Job Orchestration

HPC submission is handled by two layers:

**Pipeline renderer** (`eval/jobs/pipeline.py`) generates a chain of sbatch
scripts with SLURM dependency linking (`--dependency=afterok:<jobid>`). A typical
chain:

```
predict.sbatch -> tc_eval.sbatch -> spectra_eval.sbatch -> surface_eval.sbatch -> scoreboard.sbatch
                                                                                       ^
                                                                                afterok on all eval jobs
```

**Template renderer** (`eval/jobs/renderer.py`) patches individual sbatch
templates with run-specific directives (QOS, partition, resource requests,
environment setup). Host YAML provides the environment:

```yaml
environment_setup:
  module_loads: ["ecmwf-toolbox", "python3/3.11"]
  exports: {"OMP_NUM_THREADS": "1"}
  venv_activate: "/path/to/venv/bin/activate"
```

**Resource profiles** (`eval/jobs/resources.py`) define per-evaluator HPC resource
requirements (nodes, GPUs, walltime, memory) so the pipeline can size each stage
correctly.

**Entry points:**
```bash
# Render + submit a full pipeline
python -m eval.jobs.pipeline --lane o96_o320 --host atos_ac --checkpoint <path>

# Render a single sbatch (dry run)
python -m eval.jobs.renderer --lane o96_o320 --host atos_ac --checkpoint <path> --dry-run
```

**Current state**: `pipeline.py` and `renderer.py` render the chains described
above. The older shell flow scripts and per-step scoreboard templates were archived
(`eval/archive/jobs/`, which still holds `launch_full_eval_suite.sh`) or, on
2026-09-28, quarantined under `eval/_quarantine/20260928/jobs/`.
`eval/jobs/README.md` maps what is left in `eval/jobs/`: orchestration modules,
`scripts/` for one-off jobs and `templates/` for the sbatch templates the framework or
people copy.

## 7. Naming Conventions and Contracts

**Lane names**: `o48_o96`, `o96_o320`, `o320_o1280`, `o1280_o2560`
(underscore-separated, lowercase, `<input>_<output>` resolution).

**Host names**: `atos_ac`, `atos_ag` (underscore-separated, lowercase).

**Evaluator names**: listed once in `eval/evaluators/registry.py`. Must match
the directory name under `eval/evaluators/`.

**Prediction files**: `predictions_YYYYMMDD_stepNNN.nc` (regex at
`eval/discovery/predictions.py`).

**Score record**: `{"metric": str, "value": float, "unit": str}` -- the handoff
format between evaluator scorers and the scoreboard aggregator.

**EVALUATOR_SPEC**: every evaluator's `__init__.py` exports this dict (see section 2
and `eval/evaluators/base.py`):
```python
EVALUATOR_SPEC = {
    "name": "tc",
    "requires": ["predictions"],
    "outputs": ["stats.json: raw extremes per event ...", ...],
}
```

**Config paths**: all paths in YAML config files are absolute.

**Output root**: `<scratch_eval_root>/<lane>/<run_id>/`

**Import rules**:
- Evaluators import from `eval.config`, `eval.discovery`, `eval.plotting`, `eval.shared`, their
  own `core/`, and stdlib, plus the few evaluator-to-evaluator imports listed in section 2.
- No evaluator imports from `eval.jobs`.
