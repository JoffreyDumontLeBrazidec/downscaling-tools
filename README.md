# downscaling-tools

Code for running and evaluating ECMWF's diffusion-based machine-learning downscaling
models: it builds the inputs for a checkpoint, generates predictions, scores them
against the truth and the operational references, and ranks runs on a scoreboard. The
system has four independent resolution lanes (o48 to o96, o96 to o320, o320 to o1280,
and o1280 to o2560); they are separate models, not a cascade.

This repository is code. Decisions, open work and past verdicts live in the separate
docs repository at `~/dev/docs` on `hpc-login`; its `AGENTS.md` sets the reading order
for a working session, and the `AGENTS.md` next to this file gives the rules for
working in this repository.

## Quickstart

Run everything with the certified interpreter, `env -u PYTHONPATH
~/dev/.ds-260612/bin/python`, from the repository root. Below it is written as
`python`. The login node's own `python3` is too old to import this code.

```bash
# 1. What can be evaluated? Every evaluator, its question, role and host limit.
python -m eval.cli list

# 2. How does one evaluator work? Method, inputs, outputs, lane keys, an example.
python -m eval.cli describe tc

# 3. What would a run do? Prints the fully resolved configuration and stops.
python -m eval.cli run --lane o96_o320 --checkpoint <checkpoint> --dry-run

# 4. Evaluate predictions you already have.
python -m eval.cli evaluate --lane o96_o320 --predictions-dir <run>/predictions

# 5. Rank the result and diff it against the lane baseline.
python -m eval.cli scoreboard --lane o96_o320 --eval-dir <run> --vs-baseline
```

`python -m eval.cli --help` lists all commands in groups (pipeline, comparison,
tropical cyclone tracks, figures, maintenance), and `python -m eval.cli <command>
--help` gives the flags of one. Add `--dry-run` to any lane-based command to see
what it resolved without running it. `python -m eval.cli config <lane>` prints a
lane's configuration after its `base:` chain is merged.

## Directory map

| Directory | What it holds |
|---|---|
| `eval/` | The evaluation framework, described below |
| `manual_inference/` | Inference from a checkpoint and a prebuilt input bundle, for the unified checkpoint lineage |
| `manual_inference_legacy_ds/` | A frozen copy of the same package for the older single-dataset ("ds") checkpoints; it is installed under the name `manual_inference` where those checkpoints run, so do not restructure it |
| `interp/` | Interpretability tools, run as `python -m interp <tool>` |
| `distributed/` | Small helpers for distributed (multi-process) runs |
| `tools/` | Stand-alone toolkits: `aifsens2_regen` (rebuilds the AIFS ensemble datasets) and `station_head` (building code for the station head adapter) |
| `scripts/` | One-off analysis and figure scripts written for single investigations |
| `tests/` | GPU overfit smoke jobs (`tests/overfit/`); the unit tests live next to the code they test |
| `conftest.py`, `pytest.ini`, `sitecustomize.py` | Test configuration and interpreter start-up hooks; see `TESTING.md` |

### Inside `eval/`

| Path | What it holds |
|---|---|
| `eval/cli/` | The command line, `python -m eval.cli`: one module per command, see `eval/cli/__init__.py` |
| `eval/evaluators/` | One package per evaluator. `registry.py` is the one list of evaluators, `base.py` is the contract each package follows, `describe.py` feeds `list` and `describe` |
| `eval/_backends/` | The computation behind most evaluators (kernels, loaders, plot code); never called directly |
| `eval/config/` | Lane, host, event and ladder YAML files (`lanes/`, `hosts/`, `events/`, `ladder/`) and `loader.py`, which resolves a lane and its `base:` chain |
| `eval/predict/` | Prediction generation from bundles (`main.py`) and through prepml (`prepml.py`) |
| `eval/prepare/` | Building truth-aware input bundles from source GRIB files |
| `eval/scoreboard/` | Collecting the scoring rows of the scored evaluators and formatting `scores.csv` and `scores.md` |
| `eval/baseline.py` | The lane baseline (the top of the lane scoreboard) and the `--vs-baseline` diff |
| `eval/lean_layout.py` | Projects an evaluator tree into the tidy run-root layout |
| `eval/report/` | The HTML report of a run |
| `eval/discovery/` | Finding prediction files and identifying checkpoints |
| `eval/shared/` | Grid and plotting helpers used by several evaluators |
| `eval/jobs/` | SLURM orchestration (`pipeline.py`, `renderer.py`, `resources.py`), the ladder and evolution figures, `scripts/` for one-off jobs, `templates/` for sbatch templates |
| `eval/tools/` | `parity/`, a checker that diffs two scoreboards |
| `eval/tests/` | The unit tests of the framework as a whole |
| `eval/archive/` | Frozen legacy scripts that a few tests and one backend still import |
| `eval/notebooks/` | Example notebooks |
| `eval/_quarantine/` | Retired code, see below |

More detail: `ARCHITECTURE.md` (how the pieces fit), `eval/README.md` (the evaluation
harness and the rules for retiring a lane), `eval/config/lanes/README.md` (lane
inheritance), `eval/jobs/README.md` and `eval/predict/README.md`.

## Where retired code lives

Code that was retired on purpose is kept, not deleted, under
`eval/_quarantine/<date>/`, with a `README.md` there that says what each item was and
what replaces it. Nothing in it can be imported and its tests are not collected. The
retired evaluators (`spectra`, `spectra_ecmwf`, `sigma`, `obs_crps`, `mechanistic`,
`intermediate`, `interp`, `leadtime`) still have an entry in
`eval/evaluators/registry.py`: naming one with `--only` prints its replacement and
exits with status 1. Frozen but still imported code is in `eval/archive/`.

## Tests

```bash
python -m pytest -m "not gpu"     # every test folder of the repository, on CPU
```

`TESTING.md` explains the layout, the GPU tests, and the table of tests that are
expected to fail and why.
