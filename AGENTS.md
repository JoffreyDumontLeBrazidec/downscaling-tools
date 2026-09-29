# Agent orientation — downscaling-tools

## Start here

- **`README.md`** — what the repository is, a directory map, and a five-command quickstart.
- **`python -m eval.cli list`** and **`python -m eval.cli describe <evaluator>`** — what can
  be evaluated and how each evaluator works. Generated from the registry, so never stale.
- **`docs/eval-repertoire/REPERTOIRE.md`** — one section and one example figure per eval.cli tool:
  the question it answers, its method, cost, how to read it, and overlaps (written 2026-09-29).
- **`ARCHITECTURE.md`** — the stack map and the four independent resolution lanes.
- **`eval/README.md`** — the evaluation harness (`eval.cli`), and the lane-retirement convention.
- **`eval/config/lanes/README.md`** — how `base:` inheritance works and why the generated
  `_ladder_*` files are not tracked.
- **Decisions, open work and prior verdicts live in `~/dev/docs`** on hpc-login — a separate
  repo, not vendored here. Its `AGENTS.md` owns the session reading order. This repo is code;
  that repo is the record.

`main` is the trunk. If you are on a long-lived feature branch, rebase early — this repo has
been 5 weeks behind its own trunk before.

## Runtime

The certified runtime is `~/dev/.ds-260612/bin/python`, invoked with `env -u PYTHONPATH`.
The login node's `python3` is 3.6.8 and **cannot import this codebase**.

```bash
env -u PYTHONPATH ~/dev/.ds-260612/bin/python -m eval.cli --help
```

## Tests

```bash
python -m pytest -m "not gpu"        # CPU suite
python -m pytest -m gpu --run-gpu    # GPU suite
```

A global hook caps any single test at 30 minutes. See `TESTING.md`.

`python -m pytest` at the repository root collects every test folder. It should end with
no failing test. A test that is expected not to pass is marked `xfail` with a reason, and
`TESTING.md` lists each one. If a test fails, you probably broke it: compare with the
merge base only if the test is not in that list. The legacy ds tests
(`manual_inference_legacy_ds/tests/`) run in a subprocess, because that tree must be
installed under the name `manual_inference`.

## House rules

- **Never `rm`.** Move aside to `~/attic/<YYYYMMDD>-<topic>/` and report the path. For tracked
  files, `git rm` is fine — history is the archive.
- Prefer correct defaults over blocking validators. Checks warn; they do not gate.
- Do not delete a lane without running the four checks in `eval/README.md` — one of them is
  "no queued or running job names it", and that is the one people skip.

## Adding or changing an evaluator

An evaluator is declared once, in `eval/evaluators/registry.py` (group, question, host
limit), and implemented as a package `eval/evaluators/<name>/` that follows the contract in
`eval/evaluators/base.py`: `run`, `score`, `plot` and `EVALUATOR_SPEC` (with `outputs`) at the
top of the package, and the computation in its `core/` subpackage (see `ARCHITECTURE.md`).
Use `no_score` and `no_plot` for what the evaluator does not have. Figures use the house
style in `eval/plotting/` (role colours, variable table, `save_figure`). Run
`python -m pytest eval/tests/test_evaluator_contract.py eval/tests/test_evaluator_registry.py`
and `python -m eval.cli describe <name>`. To retire one, move its package to
`eval/_quarantine/<date>/` and change its registry entry to `retired`.

## Optional: graphify knowledge graph

`graphify-out/` is a **local-only build artifact**. It is git-ignored and is **not** present in
a fresh clone. Skip this section if the directory is absent.

If `graphify-out/graph.json` exists, prefer it over grep for codebase questions:

- `graphify query "<question>"` — scoped subgraph with source locations
- `graphify path "<A>" "<B>"` — how two things relate
- `graphify explain "<concept>"` — one focused concept
- `graphify-out/GRAPH_REPORT.md` — only for broad architecture review

Rebuild with `graphify update .` after large changes (AST-only, no API cost).
