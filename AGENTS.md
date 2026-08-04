# Agent orientation — downscaling-tools

## Start here

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

Note: a number of tests currently fail for pre-existing reasons unrelated to any change you
are making. Before assuming you broke something, diff your failure set against the failure set
on your merge base.

## House rules

- **Never `rm`.** Move aside to `~/attic/<YYYYMMDD>-<topic>/` and report the path. For tracked
  files, `git rm` is fine — history is the archive.
- Prefer correct defaults over blocking validators. Checks warn; they do not gate.
- Do not delete a lane without running the four checks in `eval/README.md` — one of them is
  "no queued or running job names it", and that is the one people skip.

## Optional: graphify knowledge graph

`graphify-out/` is a **local-only build artifact**. It is git-ignored and is **not** present in
a fresh clone. Skip this section if the directory is absent.

If `graphify-out/graph.json` exists, prefer it over grep for codebase questions:

- `graphify query "<question>"` — scoped subgraph with source locations
- `graphify path "<A>" "<B>"` — how two things relate
- `graphify explain "<concept>"` — one focused concept
- `graphify-out/GRAPH_REPORT.md` — only for broad architecture review

Rebuild with `graphify update .` after large changes (AST-only, no API cost).
