<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->

## graphify

This repo has a graphify knowledge graph at `graphify-out/` (synced 2026-06-17 from a Mac-built mirror).
- **Read `graphify-out/GRAPH_REPORT.md` first** to orient: god nodes (core abstractions), community structure, and surprising cross-file links — faster than grepping to find your way around.
- `graph.json` is the full queryable graph. Once the graphify CLI is installed here (`module load python3/3.11.8-01`, then pip-install `graphifyy` into a venv), use `graphify query "<question>"` for a scoped subgraph with source_location pointers, plus `graphify explain "<concept>"` and `graphify path "<A>" "<B>"`.
- Snapshot built 2026-06-16; rebuild incrementally with `graphify update .` after large changes.
