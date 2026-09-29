# Quarantine, 2026-09-29

Code retired from the evaluation framework on 2026-09-29, after the owner judged the
tools of the framework in a trial. It is kept, not deleted, so its history and logic
stay readable. None of it is importable under its old name, and its tests are not
collected (`norecursedirs` in `pytest.ini`).

Each retired command still parses and is a tombstone: it prints where the tool went
and exits with status 1. The tombstones are in `eval/cli/retired.py`.

| Directory | What it was | Use instead |
|---|---|---|
| `report/` | `eval.cli report`, which wrote an HTML page of a run's PDF figures and scores. Its file discovery (top-level `plots/*.pdf` and `data/scoreboard/scores.csv`) no longer matched the lean run layout, so it wrote an empty page. `__init__.py` was `eval/report/__init__.py`, `cli_report.py` was `eval/cli/report.py`, and `tests/test_report.py` holds the two report tests that were in `eval/tests/test_plotting_spec_helpers.py`. | nothing |
| `videogen/` | `eval.cli videogen`, which rendered MP4 videos of the predictions of one storm from scenes tied to an old checkpoint. `videogen/` was `eval/tools/videogen/` and `cli_videogen.py` was `eval/cli/videogen.py`. It had no tests. `eval/tools/videogen/` now holds only a tombstone for `python -m eval.tools.videogen`. | nothing |

## Renamed on the same day (nothing quarantined)

The evaluator and subcommand `membermaps` were renamed `zoom_maps`; the code moved
from `eval/evaluators/membermaps/` to `eval/evaluators/zoom_maps/` and from
`eval/cli/membermaps.py` to `eval/cli/zoom_maps.py`. The old names are tombstones:
`eval.cli membermaps` and `eval.cli evaluate --only membermaps` print the new name and
exit with status 1. A lane that still holds a `membermaps:` block, such as the tracked
`eval/config/lanes/o320_o1280.yaml`, keeps working: `zoom_maps` reads that block when
the lane has no `zoom_maps:` block, with a deprecation warning. The registry entry is
in `eval/evaluators/registry.py` (`renamed=True`).
