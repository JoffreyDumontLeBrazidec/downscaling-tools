"""Backend-parity diagnostics (debug-only).

This package is NOT registered in any `evaluator_groups` and is NOT invoked by
the standard `eval.cli` evaluation pipeline. It exists to help diagnose
backend equality questions during development (e.g. manual vs prepml).

Tools for comparing two evaluation runs side by side:
- `scoreboard_diff` — per-(evaluator, metric) delta between two scoreboard CSVs,
  with optional tolerance / sampler-noise-floor filtering.

Use cases:
- Criterion 4: same-backend two-seed noise floor (left=seed-A, right=seed-B).
- Criterion 5: cross-backend parity (left=manual, right=prepml).

See `epics/checkpoint-eval-pipeline/in-progress/20260513_prepml_manual_inference_backend_equality.md`.
"""

from .scoreboard_diff import (
    ScoreboardDelta,
    ScoreboardDiffReport,
    diff_scoreboards,
    render_markdown_report,
)

__all__ = [
    "ScoreboardDelta",
    "ScoreboardDiffReport",
    "diff_scoreboards",
    "render_markdown_report",
]
