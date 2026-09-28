"""Scoreboard aggregator — collect scores from evaluator modules."""
from __future__ import annotations

import importlib
import logging
from pathlib import Path

from eval.evaluators import registry as evaluator_registry
from eval.scoreboard.types import ScoreRecord

LOG = logging.getLogger(__name__)

# The evaluators whose score() records reach scores.csv: the registry entries
# with feeds_scoreboard=True (eval/evaluators/registry.py). Not a second list.
SCOREBOARD_EVALUATORS = evaluator_registry.scoreboard_names()


def aggregate_scores(
    eval_dir: Path,
    lane_config: dict,
    evaluators: list[str] | None = None,
) -> list[ScoreRecord]:
    """Collect scores from evaluator score() functions.

    Scans eval_dir/evaluators/<name>/ for each evaluator that feeds the
    scoreboard according to the registry, calls its score(), and returns a
    sorted ScoreRecord list. Requested evaluators that do not feed the
    scoreboard (standard and diagnostic ones) are passed over silently;
    retired or unknown names are passed over with a warning.

    Args:
        eval_dir: Root evaluation directory containing evaluator outputs.
        lane_config: Lane configuration dict.
        evaluators: Optional filter — only include these evaluator names.
                    None means all scoreboard-eligible evaluators with results.

    Returns:
        List of ScoreRecord sorted by (evaluator, metric).
    """
    eval_dir = Path(eval_dir)
    target_evaluators = evaluators if evaluators is not None else SCOREBOARD_EVALUATORS

    all_records: list[ScoreRecord] = []

    for name in target_evaluators:
        entry = evaluator_registry.get(name)
        if entry is None:
            LOG.warning("Unknown evaluator: %s (not in eval/evaluators/registry.py)", name)
            continue
        if evaluator_registry.is_retired(name):
            LOG.warning("Skipping retired evaluator. %s", evaluator_registry.retired_message(name))
            continue
        if not entry.feeds_scoreboard:
            continue

        # Import evaluator module
        try:
            mod = importlib.import_module(f"eval.evaluators.{name}")
        except ImportError:
            LOG.warning("Cannot import evaluator module: eval.evaluators.%s", name)
            continue

        # Check results directory exists
        results_dir = eval_dir / "evaluators" / name
        if not results_dir.is_dir():
            continue

        # Call score()
        score_fn = getattr(mod, "score", None)
        if score_fn is None:
            LOG.warning("Evaluator %s has no score() function", name)
            continue

        eval_config = lane_config.get(name, {})
        try:
            raw_scores = score_fn(results_dir, lane_config, eval_config)
        except Exception:
            LOG.warning("Evaluator %s score() failed", name, exc_info=True)
            continue

        # Convert dicts to ScoreRecords
        for record in raw_scores:
            all_records.append(ScoreRecord(
                evaluator=name,
                metric=record["metric"],
                value=record["value"],
                unit=record["unit"],
            ))

    all_records.sort(key=lambda r: (r.evaluator, r.metric))
    return all_records
