"""Scoreboard-level parity diff.

Loads two scoreboard CSVs (the canonical `scoreboard/scores.csv` produced by
`eval.cli` runs) and computes per-(evaluator, metric) absolute + relative
deltas. Supports two distinct framings via the same call:

- Cross-backend parity: left=manual, right=prepml. The noise band must come
  from a separate same-backend two-seed run (criterion 4 in the task note).
- Same-backend noise floor: left=seed-A, right=seed-B. The output is itself
  the noise band; feed `result.max_abs_diff` to the cross-backend tolerance.

The output is structured (`ScoreboardDiffReport`) for programmatic use and
also renderable as a markdown table for inclusion in task notes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ScoreboardDelta:
    """Per-metric delta between left and right scoreboards."""

    evaluator: str
    metric: str
    left_value: float
    right_value: float
    abs_diff: float
    rel_diff: float
    unit: str = ""


@dataclass(frozen=True)
class ScoreboardDiffReport:
    """Full diff between two scoreboards."""

    left_path: Path
    right_path: Path
    shared: list[ScoreboardDelta]
    only_left: list[tuple[str, str]] = field(default_factory=list)
    only_right: list[tuple[str, str]] = field(default_factory=list)
    tolerance: float | None = None
    tolerance_kind: str = "absolute"  # "absolute" or "relative"

    @property
    def shared_count(self) -> int:
        return len(self.shared)

    @property
    def n_within(self) -> int:
        return sum(1 for d in self.shared if self._is_within(d))

    @property
    def n_outside(self) -> int:
        return self.shared_count - self.n_within

    def _is_within(self, delta: ScoreboardDelta) -> bool:
        if self.tolerance is None:
            return True
        if self.tolerance_kind == "absolute":
            return delta.abs_diff <= self.tolerance
        if self.tolerance_kind == "relative":
            return delta.rel_diff <= self.tolerance
        raise ValueError(f"Unsupported tolerance_kind {self.tolerance_kind!r}")

    def outside_tolerance(self) -> list[ScoreboardDelta]:
        return [d for d in self.shared if not self._is_within(d)]

    def per_evaluator_summary(self) -> dict[str, dict[str, float | int]]:
        """Return aggregate stats grouped by evaluator."""
        out: dict[str, dict[str, float | int]] = {}
        for d in self.shared:
            row = out.setdefault(d.evaluator, {
                "n": 0,
                "max_abs": 0.0,
                "max_rel": 0.0,
                "mean_abs": 0.0,
                "n_outside": 0,
            })
            row["n"] += 1
            row["max_abs"] = max(row["max_abs"], d.abs_diff)
            row["max_rel"] = max(row["max_rel"], d.rel_diff)
            row["mean_abs"] += d.abs_diff
            if not self._is_within(d):
                row["n_outside"] += 1
        for row in out.values():
            if row["n"]:
                row["mean_abs"] = row["mean_abs"] / row["n"]
        return out


def _load_scoreboard(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"evaluator", "metric", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Scoreboard {path} missing required columns: {sorted(missing)}; "
            f"got {list(df.columns)}"
        )
    return df


def diff_scoreboards(
    left_path: str | Path,
    right_path: str | Path,
    *,
    tolerance: float | None = None,
    tolerance_kind: str = "absolute",
) -> ScoreboardDiffReport:
    """Compute per-metric deltas between two scoreboards.

    Args:
        left_path, right_path: scoreboard CSV files (canonical eval.cli format).
        tolerance: optional threshold below which a delta is "within band".
            If None, every shared delta counts as within.
        tolerance_kind: "absolute" (compare `abs_diff`) or "relative"
            (compare `rel_diff`).

    Returns a `ScoreboardDiffReport`.
    """
    if tolerance_kind not in ("absolute", "relative"):
        raise ValueError(
            f"Unsupported tolerance_kind {tolerance_kind!r}; expected 'absolute' or 'relative'"
        )

    left_path = Path(left_path)
    right_path = Path(right_path)
    left_df = _load_scoreboard(left_path)
    right_df = _load_scoreboard(right_path)

    merged = left_df.merge(
        right_df, on=["evaluator", "metric"], suffixes=("_left", "_right"),
    )

    deltas: list[ScoreboardDelta] = []
    for row in merged.itertuples(index=False):
        lv = float(getattr(row, "value_left"))
        rv = float(getattr(row, "value_right"))
        abs_d = abs(lv - rv)
        denom = (abs(lv) + abs(rv)) / 2.0
        rel_d = abs_d / denom if denom > 0 else 0.0
        unit = getattr(row, "unit_left", "") or getattr(row, "unit_right", "")
        deltas.append(
            ScoreboardDelta(
                evaluator=row.evaluator,
                metric=row.metric,
                left_value=lv,
                right_value=rv,
                abs_diff=abs_d,
                rel_diff=rel_d,
                unit=str(unit),
            )
        )

    left_keys = set(zip(left_df["evaluator"], left_df["metric"]))
    right_keys = set(zip(right_df["evaluator"], right_df["metric"]))

    return ScoreboardDiffReport(
        left_path=left_path,
        right_path=right_path,
        shared=sorted(deltas, key=lambda d: (d.evaluator, d.metric)),
        only_left=sorted(left_keys - right_keys),
        only_right=sorted(right_keys - left_keys),
        tolerance=tolerance,
        tolerance_kind=tolerance_kind,
    )


def render_markdown_report(
    report: ScoreboardDiffReport,
    *,
    left_label: str = "left",
    right_label: str = "right",
    top_n: int = 10,
    only_outside: bool = False,
) -> str:
    """Render a `ScoreboardDiffReport` as a self-contained markdown block.

    Useful for pasting into task-note progress logs.
    """
    lines: list[str] = []
    lines.append(f"### Scoreboard diff: `{left_label}` vs `{right_label}`")
    lines.append("")
    lines.append(f"- left: `{report.left_path}`")
    lines.append(f"- right: `{report.right_path}`")
    lines.append(f"- shared metrics: {report.shared_count}")
    lines.append(f"- only in left: {len(report.only_left)}")
    lines.append(f"- only in right: {len(report.only_right)}")
    if report.tolerance is not None:
        kind = report.tolerance_kind
        lines.append(
            f"- tolerance ({kind}): {report.tolerance:g} — "
            f"{report.n_within} within / {report.n_outside} outside"
        )
    lines.append("")

    rows = report.outside_tolerance() if only_outside else report.shared
    rows_top = sorted(rows, key=lambda d: d.abs_diff, reverse=True)[:top_n]
    if rows_top:
        lines.append("| evaluator | metric | " f"{left_label} | {right_label} | abs_diff | rel_diff |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for d in rows_top:
            lines.append(
                f"| {d.evaluator} | {d.metric} | "
                f"{d.left_value:.6g} | {d.right_value:.6g} | "
                f"{d.abs_diff:.4g} | {d.rel_diff:.4g} |"
            )
        lines.append("")

    summary = report.per_evaluator_summary()
    if summary:
        lines.append("Per-evaluator summary:")
        lines.append("")
        lines.append("| evaluator | n | max_abs | mean_abs | max_rel | n_outside |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for ev, agg in sorted(summary.items()):
            lines.append(
                f"| {ev} | {int(agg['n'])} | {agg['max_abs']:.4g} | "
                f"{agg['mean_abs']:.4g} | {agg['max_rel']:.4g} | {int(agg['n_outside'])} |"
            )
        lines.append("")

    only_left = [f"{e}/{m}" for e, m in report.only_left]
    only_right = [f"{e}/{m}" for e, m in report.only_right]
    if only_left:
        lines.append(f"Only in `{left_label}`: {', '.join(only_left)}")
    if only_right:
        lines.append(f"Only in `{right_label}`: {', '.join(only_right)}")

    return "\n".join(lines).rstrip() + "\n"
