"""Compare local probabilistic evaluator curves against exported reference curves."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


KEYS = ("step", "weather_state", "domain", "metric")


def _read_curve_csv(path: Path, *, value_column: str | None = None) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        metric = row.get("metric", "")
        domain = row.get("domain", "")
        weather_state = row.get("weather_state", "")
        step = row.get("step", "")
        if not all((metric, domain, weather_state, step)):
            continue
        col = value_column
        if col is None:
            col = "mean" if row.get("mean") not in (None, "") else "value"
        try:
            value = float(row[col])
        except (KeyError, TypeError, ValueError):
            continue
        out[(str(int(float(step))), weather_state, domain, metric)] = {**row, "value": value}
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "step",
        "weather_state",
        "domain",
        "metric",
        "local",
        "reference",
        "diff",
        "abs_diff",
        "rel_diff",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _summarize(rows: list[dict[str, Any]], *, missing_local: int, missing_reference: int) -> dict[str, Any]:
    by_metric: dict[str, list[float]] = defaultdict(list)
    by_series: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_metric[row["metric"]].append(float(row["abs_diff"]))
        by_series[f"{row['weather_state']}:{row['domain']}:{row['metric']}"].append(float(row["abs_diff"]))
    return {
        "matched_rows": len(rows),
        "missing_local": missing_local,
        "missing_reference": missing_reference,
        "mean_abs_diff_by_metric": {
            k: sum(v) / len(v) for k, v in sorted(by_metric.items()) if v
        },
        "max_abs_diff_by_metric": {
            k: max(v) for k, v in sorted(by_metric.items()) if v
        },
        "mean_abs_diff_by_series": {
            k: sum(v) / len(v) for k, v in sorted(by_series.items()) if v
        },
        "max_abs_diff_by_series": {
            k: max(v) for k, v in sorted(by_series.items()) if v
        },
    }


def _plot(rows: list[dict[str, Any]], out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["weather_state"], row["domain"], row["metric"])].append(row)
    with PdfPages(out_pdf) as pdf:
        for (weather_state, domain, metric), group in sorted(grouped.items()):
            group.sort(key=lambda r: int(r["step"]))
            steps = [int(r["step"]) for r in group]
            local = [float(r["local"]) for r in group]
            ref = [float(r["reference"]) for r in group]
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            ax.plot(steps, local, marker="o", label="local")
            ax.plot(steps, ref, marker="x", linestyle="--", label="reference")
            ax.set_title(f"{weather_state} / {domain} / {metric}")
            ax.set_xlabel("Lead time (h)")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            pdf.savefig(fig)
            plt.close(fig)


def compare(
    local_csv: Path,
    reference_csv: Path,
    out_dir: Path,
    *,
    local_value_column: str | None = None,
    reference_value_column: str | None = None,
) -> dict[str, Any]:
    local = _read_curve_csv(local_csv, value_column=local_value_column)
    reference = _read_curve_csv(reference_csv, value_column=reference_value_column)
    matched_keys = sorted(set(local) & set(reference))
    rows: list[dict[str, Any]] = []
    for key in matched_keys:
        local_value = float(local[key]["value"])
        reference_value = float(reference[key]["value"])
        diff = local_value - reference_value
        rel_diff = diff / reference_value if reference_value else math.nan
        rows.append({
            "step": key[0],
            "weather_state": key[1],
            "domain": key[2],
            "metric": key[3],
            "local": local_value,
            "reference": reference_value,
            "diff": diff,
            "abs_diff": abs(diff),
            "rel_diff": rel_diff,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = out_dir / "probabilistic_reference_comparison.csv"
    summary_json = out_dir / "probabilistic_reference_comparison.json"
    comparison_pdf = out_dir / "probabilistic_reference_overlay.pdf"
    _write_csv(comparison_csv, rows)
    if rows:
        _plot(rows, comparison_pdf)
    summary = _summarize(
        rows,
        missing_local=len(set(reference) - set(local)),
        missing_reference=len(set(local) - set(reference)),
    )
    summary.update({
        "local_csv": str(local_csv),
        "reference_csv": str(reference_csv),
        "comparison_csv": str(comparison_csv),
        "comparison_pdf": str(comparison_pdf) if rows else None,
    })
    summary_json.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-summary", required=True, type=Path)
    parser.add_argument("--reference-summary", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--local-value-column", default=None)
    parser.add_argument("--reference-value-column", default=None)
    args = parser.parse_args()
    summary = compare(
        args.local_summary,
        args.reference_summary,
        args.out_dir,
        local_value_column=args.local_value_column,
        reference_value_column=args.reference_value_column,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
