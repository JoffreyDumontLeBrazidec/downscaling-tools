#!/usr/bin/env python3
"""Generate the ENFO O320 scoreboard from evaluation artifacts."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from eval.jobs import scoreboard_metrics as metrics


def _fmt(x: float, precision: int = 4) -> str:
    """Format a float for display, showing 'na' for non-finite values."""
    if not math.isfinite(x):
        return "na"
    if precision == 3:
        return f"{x:.3f}"
    elif precision == 4:
        return f"{x:.4f}"
    elif precision == 6:
        return f"{x:.6e}"
    else:
        return f"{x:.{precision}f}"


def load_sigma_results(eval_root: Path) -> dict[str, dict[str, float]]:
    """Load sigma evaluation CSV files keyed by the emitting run identifier."""
    results: dict[str, dict[str, float]] = {}
    sigma_dir = eval_root / "scoreboards" / "sigma"
    if not sigma_dir.exists():
        return results

    for csv_file in sorted(sigma_dir.glob("*_sigma_eval.csv")):
        run_id = csv_file.stem.replace("_sigma_eval", "")
        sigma_losses = metrics.load_sigma_losses_from_csv(csv_file)
        if sigma_losses:
            results[run_id] = sigma_losses
    return results


def load_tc_results(eval_root: Path) -> dict[str, dict[str, Any]]:
    """Load TC stats with the same fallback used by the docs-side scoreboard flow."""
    results: dict[str, dict[str, Any]] = {}

    def tc_candidate_rank(path: Path) -> tuple[int, int, int, int, str]:
        text = str(path).lower()
        name = path.name.lower()
        return (
            1 if "proxy" in text else 0,
            1 if any(token in text for token in ("smoketest", "cached_refs", "template", "submit_helper")) else 0,
            1 if any(token in text for token in ("full75", "aug16_30")) else 0,
            0 if any(token in name for token in ("from_predictions", "full25")) else 1,
            0 if "idalia_franklin" in name else 1,
            text,
        )

    for run_root in sorted(path for path in eval_root.glob("manual_*") if path.is_dir()):
        if not (run_root / "surface_loss_summary.json").exists():
            continue
        tc_scores: dict[str, float] = {}
        candidates = sorted(run_root.rglob("tc_normed_pdfs_*.stats.json"), key=tc_candidate_rank)
        for stats_file in candidates:
            parsed = metrics.load_tc_extreme_scores_from_json(stats_file, run_id=run_root.name)
            for event_name, score in parsed.items():
                tc_scores.setdefault(event_name, score)
            if {"idalia", "franklin"} <= tc_scores.keys():
                break
        if tc_scores:
            results[run_root.name] = {f"{event_name}_extreme": score for event_name, score in tc_scores.items()}
    return results


def load_spectra_results(eval_root: Path) -> dict[str, float]:
    """Load full25 spectra metrics using the 3-field scoreboard contract."""
    results: dict[str, float] = {}

    for run_root in sorted(path for path in eval_root.glob("manual_*") if path.is_dir()):
        if not (run_root / "surface_loss_summary.json").exists():
            continue
        spectra_dirs = sorted(run_root.glob("spectra_step120_5dates_m10_ecmwf*"))
        for spectra_dir in spectra_dirs:
            try:
                spectra_data = metrics.load_spectra_metrics(spectra_dir)
            except ValueError as exc:
                print(f"Skipping incompatible spectra dir {spectra_dir}: {exc}")
                continue
            mean_l2 = spectra_data.get("mean")
            if mean_l2 is None:
                continue
            results[run_root.name] = float(mean_l2)
            break
    return results


def load_surface_loss_results(eval_root: Path) -> dict[str, float]:
    """Load surface loss JSON files."""
    results: dict[str, float] = {}

    for summary_file in sorted(eval_root.glob("*/surface_loss_summary.json")):
        run_id = summary_file.parent.name
        surface_loss = metrics.load_surface_weighted_mse(summary_file)
        if surface_loss is not None:
            results[run_id] = surface_loss
    return results


def _nan() -> float:
    return float("nan")


def _row_token(run_id: str) -> str:
    return metrics.extract_checkpoint_token(run_id) or run_id


def _row_key_map(run_ids: set[str]) -> dict[str, str]:
    tokens = {
        token
        for run_id in run_ids
        if run_id and (token := metrics.extract_checkpoint_token(run_id))
    }
    canonical_token: dict[str, str] = {}
    for token in tokens:
        related = [other for other in tokens if other.startswith(token) or token.startswith(other)]
        canonical_token[token] = max(related, key=len) if related else token

    mapped: dict[str, str] = {}
    for run_id in run_ids:
        token = metrics.extract_checkpoint_token(run_id)
        mapped[run_id] = canonical_token.get(token, run_id) if token else run_id
    return mapped


def _run_rank(run_id: str) -> tuple[int, int, int, int, str]:
    return (
        0 if run_id.startswith("manual_") else 1,
        1 if "proxy" in run_id else 0,
        0 if "new_o96_o320" in run_id else 1,
        -len(run_id),
        run_id,
    )


def _prefer_run_id(existing: str | None, candidate: str) -> str:
    if existing is None:
        return candidate
    return candidate if _run_rank(candidate) < _run_rank(existing) else existing


def build_scoreboard_rows(
    sigma_data: dict[str, dict[str, float]],
    tc_data: dict[str, dict[str, Any]],
    spectra_data: dict[str, float],
    surface_loss_data: dict[str, float],
) -> list[dict[str, Any]]:
    all_run_ids = set(sigma_data) | set(tc_data) | set(spectra_data) | set(surface_loss_data)
    row_key_map = _row_key_map(all_run_ids)

    rows_by_key: dict[str, dict[str, Any]] = {}

    def ensure_row(run_id: str) -> dict[str, Any]:
        row_key = row_key_map[run_id]
        row = rows_by_key.get(row_key)
        if row is None:
            row = {
                "row_key": row_key,
                "display_run_id": None,
                "sigma_1": _nan(),
                "sigma_5": _nan(),
                "sigma_10": _nan(),
                "sigma_100": _nan(),
                "idalia_extreme": _nan(),
                "franklin_extreme": _nan(),
                "spectra_l2": _nan(),
                "surface_loss": _nan(),
            }
            rows_by_key[row_key] = row
        row["display_run_id"] = _prefer_run_id(row["display_run_id"], run_id)
        return row

    for run_id, sigma in sigma_data.items():
        row = ensure_row(run_id)
        for key in ("sigma_1", "sigma_5", "sigma_10", "sigma_100"):
            value = sigma.get(key)
            if value is not None and math.isfinite(value):
                row[key] = float(value)

    for run_id, tc in tc_data.items():
        row = ensure_row(run_id)
        for key in ("idalia_extreme", "franklin_extreme"):
            value = tc.get(key)
            if value is not None and math.isfinite(value):
                row[key] = float(value)

    for run_id, spectra_l2 in spectra_data.items():
        row = ensure_row(run_id)
        if math.isfinite(spectra_l2):
            row["spectra_l2"] = float(spectra_l2)

    for run_id, surface_loss in surface_loss_data.items():
        row = ensure_row(run_id)
        if math.isfinite(surface_loss):
            row["surface_loss"] = float(surface_loss)

    rows = sorted(rows_by_key.values(), key=lambda row: row["display_run_id"])
    for row in rows:
        token = metrics.extract_checkpoint_token(row["row_key"]) or metrics.extract_checkpoint_token(row["display_run_id"] or "")
        row["short_id"] = token[:8] if token else (row["display_run_id"] or row["row_key"])[:8]
        row["display_run_id"] = row["display_run_id"] or row["row_key"]
    return rows


def generate_scoreboard_markdown(rows: list[dict[str, Any]]) -> str:
    """Generate markdown table from normalized scoreboard rows."""
    lines = [
        "# ENFO O320 Scoreboard",
        "",
        "Standard 4-pillar evaluation protocol for o96→o320 downscaling runs.",
        "Lower is better for sigma, spectra, and surface loss; higher is better for TC extreme scores.",
        "",
        "| Ckpt | Run ID | σ=1 loss | σ=5 loss | σ=10 loss | σ=100 loss | TC Idalia extreme | TC Franklin extreme | Spectra L2 (mean) | Sfc Loss |",
        "|------|--------|----------|----------|-----------|------------|-------------------|---------------------|-------------------|----------|",
    ]

    for row in rows:
        line = (
            f"| {row['short_id']} | {row['display_run_id']} | "
            f"{_fmt(row['sigma_1'])} | {_fmt(row['sigma_5'])} | "
            f"{_fmt(row['sigma_10'])} | {_fmt(row['sigma_100'])} | "
            f"{_fmt(row['idalia_extreme'], 3)} | {_fmt(row['franklin_extreme'], 3)} | "
            f"{_fmt(row['spectra_l2'], 4)} | {_fmt(row['surface_loss'], 6)} |"
        )
        lines.append(line)
    
    lines.extend([
        "",
        "---",
        "",
        "**Legend:**",
        "- **Sigma loss**: Diffusion validation loss at fixed noise levels (σ=1,5,10,100)",
        "- **TC extreme**: TC extreme reproduction score for Idalia and Franklin events (0-1, higher is better)",
        "- **Spectra L2**: Mean relative L2 distance across 3 surface weather variables (10u, 10v, 2t) at step=120",
        "- **Sfc Loss**: Area-weighted and variable-weighted MSE over 8 surface variables",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate enfo_o320_scoreboard.md from evaluation artifacts"
    )
    parser.add_argument(
        "--eval-root",
        type=str,
        default="/home/ecm5702/perm/eval",
        help="Root directory containing evaluation outputs",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="/home/ecm5702/perm/eval/scoreboards/enfo_o320_scoreboard.md",
        help="Output markdown file path",
    )
    
    args = parser.parse_args()
    
    eval_root = Path(args.eval_root)
    out_md = Path(args.out_md)
    
    print(f"Loading evaluation data from: {eval_root}")

    sigma_data = load_sigma_results(eval_root)
    print(f"Loaded sigma data for {len(sigma_data)} runs")

    tc_data = load_tc_results(eval_root)
    print(f"Loaded TC data for {len(tc_data)} runs")

    spectra_data = load_spectra_results(eval_root)
    print(f"Loaded spectra data for {len(spectra_data)} runs")

    surface_loss_data = load_surface_loss_results(eval_root)
    print(f"Loaded surface loss data for {len(surface_loss_data)} runs")

    rows = build_scoreboard_rows(sigma_data, tc_data, spectra_data, surface_loss_data)
    markdown = generate_scoreboard_markdown(rows)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w") as f:
        f.write(markdown)

    print(f"\nScoreboard written to: {out_md}")


if __name__ == "__main__":
    main()
