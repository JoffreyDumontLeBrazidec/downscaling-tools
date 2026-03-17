#!/usr/bin/env python3
"""
Generate enfo_o320_scoreboard.md from sigma CSVs, TC stats, spectra summaries, and surface loss JSONs.

Reads all evaluation artifacts and produces a clean markdown table scoreboard.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any


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
    """
    Load sigma evaluation CSV files.
    
    Returns:
        {run_id: {sigma_1: loss, sigma_5: loss, ...}}
    """
    results = {}
    sigma_dir = eval_root / "scoreboards" / "sigma"
    
    if not sigma_dir.exists():
        return results
    
    csv_files = sorted(sigma_dir.glob("*_sigma_eval.csv"))
    
    for csv_file in csv_files:
        # Extract run_id from filename: <run_id>_sigma_eval.csv
        run_id = csv_file.stem.replace("_sigma_eval", "")
        
        sigma_losses = {}
        with csv_file.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                sigma = row.get("sigma", "")
                loss = row.get("loss", "")
                if sigma and loss:
                    try:
                        sigma_losses[f"sigma_{sigma}"] = float(loss)
                    except ValueError:
                        pass
        
        if sigma_losses:
            results[run_id] = sigma_losses
    
    return results


def load_tc_results(eval_root: Path) -> dict[str, dict[str, Any]]:
    """
    Load TC extreme stats JSON files.
    
    Returns:
        {run_id: {idalia_extreme: score, franklin_extreme: score, ...}}
    """
    results = {}
    
    # Find all TC stats files with idalia_franklin pattern
    stats_files = sorted(glob.glob(str(eval_root / "*" / "tc_normed_pdfs_idalia_franklin_*from_predictions.stats.json")))
    
    for stats_path in stats_files:
        stats_file = Path(stats_path)
        
        # Extract run_id from parent directory name
        run_id = stats_file.parent.name
        
        with stats_file.open() as f:
            data = json.load(f)
        
        events = data.get("events", {})
        tc_scores = {}
        
        # Extract extreme_score for idalia and franklin
        for event_name in ["idalia", "franklin"]:
            if event_name in events:
                event_data = events[event_name]
                extreme_tail = event_data.get("extreme_tail", {})
                rows = extreme_tail.get("rows", [])
                
                # Find the row for this run (look for matching run_id in exp field)
                for row in rows:
                    exp = row.get("exp", "")
                    # Match if exp contains the run_id or if it's the first non-reference row
                    if run_id in exp or (not exp.startswith("ENFO_O320") and not exp.startswith("ip6y")):
                        extreme_score = row.get("extreme_score")
                        if extreme_score is not None:
                            tc_scores[f"{event_name}_extreme"] = float(extreme_score)
                        break
        
        if tc_scores:
            results[run_id] = tc_scores
    
    return results


def load_spectra_results(eval_root: Path) -> dict[str, float]:
    """
    Load spectra summary JSON files.
    
    Returns:
        {run_id: mean_relative_l2_mean_curve}
    """
    results = {}
    
    # Find all spectra_summary.json files
    summary_files = sorted(glob.glob(str(eval_root / "*" / "spectra_*" / "spectra_summary.json")))
    
    for summary_path in summary_files:
        summary_file = Path(summary_path)
        
        # Extract run_id from grandparent directory name
        run_id = summary_file.parent.parent.name
        
        with summary_file.open() as f:
            data = json.load(f)
        
        # Extract relative_l2_mean_curve for each variable and compute mean
        l2_values = []
        for var_name, var_data in data.items():
            if isinstance(var_data, dict) and "relative_l2_mean_curve" in var_data:
                l2_val = var_data["relative_l2_mean_curve"]
                if math.isfinite(l2_val):
                    l2_values.append(l2_val)
        
        if l2_values:
            mean_l2 = sum(l2_values) / len(l2_values)
            results[run_id] = mean_l2
    
    return results


def load_surface_loss_results(eval_root: Path) -> dict[str, float]:
    """
    Load surface loss summary JSON files.
    
    Returns:
        {run_id: weighted_surface_mse}
    """
    results = {}
    
    # Find all surface_loss_summary.json files
    summary_files = sorted(glob.glob(str(eval_root / "*" / "surface_loss_summary.json")))
    
    for summary_path in summary_files:
        summary_file = Path(summary_path)
        
        # Extract run_id from parent directory name
        run_id = summary_file.parent.name
        
        with summary_file.open() as f:
            data = json.load(f)
        
        wsurf_mse = data.get("weighted_surface_mse")
        if wsurf_mse is not None:
            results[run_id] = float(wsurf_mse)
    
    return results


def generate_scoreboard_markdown(
    sigma_data: dict[str, dict[str, float]],
    tc_data: dict[str, dict[str, Any]],
    spectra_data: dict[str, float],
    surface_loss_data: dict[str, float],
) -> str:
    """
    Generate markdown table from all collected data.
    """
    # Collect all run_ids
    all_run_ids = set()
    all_run_ids.update(sigma_data.keys())
    all_run_ids.update(tc_data.keys())
    all_run_ids.update(spectra_data.keys())
    all_run_ids.update(surface_loss_data.keys())
    
    # Build table rows
    rows = []
    for run_id in sorted(all_run_ids):
        # Short ID (first 7 chars)
        short_id = run_id[:7] if len(run_id) > 7 else run_id
        
        # Sigma losses
        sigma = sigma_data.get(run_id, {})
        sigma_1 = sigma.get("sigma_1", float("nan"))
        sigma_5 = sigma.get("sigma_5", float("nan"))
        sigma_10 = sigma.get("sigma_10", float("nan"))
        sigma_100 = sigma.get("sigma_100", float("nan"))
        
        # TC extreme scores
        tc = tc_data.get(run_id, {})
        idalia_extreme = tc.get("idalia_extreme", float("nan"))
        franklin_extreme = tc.get("franklin_extreme", float("nan"))
        
        # Spectra
        spectra_l2 = spectra_data.get(run_id, float("nan"))
        
        # Surface loss
        surface_loss = surface_loss_data.get(run_id, float("nan"))
        
        rows.append({
            "run_id": run_id,
            "short_id": short_id,
            "sigma_1": sigma_1,
            "sigma_5": sigma_5,
            "sigma_10": sigma_10,
            "sigma_100": sigma_100,
            "idalia_extreme": idalia_extreme,
            "franklin_extreme": franklin_extreme,
            "spectra_l2": spectra_l2,
            "surface_loss": surface_loss,
        })
    
    # Generate markdown
    lines = [
        "# ENFO O320 Scoreboard",
        "",
        "Standard 4-pillar evaluation protocol for o96→o320 downscaling runs.",
        "Lower is better for all metrics.",
        "",
        "| Run | Ckpt | σ=1 loss | σ=5 loss | σ=10 loss | σ=100 loss | TC Idalia extreme | TC Franklin extreme | Spectra L2 (mean) | Sfc Loss |",
        "|-----|------|----------|----------|-----------|------------|-------------------|---------------------|-------------------|----------|",
    ]
    
    for row in rows:
        line = (
            f"| {row['short_id']} | {row['run_id']} | "
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
        "- **TC extreme**: TC extreme reproduction score for Idalia and Franklin events (0-1, higher is better displayed, but stored as distance metric)",
        "- **Spectra L2**: Mean relative L2 distance across 6 weather variables (step=120)",
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
    
    # Load all data
    sigma_data = load_sigma_results(eval_root)
    print(f"Loaded sigma data for {len(sigma_data)} runs")
    
    tc_data = load_tc_results(eval_root)
    print(f"Loaded TC data for {len(tc_data)} runs")
    
    spectra_data = load_spectra_results(eval_root)
    print(f"Loaded spectra data for {len(spectra_data)} runs")
    
    surface_loss_data = load_surface_loss_results(eval_root)
    print(f"Loaded surface loss data for {len(surface_loss_data)} runs")
    
    # Generate markdown
    markdown = generate_scoreboard_markdown(
        sigma_data, tc_data, spectra_data, surface_loss_data
    )
    
    # Write output
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w") as f:
        f.write(markdown)
    
    print(f"\nScoreboard written to: {out_md}")


if __name__ == "__main__":
    main()
