# ============================
# Sigma-eval plotter (runnable)
# - choose experiments to load
# - choose metrics to plot
# - prints available metrics
# ============================

from __future__ import annotations


import os
from pathlib import Path
from tkinter import ALL
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ----------------------------
# Config: base dir for run ids
# ----------------------------
DIR_EXP = Path("/home/ecm5702/scratch/aifs/checkpoint")


# ----------------------------
# Reader + cleaner
# ----------------------------
def clean_sigma_table(df: pd.DataFrame, csv_path: Path | str = "<in-memory>") -> pd.DataFrame:
    required = ["sigma", "prediction_on_pure_noise", "loss"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}; got {list(df.columns)}")

    df = df.copy()

    # cast boolean-ish column
    df["prediction_on_pure_noise"] = (
        df["prediction_on_pure_noise"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    # numeric coercion for all others
    for c in df.columns:
        if c != "prediction_on_pure_noise":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    bad = df[required].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"{csv_path}: contains NaNs after coercion:\n{df.loc[bad, required]}")
    return df


def read_sigma_table_from_runfile(
    run_and_csv: str,
    *,
    dir_exp: Path = DIR_EXP,
    exp_label: str | None = None,
) -> pd.DataFrame:
    """
    run_and_csv: "RUN_ID/sigma_eval_table_XXXX.csv" (relative to dir_exp)
    """
    run_and_csv = run_and_csv.strip().lstrip("/")
    csv_path = (dir_exp / run_and_csv).resolve()

    dir_exp = dir_exp.resolve()
    if dir_exp not in csv_path.parents:
        raise ValueError(f"Refusing path outside dir_exp: {csv_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    df = clean_sigma_table(df, csv_path=csv_path)

    run_id = csv_path.parent.name
    df.insert(0, "exp", exp_label if exp_label is not None else run_id)
    df.insert(1, "run_id", run_id)
    df.insert(2, "source_csv", csv_path.name)
    return df


def build_all_df(specs: Sequence[str] | Sequence[tuple[str, str]], *, dir_exp: Path = DIR_EXP) -> pd.DataFrame:
    """
    Two accepted formats:
      1) list[str]:
         ["RUNID/file.csv", ...]           -> exp defaults to run_id
      2) list[tuple[str,str]]:
         [("my_label", "RUNID/file.csv")]  -> exp = my_label
    """
    if len(specs) == 0:
        raise ValueError("specs is empty")

    frames: list[pd.DataFrame] = []
    for item in specs:
        if isinstance(item, tuple):
            exp_label, run_and_csv = item
        else:
            exp_label, run_and_csv = None, item

        frames.append(
            read_sigma_table_from_runfile(
                run_and_csv,
                dir_exp=dir_exp,
                exp_label=exp_label,
            )
        )

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["exp", "prediction_on_pure_noise", "sigma"]).reset_index(drop=True)
    return all_df


# ----------------------------
# Metrics helpers
# ----------------------------
def available_metrics(df: pd.DataFrame) -> list[str]:
    """
    Returns candidate metrics to plot (numeric columns, excluding metadata).
    """
    exclude = {"exp", "run_id", "source_csv", "prediction_on_pure_noise"}
    # keep only numeric columns
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    # commonly you want sigma on x-axis, so exclude from y candidates
    y_candidates = [c for c in numeric_cols if c != "sigma"]
    return sorted(y_candidates)


def print_available_metrics(df: pd.DataFrame, *, max_show: int = 200) -> None:
    mets = available_metrics(df)
    print(f"Available metrics ({len(mets)}):")
    if len(mets) <= max_show:
        for m in mets:
            print(" -", m)
    else:
        for m in mets[:max_show]:
            print(" -", m)
        print(f"... ({len(mets) - max_show} more)")


# ----------------------------
# Plotting
# ----------------------------


def save_sigma_curves_pdf(
    all_df: pd.DataFrame,
    metrics: Sequence[str],
    pdf_path: str | Path,
    *,
    sigma_min: float = 0.02,
    pred_flags: Sequence[bool] = (False, True),
    agg: str = "mean",  # "mean" or "median"
    figsize: tuple[float, float] = (22, 10),
    title_size: int = 26,
    label_size: int = 22,
    tick_size: int = 18,
    legend_size: int = 18,
    line_width: float = 2.5,
    marker_size: float = 7.0,
    legend_outside: bool = True,
    pdf_dpi: int = 300,  # only matters for rasterized artists; vector PDF zooms well anyway
) -> Path:
    pdf_path = Path(pdf_path)

    # validate metrics
    missing = [m for m in metrics if m not in all_df.columns]
    if missing:
        raise ValueError(f"Requested metrics not in dataframe: {missing}")

    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")

    n_pages = 0
    with PdfPages(pdf_path) as pdf:
        for metric in metrics:
            for pred_flag in pred_flags:
                sub = all_df[all_df["prediction_on_pure_noise"] == pred_flag]
                sub = sub[sub["sigma"] > sigma_min]

                if sub.empty:
                    print(
                        f"[WARN] No rows after filtering for metric={metric}, "
                        f"pred_flag={pred_flag}, sigma_min={sigma_min}"
                    )
                    continue

                if agg == "mean":
                    g = sub.groupby(["exp", "sigma"], as_index=False)[metric].mean()
                else:
                    g = sub.groupby(["exp", "sigma"], as_index=False)[metric].median()

                fig, ax = plt.subplots(figsize=figsize)

                for exp, gg in g.groupby("exp"):
                    gg = gg.sort_values("sigma")
                    ax.plot(
                        gg["sigma"],
                        gg[metric],
                        marker="o",
                        linewidth=line_width,
                        markersize=marker_size,
                        label=exp,
                    )

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("sigma", fontsize=label_size)
                ax.set_ylabel(metric, fontsize=label_size)
                ax.set_title(
                    f"{metric} – prediction_on_pure_noise={pred_flag}, σ > {sigma_min} ({agg})",
                    fontsize=title_size,
                )
                ax.tick_params(labelsize=tick_size)

                if legend_outside:
                    ax.legend(
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        frameon=True,
                        fontsize=legend_size,
                    )
                    fig.tight_layout(rect=[0, 0, 0.8, 1])
                else:
                    ax.legend(fontsize=legend_size)
                    fig.tight_layout()

                # Write this page to the PDF (vector by default)
                pdf.savefig(fig, bbox_inches="tight", dpi=pdf_dpi)
                plt.close(fig)
                n_pages += 1

        # optional: embed metadata
        d = pdf.infodict()
        d["Title"] = "Sigma evaluation curves"
        d["Creator"] = "matplotlib"

    if n_pages == 0:
        raise RuntimeError("No pages were written (all plots empty after filtering).")

    return pdf_path


# ----------------------------
# Experiment selection
# ----------------------------
def filter_specs(
    all_specs: Sequence[tuple[str, str]],
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Filter by exp label (first element of tuple).
    include/exclude are lists of labels; if include is None -> keep all.
    """
    specs = list(all_specs)

    if include is not None:
        include_set = set(include)
        specs = [s for s in specs if s[0] in include_set]

    if exclude is not None:
        exclude_set = set(exclude)
        specs = [s for s in specs if s[0] not in exclude_set]

    if not specs:
        raise ValueError("No experiments left after filtering.")
    return specs


# --------------------s--------
# Example usage
# ----------------------------
ALL_SPECS: list[tuple[str, str]] = [
    (
        "end 1e6 pretraining",
        "8ef18bfef7804fcda809f032261a2d39/sigma_eval_table_998784.csv",
    ),
    (
        "ft noise 100 e43a0c68c14e41c48d685c2e77d4c0bd",
        "4a5b2f1b24b84c52872bfcec1410b00f/sigma_eval_table__100000.csv",
    ),
    (
        "ft noise 1000 e1112845862540318da83bfac1c90841",
        "4a5b2f1b24b84c52872bfcec1410b00f/sigma_eval_table__100000.csv",
    ),
    (
        "ft noise 10000 4a5b2f1b24b84c52872bfcec1410b00f",
        "4a5b2f1b24b84c52872bfcec1410b00f/sigma_eval_table_100000.csv",
    ),
]

ALL_SPECS: list[tuple[str, str]] = [
    (
        "iz2r (epoch 21, step 101728)",
        "ec4d16fb6f8c402992e1e29ec7ddfc0e/sigma_eval_table_101728.csv",
    ),
    (
        "iyl7 (epoch 21, step 101728)",
        "69f04ba99f034d3894787fb845159dbf/sigma_eval_table_101728.csv",
    ),
    (
        "j10e (epoch 21, step 100000)",
        "4a5b2f1b24b84c52872bfcec1410b00f/sigma_eval_table_100000.csv",
    ),
    (
        "iz2s (epoch 21, step 100000, by_time)",
        "d980b237c109481b9ae432d762967ac2/sigma_eval_table_100000.csv",
    ),
]

ALL_SPECS: list[tuple[str, str]] = [
    (
        "1e6 iter training - low noise lognormal",
        "8ef18bfef7804fcda809f032261a2d39/sigma_eval_table_998784.csv",
    ),
    (
        "1e5 iter training - low noise lognormal",
        "3aeb36edfa954fcf81f5d1db2d82b72f/sigma_eval_table_100000.csv",
    ),
    (
        "1.16 iter training - low noise lognormal then finetuning high noise",
        "4a5b2f1b24b84c52872bfcec1410b00f/sigma_eval_table_100000.csv",
    ),
    (
        "1e5 iter training - moderate noise lognormal",
        "f27ed37e541c4b86bc80610a175abca0/sigma_eval_table_100000.csv",
    ),
]


# Choose which experiments to load:
SELECT_EXPS = None  # Select all experiments
specs = filter_specs(ALL_SPECS, include=SELECT_EXPS)

# Load
all_df = build_all_df(specs, dir_exp=DIR_EXP)
print(all_df.head())
print("rows:", len(all_df), "exps:", all_df["exp"].nunique())

# Print which metrics exist
print_available_metrics(all_df)

# Choose metrics to plot (examples)
METRICS_TO_PLOT = [
    "loss",
    "metric__diff_all_var_non_weighted",
    "metric__mse_metric/sfc_10u/1",
    "metric__mse_10u_non_weighted",
    "metric__mse_metric/sfc_10v/1",
    "metric__mse_10v_non_weighted",
    "metric__mse_metric/z_500/1",
    "metric__mse_z_500_non_weighted",
    "metric__mse_metric/sfc_2t/1",
    "metric__mse_2t_non_weighted",
]

pdf_path = save_sigma_curves_pdf(
    all_df,
    metrics=METRICS_TO_PLOT,
    pdf_path="sigma_eval_plots.pdf",
    sigma_min=0.02,
    figsize=(22, 10),  # big pages
    legend_outside=True,  # keeps curves large
)
print("Wrote:", pdf_path)
