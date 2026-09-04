"""Plots for the observation CRPS evaluator.

One page per parameter, one panel per domain, fair CRPS against lead time with
the forecast start dates averaged out.  The shaded band is the spread across
those start dates, so a curve drawn from few dates cannot be mistaken for a
well-determined one.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

LOG = logging.getLogger(__name__)

UNITS = {"2t": "K", "2d": "K", "10ff": "m/s", "msl": "Pa"}


def plot_obs_crps_summary(
    summary_csv: str | Path,
    rows_csv: str | Path,
    out_pdf: str | Path,
    title_suffix: str = "",
) -> Path:
    summary = pd.read_csv(summary_csv)
    rows = pd.read_csv(rows_csv) if Path(rows_csv).exists() else None
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        for parameter, per_param in summary.groupby("parameter"):
            domains = sorted(per_param["domain"].unique())
            ncol = min(len(domains), 2) or 1
            nrow = (len(domains) + ncol - 1) // ncol
            fig, axes = plt.subplots(
                nrow, ncol, figsize=(6.0 * ncol, 4.0 * nrow), squeeze=False
            )
            unit = UNITS.get(parameter, "native")
            for ax, domain in zip(axes.ravel(), domains):
                curve = per_param[per_param["domain"] == domain].sort_values("step")
                ax.plot(curve["step"], curve["fcrps"], "-o", color="#377eb8",
                        label="fair CRPS")
                if "fcrps_std" in curve and curve["fcrps_std"].notna().any():
                    ax.fill_between(
                        curve["step"],
                        curve["fcrps"] - curve["fcrps_std"],
                        curve["fcrps"] + curve["fcrps_std"],
                        color="#377eb8", alpha=0.15,
                        label="spread across start dates",
                    )
                ndates = int(curve["ndates"].max()) if "ndates" in curve else 0
                ax.set_title(f"{domain} ({ndates} start dates)")
                ax.set_xlabel("lead time (hours)")
                ax.set_ylabel(f"fair CRPS ({unit})")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
            for ax in axes.ravel()[len(domains):]:
                ax.axis("off")
            fig.suptitle(
                f"{parameter} against station observations{title_suffix}", fontsize=13
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)

        if rows is not None and not rows.empty:
            for parameter, per_param in rows.groupby("parameter"):
                nhem = per_param[per_param["domain"] == "n.hem"]
                if nhem.empty:
                    continue
                fig, ax = plt.subplots(figsize=(8.0, 4.5))
                for date, per_date in nhem.groupby("date"):
                    ax.plot(per_date["step"], per_date["fcrps"], "-", alpha=0.5,
                            lw=1.0, label=str(date))
                ax.set_xlabel("lead time (hours)")
                ax.set_ylabel(f"fair CRPS ({UNITS.get(parameter, 'native')})")
                ax.set_title(f"{parameter}, northern hemisphere, one line per start date")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=6, ncol=4)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    LOG.info("obs_crps plots written to %s", out_pdf)
    return out_pdf
