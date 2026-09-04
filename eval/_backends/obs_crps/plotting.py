"""Plots for the observation CRPS evaluator.

One page per parameter, one panel per domain, fair CRPS against lead time with
the forecast start dates averaged out. Three curves are drawn on every panel: the
experiment, the coarse input that fed the downscaler, and the high-resolution
reference. A curve on its own says only how good the forecast was; the three
together say what the downscaling added, which is the question being asked.

The colours follow the existing quaver scorecards so the two can be read side by
side: the experiment is black, the input orange and the reference blue.
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
CURVE_STYLE = {
    "experiment": ("#000000", "-", "experiment"),
    "input": ("#ff7f00", "--", "input (coarse driver)"),
    "reference": ("#377eb8", "-.", "reference (high resolution)"),
}
CURVE_ORDER = ["input", "reference", "experiment"]


def _curves_present(frame: pd.DataFrame) -> list[str]:
    if "curve" not in frame:
        return ["experiment"]
    return [c for c in CURVE_ORDER if c in set(frame["curve"])]


def plot_obs_crps_summary(
    summary_csv: str | Path,
    rows_csv: str | Path,
    out_pdf: str | Path,
    title_suffix: str = "",
) -> Path:
    summary = pd.read_csv(summary_csv)
    if "curve" not in summary:
        summary["curve"] = "experiment"
    rows = pd.read_csv(rows_csv) if Path(rows_csv).exists() else None
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    curves = _curves_present(summary)

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
                per_domain = per_param[per_param["domain"] == domain]
                ndates = 0
                for curve in curves:
                    line = per_domain[per_domain["curve"] == curve].sort_values("step")
                    if line.empty:
                        continue
                    colour, style, label = CURVE_STYLE[curve]
                    ax.plot(line["step"], line["fcrps"], style, marker="o", ms=3,
                            color=colour, label=label)
                    if curve == "experiment" and "fcrps_std" in line:
                        if line["fcrps_std"].notna().any():
                            ax.fill_between(
                                line["step"],
                                line["fcrps"] - line["fcrps_std"],
                                line["fcrps"] + line["fcrps_std"],
                                color=colour, alpha=0.12,
                            )
                    if "ndates" in line:
                        ndates = max(ndates, int(line["ndates"].max()))
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

        # What the downscaling added, as a percentage of the input's own error.
        if "input" in curves and "experiment" in curves:
            for parameter, per_param in summary.groupby("parameter"):
                fig, ax = plt.subplots(figsize=(8.0, 4.5))
                for domain in sorted(per_param["domain"].unique()):
                    per_domain = per_param[per_param["domain"] == domain]
                    exp = per_domain[per_domain["curve"] == "experiment"].set_index("step")
                    inp = per_domain[per_domain["curve"] == "input"].set_index("step")
                    common = exp.index.intersection(inp.index)
                    if common.empty:
                        continue
                    gain = 100.0 * (inp.loc[common, "fcrps"] - exp.loc[common, "fcrps"]) \
                        / inp.loc[common, "fcrps"]
                    ax.plot(common, gain, "-o", ms=3, label=domain)
                ax.axhline(0, color="k", lw=0.8)
                ax.set_xlabel("lead time (hours)")
                ax.set_ylabel("improvement over the input (per cent)")
                ax.set_title(f"{parameter}: what the downscaling added")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        if rows is not None and not rows.empty:
            if "curve" in rows:
                rows = rows[rows["curve"] == "experiment"]
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
                ax.set_title(
                    f"{parameter}, northern hemisphere, experiment, one line per start date"
                )
                ax.grid(alpha=0.3)
                ax.legend(fontsize=6, ncol=4)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    LOG.info("obs_crps plots written to %s", out_pdf)
    return out_pdf
