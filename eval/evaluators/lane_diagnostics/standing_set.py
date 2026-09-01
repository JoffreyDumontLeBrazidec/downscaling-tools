"""Figure 12: the standing tropical-cyclone set and the spectra panel.

The standing set on this lane is eight probability-density figures. They are the
two arms of the campaign, unguided and mass-only autoguidance at weight 1.3,
crossed with the two ways these distributions are normally read, as a ratio to
the operational analysis and as a raw density on a logarithmic axis, crossed
with the two variables. Wind speed gets its OWN figure rather than sharing one
with pressure, because a change that deepens a cyclone without strengthening its
wind is a different result from one that moves both, and it must be readable on
its own.

The spectra panel is reported in ABSOLUTE percentage points of deviation from
the target's spectrum. A spectral error on this lane is already small, so quoting
a change as a percentage of that small error makes a negligible cost sound real.
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

SUPPORT_TC = ("Support: one support for all curves, the regridded 0.25 degree "
              "comparison grid over the box 15-45N, 90-50W, forecast steps 24 to 120 h, "
              "5 initialisations 2025-09-26 to 2025-09-30, 10 ensemble members.")


def _arm_caption(arm_label: str, event_stats: dict, variable: str, mode: str) -> str:
    contract = event_stats.get("comparison_contract", {})
    n_dates = len(contract.get("start_dates", []))
    n_mem = contract.get("ensemble_members", "?")
    leads = contract.get("lead_times_hours", [])
    what = ("10 m WIND SPEED, on its own figure so it can be read without pressure "
            "beside it" if variable.startswith("wind") else "sea level pressure")
    how = ("Each curve is divided by the operational analysis's own distribution, so the "
           "flat line at one is the analysis and a curve above one means that field puts "
           "more of its probability at that value than the analysis does."
           if mode == "ratio" else
           "Raw probability densities on a logarithmic vertical axis, so the rare tail is "
           "visible rather than crushed against zero.")
    return (f"Distribution of {what} for the {arm_label}. {how} The curves are the "
            "downscaler, the IEKM 4.4 km target, and the ENFO 9 km ensemble that drove "
            "it, so the question the figure answers is whether the model's distribution "
            "has moved from its driver's towards the target's.\n"
            + SUPPORT_TC + f" Sample: {n_dates} initialisations x {n_mem} members x "
            f"{len(leads)} lead times.\n"
            "This is a POOLED distribution over all cases. It is the right object for a "
            "distribution question and the WRONG object for reading this lane's cyclone "
            "depth: a pooled extreme is set by the single best case in the campaign and "
            "makes the driver look as deep as the target. Figures 1 and 4 carry the "
            "per-case reading.")


def _slug(text: str) -> str:
    """A filename-safe form of a human-readable arm label."""
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in text.lower()]
    return "".join(keep).strip("_").replace("__", "_")


def _load_tc_stats(path: str | Path) -> dict:
    events = json.loads(Path(path).read_text()).get("events", {})
    for stats in events.values():
        if not stats.get("prediction_only"):
            return stats
    raise ValueError(f"no usable event statistics in {path}")


def build_tc_set(arms: dict, event: str, out_dir: Path):
    """Eight figures: two arms x two readings x two variables."""
    from eval._backends.tc.pdf_plot import plot_pdf_single_variable
    from eval._backends.tc.plot_config import resolve_plot_config

    figures, captions = [], {}
    letters = iter("abcdefgh")
    for arm_label, stats_path in arms.items():
        stats = _load_tc_stats(stats_path)
        cfg = resolve_plot_config(event, {"events": [event]})
        for variable, vshort in (("mslp_hpa", "pressure"), ("wind10m_ms", "wind")):
            for mode in ("ratio", "log"):
                cfg_arm = replace(cfg, plot_title=f"{event.capitalize()} — {arm_label}")
                fig = plot_pdf_single_variable(
                    cfg_arm, event_stats=stats, variable=variable, mode=mode)
                letter = next(letters)
                name = f"12{letter}_tc_{vshort}_{mode}_{_slug(arm_label)}.pdf"
                fig.savefig(out_dir / name, dpi=200)
                figures.append(fig)
                captions[f"12{letter}"] = {
                    "slug": f"tc {vshort} {mode} {arm_label}",
                    "file": name,
                    "caption": _arm_caption(arm_label, stats, variable, mode)}
    return figures, captions


def build_spectra_panel(spectra_cfg: dict, out_dir: Path):
    """One spectra figure, with the cost quoted in absolute percentage points."""
    import matplotlib.pyplot as plt

    source = None
    for candidate in spectra_cfg.get("candidates", []):
        if Path(candidate["path"]).exists():
            source = candidate
            break
    if source is None:
        LOG.error("no spectra source available: %s",
                  [c["path"] for c in spectra_cfg.get("candidates", [])])
        return [], {}

    curves = json.loads(Path(source["path"]).read_text())
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.4))
    rows = _spectra_rows(curves)
    if not rows:
        LOG.error("spectra source %s carried no usable curves", source["path"])
        plt.close(fig)
        return [], {}

    fields = [r["field"] for r in rows]
    xs = np.arange(len(rows))
    model = 100 * np.asarray([r["model_dev"] for r in rows])
    driver = 100 * np.asarray([r["input_dev"] for r in rows])
    width = 0.38
    axs[0].bar(xs - width / 2, driver, width, color="#1f77b4", alpha=0.85,
               label="interpolated 9 km driver")
    axs[0].bar(xs + width / 2, model, width, color="#d62728", alpha=0.85,
               label="downscaler")
    for x, v in zip(xs, model):
        axs[0].text(x + width / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=8.5)
    axs[0].set_xticks(xs)
    axs[0].set_xticklabels(fields)
    axs[0].set_ylabel("deviation from the target's spectrum (percentage points)")
    axs[0].set_title("How far each field's spectrum sits from the target's")
    axs[0].legend(fontsize=9)
    axs[0].grid(axis="y", alpha=0.25)

    gain = driver - model
    axs[1].bar(xs, gain, color="#2ca02c", alpha=0.85)
    for x, v in zip(xs, gain):
        axs[1].text(x, v + 0.05 * np.sign(v or 1), f"{v:+.1f}", ha="center", fontsize=9)
    axs[1].axhline(0.0, color="#000000", lw=0.9)
    axs[1].set_xticks(xs)
    axs[1].set_xticklabels(fields)
    axs[1].set_ylabel("percentage points of deviation removed by the model")
    axs[1].set_title("What the downscaler buys, in absolute percentage points")
    axs[1].grid(axis="y", alpha=0.25)

    fig.suptitle("12i. Spectra: how much fine-scale structure each field has",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    caption = (
        "The spectrum says how a field's variance is spread across spatial scales. What "
        "is plotted is the deviation of each field's spectrum from the target's, in "
        "ABSOLUTE PERCENTAGE POINTS, and the right-hand panel is how many of those points "
        "the downscaler removes. Percentage points, not a percentage of the error: these "
        "errors are already small, and quoting a change as a fraction of a small error "
        "has previously made a negligible cost read as a real one.\n"
        f"Support: {source['support']}. Source: {source['label']}, {source['path']}.\n"
        "Wind is listed as its own field here, not folded into a surface average."
    )
    name = "12i_spectra_panel.pdf"
    fig.savefig(out_dir / name, dpi=200)
    fig.text(0.012, 0.008, caption, fontsize=8.2, va="bottom", ha="left", color="#333333",
             wrap=True)
    return [fig], {"12i": {"slug": "spectra panel", "file": name, "caption": caption}}


def _spectra_rows(curves) -> list[dict]:
    """Pull per-field relative deviations out of whichever spectra product is present.

    Both spectra products store a per-field relative L2 deviation between a
    curve and the target's curve; the layouts differ, so this reads the shapes
    known to occur and returns an empty list rather than guessing.
    """
    rows: list[dict] = []
    if isinstance(curves, dict) and "fields" in curves:
        for field, entry in curves["fields"].items():
            if not isinstance(entry, dict):
                continue
            m = entry.get("prediction_relative_l2", entry.get("relative_l2"))
            i = entry.get("input_relative_l2")
            if m is not None and i is not None:
                rows.append({"field": field, "model_dev": float(m), "input_dev": float(i)})
    elif isinstance(curves, dict):
        for field, entry in curves.items():
            if isinstance(entry, dict) and "prediction_relative_l2" in entry \
                    and "input_relative_l2" in entry:
                rows.append({"field": field,
                             "model_dev": float(entry["prediction_relative_l2"]),
                             "input_dev": float(entry["input_relative_l2"])})
    return rows


def build_standing_set(eval_config: dict, out_dir: Path) -> dict:
    cfg = eval_config.get("standing_set", {})
    figures, captions = [], {}
    if cfg.get("tc_arms"):
        f, c = build_tc_set(cfg["tc_arms"], cfg.get("event", "humberto"), out_dir)
        figures += f
        captions.update(c)
    else:
        LOG.error("figure 12 TC set skipped: no standing_set.tc_arms configured")
    if cfg.get("spectra"):
        f, c = build_spectra_panel(cfg["spectra"], out_dir)
        figures += f
        captions.update(c)
    else:
        LOG.error("figure 12 spectra panel skipped: no standing_set.spectra configured")
    return {"figures": figures, "captions": captions}
