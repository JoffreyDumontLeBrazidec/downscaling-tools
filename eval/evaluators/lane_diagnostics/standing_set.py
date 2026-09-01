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
import textwrap
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


def _stamp(fig, caption: str) -> None:
    """Lay a caption under a figure that was built by a shared backend plotter."""
    wrapped = "\n".join(textwrap.fill(part, 118) for part in caption.split("\n"))
    n_lines = wrapped.count("\n") + 1
    fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1] + 0.16 * n_lines)
    fig.subplots_adjust(bottom=0.06 + 0.030 * n_lines)
    fig.text(0.015, 0.006, wrapped, ha="left", va="bottom", fontsize=7.8, color="#333333")


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
                _stamp(fig, _arm_caption(arm_label, stats, variable, mode))
                letter = next(letters)
                name = f"12{letter}_tc_{vshort}_{mode}_{_slug(arm_label)}.pdf"
                fig.savefig(out_dir / name, dpi=200)
                figures.append(fig)
                captions[f"12{letter}"] = {
                    "slug": f"tc {vshort} {mode} {arm_label}",
                    "file": name,
                    "caption": _arm_caption(arm_label, stats, variable, mode)}
    return figures, captions


def _curve_key(name: str) -> str:
    """The (date, step, state, member) part of an amplitude file name."""
    return name[len("ampl_"):]


def _load_spectra_dir(spectra_dir: Path) -> dict:
    """Read every amplitude/wavenumber pair under a spectra directory.

    Returns {weather_state: {curve_key: (wavenumbers, amplitudes)}}.
    """
    out: dict = {}
    for state_dir in sorted(Path(spectra_dir).glob("*")):
        if not state_dir.is_dir():
            continue
        curves = {}
        for ampl in sorted(state_dir.glob("ampl_*.npy")):
            wvn = state_dir / ("wvn_" + _curve_key(ampl.name))
            if not wvn.exists():
                continue
            curves[_curve_key(ampl.name)] = (np.load(wvn), np.load(ampl))
        if curves:
            out[state_dir.name] = curves
    return out


def _relative_deviation(pred, truth, wavenumbers, wmin: float) -> float:
    """Relative L2 distance between two amplitude curves above a wavenumber."""
    sel = wavenumbers > wmin
    t = np.asarray(truth)[sel]
    p = np.asarray(pred)[sel]
    denom = np.sqrt(np.sum(t ** 2))
    if denom == 0:
        return float("nan")
    return float(np.sqrt(np.sum((p - t) ** 2)) / denom)


def _spectra_measure(summary_path: Path, wmin: float) -> dict:
    """Per-field deviation from the target's spectrum, for the model and the driver."""
    summary = json.loads(Path(summary_path).read_text())
    pred_dir = Path(summary["out_dir"])
    truth_dir = Path(summary["reference_spectra_dir"])
    input_dir = Path(str(truth_dir).replace("/truth/", "/input/"))
    if not input_dir.exists():
        LOG.warning("no input spectra beside the truth reference at %s", input_dir)

    pred = _load_spectra_dir(pred_dir)
    truth = _load_spectra_dir(truth_dir)
    inp = _load_spectra_dir(input_dir) if input_dir.exists() else {}

    rows, curves = [], {}
    for state in sorted(truth):
        shared = sorted(set(truth[state]) & set(pred.get(state, {})))
        if not shared:
            continue
        model_dev, input_dev = [], []
        for key in shared:
            wvn, t_amp = truth[state][key]
            model_dev.append(_relative_deviation(pred[state][key][1], t_amp, wvn, wmin))
            if key in inp.get(state, {}):
                input_dev.append(_relative_deviation(inp[state][key][1], t_amp, wvn, wmin))
        rows.append({
            "field": state.replace("_sfc", ""),
            "n_curves": len(shared),
            "model_dev": float(np.mean(model_dev)),
            "input_dev": float(np.mean(input_dev)) if input_dev else float("nan"),
        })
        wvn = truth[state][shared[0]][0]
        curves[state.replace("_sfc", "")] = {
            "wavenumbers": wvn,
            "truth": np.mean([truth[state][k][1] for k in shared], axis=0),
            "model": np.mean([pred[state][k][1] for k in shared], axis=0),
            "driver": (np.mean([inp[state][k][1] for k in shared if k in inp.get(state, {})], axis=0)
                       if input_dev else None),
        }
    return {"rows": rows, "curves": curves, "wmin": wmin,
            "truncation": summary.get("truncation")}


def build_spectra_panel(spectra_cfg: dict, out_dir: Path):
    """One spectra figure, with the cost quoted in absolute percentage points."""
    import matplotlib.pyplot as plt

    source = None
    for candidate in spectra_cfg.get("candidates", []):
        if Path(candidate["path"]).exists():
            source = candidate
            break
    if source is None:
        LOG.error("figure 12i skipped: no spectra source exists among %s",
                  [c["path"] for c in spectra_cfg.get("candidates", [])])
        return [], {}

    wmin = float(spectra_cfg.get("wavenumber_min", 100.0))
    measured = _spectra_measure(Path(source["path"]), wmin)
    rows = measured["rows"]
    if not rows:
        LOG.error("figure 12i skipped: %s carried no matched curves", source["path"])
        return [], {}

    show = [f for f in ("10u", "msl") if f in measured["curves"]][:2]
    fig, axs = plt.subplots(1, 2 + len(show), figsize=(5.0 * (2 + len(show)), 5.2))
    for ax, field in zip(axs, show):
        c = measured["curves"][field]
        ax.loglog(c["wavenumbers"], c["truth"], color="#111111", lw=2.0,
                  label="IEKM 4.4 km target")
        if c["driver"] is not None:
            ax.loglog(c["wavenumbers"], c["driver"], color="#1f77b4", lw=1.6,
                      label="interpolated 9 km driver")
        ax.loglog(c["wavenumbers"], c["model"], color="#d62728", lw=1.6,
                  label="downscaler")
        ax.axvline(wmin, color="#888888", ls=":", lw=1.2)
        ax.set_xlabel("spherical wavenumber")
        ax.set_ylabel("power")
        ax.set_title(f"{field}: how power is spread over scales")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, which="both")

    fields = [r["field"] for r in rows]
    xs = np.arange(len(rows))
    model = 100 * np.asarray([r["model_dev"] for r in rows])
    driver = 100 * np.asarray([r["input_dev"] for r in rows])
    width = 0.38
    ax = axs[len(show)]
    ax.bar(xs - width / 2, driver, width, color="#1f77b4", alpha=0.85,
           label="interpolated 9 km driver")
    ax.bar(xs + width / 2, model, width, color="#d62728", alpha=0.85, label="downscaler")
    for x, v in zip(xs, model):
        if np.isfinite(v):
            ax.text(x + width / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(fields)
    ax.set_ylabel("deviation from the target's spectrum (percentage points)")
    ax.set_title(f"Deviation above wavenumber {wmin:.0f}")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)

    ax = axs[len(show) + 1]
    gain = driver - model
    ax.bar(xs, gain, color="#2ca02c", alpha=0.85)
    for x, v in zip(xs, gain):
        if np.isfinite(v):
            ax.text(x, v, f"{v:+.1f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0.0, color="#000000", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(fields)
    ax.set_ylabel("percentage points of deviation removed")
    ax.set_title("What the downscaler buys, absolute")
    ax.grid(axis="y", alpha=0.25)

    detail = "; ".join(f"{r['field']} driver {100*r['input_dev']:.1f} pp, "
                       f"model {100*r['model_dev']:.1f} pp" for r in rows)
    caption = (
        "A spectrum says how a field's variance is spread across spatial scales, from "
        "planetary waves at low wavenumber to the smallest features the grid can carry at "
        "high wavenumber. The left panels are the mean spectra themselves, both axes "
        "logarithmic. The remaining panels give the deviation of each field's spectrum "
        "from the target's in ABSOLUTE PERCENTAGE POINTS, and how many of those points the "
        "downscaler removes relative to its driver. Percentage points, not a percentage of "
        "the error: these deviations are small, and quoting a change as a fraction of a "
        f"small error has previously made a negligible cost read as a real one. {detail}.\n"
        "10 m wind is shown as its own field, not folded into a surface average.\n"
        f"Support: {source['support']}, scored above wavenumber {wmin:.0f} only, "
        f"spectral truncation {measured['truncation']}. "
        f"Sample: {rows[0]['n_curves']} curves per field. Source: {source['label']}."
    )
    name = "12i_spectra_panel.pdf"
    fig.suptitle("12i. Spectra: how much fine-scale structure each field has",
                 fontsize=13, fontweight="bold")
    n_lines = caption.count("\n") + 6
    fig.subplots_adjust(bottom=0.10 + 0.022 * n_lines, top=0.88, wspace=0.30)
    fig.text(0.012, 0.008, "\n".join(textwrap.fill(part, 190) for part in caption.split("\n")),
             ha="left", va="bottom", fontsize=8.2, color="#333333")
    fig.savefig(out_dir / name, dpi=200)
    return [fig], {"12i": {"slug": "spectra panel", "file": name, "caption": caption}}


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
