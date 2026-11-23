"""The suite pages drawn from data (the others are eval.cli figures pasted together, see ``runner``).

House roles everywhere: target black, input blue dashed, first model red, further models in
``eval.plotting.roles.sequence_style`` colours.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

TARGET = "Target"


def model_style(i: int, **kw) -> dict:
    from eval.plotting import role_style
    from eval.plotting.roles import sequence_style

    return role_style("model", **kw) if i == 0 else sequence_style(i - 1, **kw)


# ---------------------------------------------------------------- 01-02 spectra

def spectra_page(reads: dict[str, Path], out_stem: Path, *, step: int, box: str | None,
                 res_note: str = "") -> list[Path]:
    """One spectra grid: target and input from the first read, one model curve per read (box_spectra bins.csv)."""
    from eval.evaluators.box_spectra.plotter import grid_panels, input_limit_marks, target_limit_marks
    from eval.evaluators.box_spectra.runner import read_bins
    from eval.plotting.spectra_grid import Curve, plot_spectra_grid

    panels: dict[str, list] = {}
    draws = []
    for i, (label, res) in enumerate(reads.items()):
        rows = [r for r in read_bins(Path(res) / "bins.csv") if r["scope"] == "all"]
        if not rows:
            continue
        draws.append(f"{label} {rows[0]['n_draws']} draws")
        for var, curves in grid_panels(rows).items():
            truth, inp, model = curves
            if var not in panels:
                panels[var] = [Curve(TARGET, truth.wavenumber, truth.amplitude, role="truth"),
                               Curve("Input", inp.wavenumber, inp.amplitude, role="input")]
            panels[var].append(Curve(label, model.wavenumber, model.amplitude, role="model",
                                     style=model_style(i, linewidth=1.5)))
    if not panels:
        return []
    title = f"Spectra +{step} h{f', box {box}' if box else ''}{res_note}: target, input and model ({'; '.join(draws)})"
    foot = ("Box FFT amplitude per wavelength bin, placed at total wavenumber 40030 km / wavelength; ratio = amplitude "
            "/ target amplitude. Target and input curves are those of the first run's files.")
    first = next(iter(reads.values()))
    marks = input_limit_marks(first) + target_limit_marks(first)   # input T1279 31 km, target T2559 16 km on 4 km
    return plot_spectra_grid(panels, out_stem, title=title, footnote=foot, marks=marks)


# ---------------------------------------------------------------- 06 scores at stations

SCORE_ROWS = (("fcrps", "Fair CRPS ({unit})"), ("spread", "Spread ({unit})"),
              ("ssr_corrected", "Spread / skill, corrected"))


def scores_page(obs: dict[str, Path], out_stem: Path, *, variables, region: str = "all") -> list[Path]:
    """Fair CRPS, spread and spread/skill by lead at stations, one column per variable, every run on one page.

    The input and target rows come from the first run's obs_points read (they are the same files' input and
    target); each run adds its model row.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from eval.evaluators.obs_points.core.summary import var_unit
    from eval.evaluators.obs_points.plotter import VAR_TITLE
    from eval.plotting import eval_style, role_style, save_figure

    import json

    tabs = {}
    first_dir = Path(next(iter(obs.values())))
    meta = json.loads((first_dir / "run_meta.json").read_text()) if (first_dir / "run_meta.json").exists() else {}
    for label, d in obs.items():
        s = pd.read_csv(Path(d) / "summary.csv")
        tabs[label] = s[(s.region == region) & (s.radius_km == 0)]
    first = next(iter(tabs))
    cols = []
    for var, variant in variables:
        s = tabs[first]
        if not len(s[(s["var"] == var) & (s.variant == variant)]):
            variant = "raw"
        if len(s[(s["var"] == var) & (s.variant == variant)]):
            cols.append((var, variant))
    if not cols:
        return []
    with eval_style():
        fig, axes = plt.subplots(len(SCORE_ROWS), len(cols), figsize=(4.6 * len(cols), 3.0 * len(SCORE_ROWS)),
                                 squeeze=False, sharex=True)
        for j, (var, variant) in enumerate(cols):
            unit = var_unit(var, meta)
            info = ""
            for i, (key, ylabel) in enumerate(SCORE_ROWS):
                ax = axes[i, j]
                series = [(lab, "model", model_style(k)) for k, lab in enumerate(tabs)]
                series += [(first, "input", role_style("input")), (first, "truth", role_style("truth"))]
                for lab, row, st in series:
                    s = tabs[lab]
                    q = s[(s["var"] == var) & (s.variant == variant) & (s.row == row)].sort_values("lead")
                    if not len(q) or key not in q or q[key].isna().all():
                        continue
                    if key != "fcrps" and row != "model" and q.get("members", pd.Series([2])).max() <= 1:
                        continue  # a single field has no spread
                    name = {"model": lab, "input": "Input", "truth": TARGET}[row]
                    x, m = q.lead.to_numpy(float), q[key].to_numpy(float)
                    ax.plot(x, m, marker="o" if row == "model" else "s", label=name, **st)
                    se = f"{key}_se"
                    if se in q and q[se].notna().any():
                        ax.fill_between(x, m - q[se], m + q[se], color=st["color"], alpha=0.15, lw=0)
                    if row == "model" and lab == first and key == "fcrps":
                        info = f"{int(q.dates.max())} dates, ~{q.mean_n.mean():.0f} stations per date"
                if key == "ssr_corrected":
                    ax.axhline(1.0, color="#0b0b0b", lw=1, ls="--")
                if i == 0:
                    ax.set_title(f"{VAR_TITLE.get(var, var)} ({variant})\n{info}", fontsize=10)
                ax.set_ylabel(ylabel.format(unit=unit), fontsize=9)
                if i == len(SCORE_ROWS) - 1:
                    leads = sorted(set(tabs[first].lead))
                    ax.set_xticks(leads)
                    ax.set_xticklabels([f"+{int(l)}" for l in leads])
                    ax.set_xlabel("Lead time (h)")
        h, lab = [], []
        for ax in axes.ravel():
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                if ll not in lab:
                    h.append(hh)
                    lab.append(ll)
        fig.legend(h, lab, loc="upper center", ncol=min(6, len(lab)), bbox_to_anchor=(0.5, 0.97))
        fig.suptitle(f"Scores against station observations, region {region}", fontweight="bold")
        fig.text(0.01, 0.0, "Mean over dates of the per-date mean over stations; bands are one standard error over "
                 "dates. Model rows: one draw per input member. Input and target rows from the first run's read.",
                 fontsize=8.5, color="#52514e", ha="left", va="top")
        fig.tight_layout(rect=(0, 0.02, 1, 0.92))
        return save_figure(fig, out_stem, close=True)


# ---------------------------------------------------------------- 08 tp distribution

def tp_distribution_page(runs: dict[str, Path], out_stem: Path, *, steps, box_edge_deg=None) -> tuple[list[Path], dict]:
    """tp PDF pooled over every member and the given leads: model(s) against input and target."""
    import matplotlib.pyplot as plt

    from eval.evaluators.precip_dist.core.tp_histogram_comparison import (
        StreamingDist, _plot_density, accumulate_tp_by_step)
    from eval.plotting import eval_style, role_style, save_figure
    from eval.plotting.labels import pdf_label

    def pooled(step_data, key):
        tot = StreamingDist()
        for s in steps:
            if s in step_data:
                tot = tot.merged(step_data[s][key])
        return tot

    dists, info = {}, {}
    for i, (label, d) in enumerate(runs.items()):
        sd = accumulate_tp_by_step(Path(d), ensemble_member_indices="all", info=(meta := {}),
                                   box_edge_deg=box_edge_deg)
        info[label] = meta
        dists[label] = pooled(sd, "pred")
        if i == 0:
            dists["__input"], dists["__truth"] = pooled(sd, "input"), pooled(sd, "truth")
    ref = dists["__truth"] if not dists["__truth"].empty else next(iter(dists.values()))
    q995 = ref.quantile(99.5)
    xmax = max([5.0, ref.quantile(99.95)] + [v.quantile(99.95) for k, v in dists.items()
                                             if not k.startswith("__") and not v.empty])
    with eval_style():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        for ax, title, xlim in ((axes[0], "Overall distribution", (0.0, max(2.0, q995))),
                                (axes[1], "Wet tail", (max(0.25, q995 * 0.15), xmax))):
            for key, name, st in (("__truth", TARGET, role_style("truth", linewidth=1.8)),
                                  ("__input", "Input", role_style("input", linewidth=1.8))):
                if not dists[key].empty:
                    st = dict(st)
                    _plot_density(ax, dists[key], f"{name} (n = {dists[key].n:,})", st.pop("color"), xlim=xlim,
                                  log=True, **st)
            for i, label in enumerate(runs):
                st = model_style(i, linewidth=1.8)
                _plot_density(ax, dists[label], f"{label} (n = {dists[label].n:,})", st.pop("color"), xlim=xlim,
                              log=True, **st)
            ax.set_title(title)
            ax.set_xlabel("Total precipitation per accumulation window (mm)")
            ax.set_ylabel(f"{pdf_label('mm')}, logarithmic axis")
        axes[1].legend(loc="upper right")
        src = info[next(iter(runs))]
        fig.suptitle(f"tp distribution, every member, leads {', '.join(f'+{s}' for s in steps)} h "
                     f"(input from {src.get('input_source')}, target from {src.get('truth_source')})", fontsize=11)
        return save_figure(fig, out_stem, close=True), info


# ---------------------------------------------------------------- 09 TC distributions

def common_kernel_width(samples) -> float:
    """Silverman's width (1.06 x standard deviation x n^-1/5) of the smallest sample: one width for a whole panel."""
    usable = [np.asarray(v, float) for v in samples if np.size(v) >= 2 and np.std(v) > 0]
    if not usable:
        return 1.0
    v = min(usable, key=np.size)
    return float(1.06 * np.std(v) * v.size ** -0.2)


def kernel_density(values, grid, width: float) -> np.ndarray:
    """Gaussian kernel density of ``values`` on ``grid`` (integrates to one)."""
    v = np.asarray(values, float)[:, None]
    z = (np.asarray(grid, float)[None, :] - v) / width
    return np.exp(-0.5 * z * z).sum(axis=0) / (v.shape[0] * width * np.sqrt(2 * np.pi))


def tc_distribution_page(tc_reads: dict[str, Path], cache_dir: Path | None, out_stem: Path,
                         *, wind_col: str = "maxwind300_ms") -> tuple[list[Path], dict]:
    """Central pressure and maximum wind of every member at every storm case (storm_ok cases of the first run),
    model(s), input and target members, and IBTrACS at the same cases."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from eval.plotting import eval_style, role_style, save_figure

    cases = {lab: pd.read_csv(Path(d) / "cases.csv", dtype={"date": str}) for lab, d in tc_reads.items()}
    first = next(iter(cases))
    c0 = cases[first]
    keys = c0[c0.storm_ok == 1][["event", "date", "lead"]].drop_duplicates()
    kset = set(map(tuple, keys.astype({"lead": int}).itertuples(index=False)))

    def members(df, field):
        d = df[(df.field == field) & (df.found == 1)]
        d = d[[k in kset for k in zip(d.event, d.date, d.lead.astype(int))]]
        return d

    series = [(lab, members(df, "model"), model_style(i)) for i, (lab, df) in enumerate(cases.items())]
    series += [("Input", members(c0, "input"), role_style("input")), (TARGET, members(c0, "truth"), role_style("truth"))]
    bt_p, bt_w = [], []
    if cache_dir:
        from datetime import datetime, timedelta

        from eval.evaluators.tc_obs.core.reference import best_track_at, load_cache

        bt, _, _ = load_cache(cache_dir)
        for e, d, l in sorted(kset):
            b = best_track_at(bt, e, datetime.strptime(d, "%Y%m%d") + timedelta(hours=int(l)))
            if b is not None:
                bt_p.append(b.get("pmin_hpa", np.nan))
                bt_w.append(b.get("wind_ms", np.nan))
    info = {"cases": len(kset), "dates": len({k[1] for k in kset}), "events": sorted({k[0] for k in kset}),
            "ibtracs_fixes": len(bt_p)}
    with eval_style():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        for ax, col, bt_vals, xlabel in ((axes[0], "pmin_hpa", bt_p, "Central pressure (hPa)"),
                                         (axes[1], wind_col, bt_w, "Maximum 10 m wind (m/s)")):
            vals = [s[1][col].to_numpy(float) for s in series if col in s[1]]
            allv = np.concatenate([v[np.isfinite(v)] for v in vals] + [np.asarray(bt_vals, float)])
            allv = allv[np.isfinite(allv)]
            if not allv.size:
                continue
            # One kernel width for every curve of the panel, set by the smallest sample (IBTrACS when present),
            # so that no curve is smoothed more than another (owner, 23 Nov 2026: smooth lines, not steps).
            b = np.asarray(bt_vals, float)
            b = b[np.isfinite(b)]
            samples = []
            for name, df, st in series:
                if col in df and df[col].notna().any():
                    v = df[col].to_numpy(float)
                    samples.append((name, v[np.isfinite(v)], st))
            width = common_kernel_width([v for _, v, _ in samples] + ([b] if b.size else []))
            grid = np.linspace(allv.min() - 3 * width, allv.max() + 3 * width, 400)
            for name, v, st in samples:
                ax.plot(grid, kernel_density(v, grid, width), label=f"{name} ({v.size} members)",
                        color=st["color"], linestyle=st.get("linestyle", "-"), linewidth=st.get("linewidth", 2))
            if b.size:
                dens = kernel_density(b, grid, width)
                ax.fill_between(grid, dens, color="#7f3b08", alpha=0.22, linewidth=0)
                ax.plot(grid, dens, color="#7f3b08", linewidth=1.6, label=f"IBTrACS ({b.size} fixes)")
            ax.set_ylim(bottom=0)
            ax.text(0.02, 0.97, f"Gaussian kernel, width {width:.2g} for every curve", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.5, color="0.4")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
        axes[1].legend(loc="upper right", fontsize=8.5)
        fig.suptitle(f"Tropical cyclones: {info['cases']} storm cases ({info['dates']} dates; "
                     f"{', '.join(info['events'])}), every member", fontsize=11)
        return save_figure(fig, out_stem, close=True), info
