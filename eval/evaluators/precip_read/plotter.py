"""precip_read plot(): every figure from the files run() wrote. Colours by role: truth black,
model red, interpolated input blue dashed. Every figure carries its n (dates x members)."""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.evaluators.precip_read.core.metrics import pooled

LOG = logging.getLogger(__name__)

REGION_TITLE = {
    "all": "Whole domain", "10-25N": "Tropics 10-25 N", "25-40N": "Subtropics 25-40 N",
    "40-70N": "Mid-latitudes 40-70 N", "europe": "Europe (35-70 N, 12 W-40 E)",
}
PDF_REGIONS = ("all", "10-25N", "25-40N", "europe")
SCORE_REGIONS = ("all", "10-25N", "europe")
MAP_LEVELS = [0.5, 1, 2, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300]


def _meta(results_dir):
    return json.loads((Path(results_dir) / "run_meta.json").read_text())


def _labels(meta, eval_config):
    run_label = str(eval_config.get("run_label") or meta.get("run_label") or "model")
    acc = int(eval_config.get("accum_hours", meta.get("accum_hours", 6)))
    return run_label, acc


def _footer(fig, text, y=0.0):
    fig.text(0.01, y, text, ha="left", va="top", fontsize=9, color="#52514e")


def _bootstrap_idx(n_dates, n_boot, seed=0):
    return np.random.default_rng(seed).integers(0, n_dates, (n_boot, n_dates))


# --------------------------------------------------------------------------- PDFs
def plot_pdfs(results_dir, step, *, run_label, acc, n_boot=2000, regions=PDF_REGIONS):
    import matplotlib.pyplot as plt
    from eval.plotting import eval_style, role_style, save_figure

    z = np.load(Path(results_dir) / f"hist_step{step:03d}.npz")
    names, dates = list(z["regions"]), z["dates"]
    regions = [r for r in regions if r in names]
    T, I, C = z["truth"].astype(float), z["input"].astype(float), z["model"].astype(float)
    nmem = C.shape[1]

    def merge(a):  # 0.05-dex bins from 0.1 mm into 0.1-dex bins
        a = a[..., 1:]
        return a[..., 0::2] + a[..., 1::2]

    T, I, C = merge(T), merge(I), merge(C).sum(1)
    e = z["edges"][1::2]
    x, w = np.sqrt(e[:-1] * e[1:]), np.diff(e)
    boot = _bootstrap_idx(len(dates), n_boot)
    heavy_dates = int((T[:, names.index(regions[0]), x >= 100].sum(1) > 0).sum())
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    with eval_style():
        nrow = (len(regions) + 1) // 2
        fig = plt.figure(figsize=(14, 4.8 * nrow))
        hr = []
        for i in range(nrow):
            hr += [3, 1.25] + ([0.55] if i < nrow - 1 else [])
        gs = fig.add_gridspec(len(hr), 2, height_ratios=hr, hspace=0.12, wspace=0.2)
        for p, reg in enumerate(regions):
            r = names.index(reg)
            row0 = 3 * (p // 2)
            a = fig.add_subplot(gs[row0, p % 2])
            b = fig.add_subplot(gs[row0 + 1, p % 2], sharex=a)
            npts, nd = int(z["npts"][r]), len(dates)
            t, i, c = T[:, r].sum(0), I[:, r].sum(0), C[:, r].sum(0)
            dens = lambda cnt, nf: cnt / (nf * npts * w)
            a.plot(x[t > 0], dens(t, nd)[t > 0], label="Truth", **role_style("truth"))
            a.plot(x[i > 0], dens(i, nd)[i > 0], label="Input (interpolated)", **role_style("input"))
            a.plot(x[c > 0], dens(c, nd * nmem)[c > 0], label=run_label, **role_style("model"))
            a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1, 800)
            a.set_title(REGION_TITLE.get(reg, reg), loc="left")
            if p % 2 == 0:
                a.set_ylabel("Probability density (per mm)")
            plt.setp(a.get_xticklabels(), visible=False)
            if p == 0:
                a.legend(loc="lower left")
            tail = int(T[:, r, x >= 100].sum())
            a.text(0.98, 0.95, f"{nd} dates · {npts:,} points per field\ntruth points ≥ 100 mm: {tail}",
                   transform=a.transAxes, ha="right", va="top", fontsize=9.5, color="#52514e")
            tb, cb, ib = T[boot, r].sum(1), C[boot, r].sum(1) / nmem, I[boot, r].sum(1)
            ok = t >= 10
            with np.errstate(divide="ignore", invalid="ignore"):
                rc = np.where(tb > 0, cb / tb, np.nan)
                lo, hi = (np.nanpercentile(rc[:, ok], [5, 95], axis=0) if ok.any() else (np.array([]), np.array([])))
                mc, mi = c / nmem / t, i / t
            b.axhline(1, color="#0b0b0b", lw=1, ls="--")
            if ok.any():
                b.fill_between(x[ok], lo, hi, color=role_style("model")["color"], alpha=0.2, lw=0)
                b.plot(x[ok], mc[ok], **role_style("model"))
                b.plot(x[ok], mi[ok], **role_style("input"))
                xmax = x[ok].max()
                for ax in (a, b):
                    ax.axvspan(xmax * 10 ** 0.05, 800, color="#e4e3df", alpha=0.6, lw=0, zorder=0)
                b.text(min(xmax * 1.15, 400), 2.6, "truth < 10 points\nper bin", fontsize=8.5, color="#52514e", va="center")
            b.set_yscale("log"); b.set_ylim(0.1, 4)
            b.set_yticks([0.1, 0.25, 0.5, 1, 2, 4]); b.set_yticklabels(["0.1", "0.25", "0.5", "1", "2", "4"])
            b.minorticks_off()
            b.set_xticks(ticks); b.set_xticklabels([str(v) for v in ticks])
            if p % 2 == 0:
                b.set_ylabel("Ratio to truth")
            if p >= len(regions) - 2:
                b.set_xlabel(f"tp at one grid point (mm per {acc} h)")
        fig.subplots_adjust(top=0.93, bottom=0.11)
        fig.suptitle(f"tp distribution at +{step} h: truth, input and {run_label}", fontweight="bold", y=0.975)
        _footer(fig, f"tp in mm per {acc} h at one O1280 point, window ending at the lead time, clipped at 0. n = {len(dates)} dates; {run_label} {nmem} members per date "
                f"({len(dates) * nmem} fields), truth and input 1 field per date. Bins 0.1 dex.\n"
                f"Ratio strip: {run_label} band = 5-95 % bootstrap over dates, drawn where the truth has ≥ 10 points in the bin; grey beyond = anecdote. "
                f"Heavy tail carried by few dates: truth ≥ 100 mm on {heavy_dates} of {len(dates)} dates.", y=0.045)
        return save_figure(fig, Path(results_dir) / f"tp_pdf_step{step:03d}", close=True)


# --------------------------------------------------------------------------- scores by lead
def _rows(results_dir):
    with open(Path(results_dir) / "scores_rows.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in ("date", "region"):
                r[k] = float(v)
    return rows


def lead_table(rows, region, n_boot=2000, thresholds=(100,)):
    """Per lead: pooled scores and heavy-area ratios with 5-95 % bootstrap over dates."""
    by = defaultdict(list)
    for r in rows:
        if r["region"] == region:
            by[int(r["lead"])].append(r)
    out = {}
    for lead, rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["date"])
        base = pooled(rs)
        idx = _bootstrap_idx(len(rs), n_boot, seed=lead)
        draws = [pooled([rs[i] for i in ii]) for ii in idx]
        band = {k: np.nanpercentile([d[k] for d in draws], [5, 95]) for k in base}
        heavy = {}
        for t in thresholds:
            nt = np.array([r[f"n_truth_ge{t:g}"] for r in rs]); nm = np.array([r[f"n_model_ge{t:g}"] for r in rs])
            ni = np.array([r[f"n_input_ge{t:g}"] for r in rs])
            with np.errstate(divide="ignore", invalid="ignore"):
                bt = nt[idx].sum(1)
                heavy[t] = {"truth_points": int(nt.sum()), "truth_dates": int((nt > 0).sum()),
                            "model": nm.sum() / nt.sum() if nt.sum() else np.nan,
                            "input": ni.sum() / nt.sum() if nt.sum() else np.nan,
                            "model_band": np.nanpercentile(np.where(bt > 0, nm[idx].sum(1) / bt, np.nan), [5, 95]),
                            "input_band": np.nanpercentile(np.where(bt > 0, ni[idx].sum(1) / bt, np.nan), [5, 95])}
        out[lead] = {"n_dates": len(rs), "members": int(rs[0]["members"]), "pooled": base, "band": band, "heavy": heavy}
    return out


def plot_scores_by_lead(results_dir, *, run_label, acc, n_boot=2000, regions=SCORE_REGIONS):
    import matplotlib.pyplot as plt
    from eval.plotting import eval_style, role_style, save_figure

    rows = _rows(results_dir)
    present = {r["region"] for r in rows}
    regions = [r for r in regions if r in present]
    metrics = [("bias", f"Bias (mm / {acc} h)", 0.0), ("nrmse", "RMSE / truth SD", None),
               ("corr", "Correlation", None), ("crps", f"CRPS (mm / {acc} h)", None)]
    tables = {reg: lead_table(rows, reg, n_boot) for reg in regions}
    with eval_style():
        fig, axes = plt.subplots(len(metrics), len(regions), figsize=(4.6 * len(regions), 2.9 * len(metrics)),
                                 sharex=True, squeeze=False)
        for j, reg in enumerate(regions):
            tab = tables[reg]
            leads = np.array(sorted(tab))
            for i, (key, ylabel, ref) in enumerate(metrics):
                ax = axes[i, j]
                for who, role, lab in (("model", "model", f"{run_label}, ensemble mean" if key != "crps" else f"{run_label}, fair CRPS"),
                                       ("input", "input", "Input (interpolated)" if key != "crps" else "Input (MAE = CRPS of one member)")):
                    k = f"{key}_{who}"
                    v = np.array([tab[l]["pooled"][k] for l in leads])
                    lo = np.array([tab[l]["band"][k][0] for l in leads]); hi = np.array([tab[l]["band"][k][1] for l in leads])
                    st = role_style(role, marker="o", markersize=4)
                    ax.fill_between(leads, lo, hi, color=st["color"], alpha=0.15, lw=0)
                    ax.plot(leads, v, label=lab, **st)
                if ref is not None:
                    ax.axhline(ref, color="#0b0b0b", lw=1, ls="--")
                if i == 0:
                    nd = max(tab[l]["n_dates"] for l in leads)
                    ax.set_title(f"{REGION_TITLE.get(reg, reg)}\n{nd} dates × {tab[leads[0]]['members']} members", fontsize=11)
                if j == 0:
                    ax.set_ylabel(ylabel)
                if i == len(metrics) - 1:
                    ax.set_xticks(leads); ax.set_xticklabels([f"+{int(l)}" for l in leads]); ax.set_xlabel("Lead time (h)")
            if j == 0:
                h1, l1 = axes[0, 0].get_legend_handles_labels()
                h2, l2 = axes[-1, 0].get_legend_handles_labels()
                fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.965), fontsize=9.5)
        fig.suptitle(f"tp scores against the truth by lead time: {run_label} and its input", fontweight="bold")
        _footer(fig, f"Pooled over points and dates (points unweighted). Bias = mean forecast − mean truth; RMSE of the ensemble mean "
                f"over the truth's standard deviation; Pearson correlation of the ensemble mean; fair CRPS of the {run_label} members "
                f"(input: absolute error).\nBands: 5-95 % bootstrap over dates. Lower is better for |bias|, RMSE and CRPS; higher for "
                f"correlation. Ensemble-mean RMSE and correlation favour smooth forecasts: read CRPS for the ensemble.", y=0.0)
        fig.tight_layout(rect=(0, 0.03, 1, 0.94))
        return save_figure(fig, Path(results_dir) / "tp_scores_by_lead", close=True)


def plot_heavy_area_by_lead(results_dir, *, run_label, acc, threshold=100, region="all", n_boot=2000):
    import matplotlib.pyplot as plt
    from eval.plotting import eval_style, role_style, save_figure

    tab = lead_table(_rows(results_dir), region, n_boot, thresholds=(threshold,))
    leads = np.array(sorted(tab))
    with eval_style():
        fig, ax = plt.subplots(figsize=(9.5, 5))
        ax.axhline(1, color="#0b0b0b", lw=1, ls="--")
        for who, role, lab, dx in (("model", "model", run_label, -1.0), ("input", "input", "Input (interpolated)", 1.0)):
            v = np.array([tab[l]["heavy"][threshold][who] for l in leads])
            band = np.array([tab[l]["heavy"][threshold][f"{who}_band"] for l in leads])
            st = role_style(role, marker="o", markersize=8)
            ax.errorbar(leads + dx, v, yerr=[v - band[:, 0], band[:, 1] - v], capsize=4, label=lab, **st)
        hi = np.array([tab[l]["heavy"][threshold]["model_band"][1] for l in leads], float)
        top = min(max(2.0, float(np.nanmax(hi)) * 1.05 if np.isfinite(hi).any() else 2.0), 4.0)
        ax.set_ylim(0, top * 1.18)
        for l in leads:
            h = tab[l]["heavy"][threshold]
            ax.text(l, top * 1.16, f"{h['truth_dates']} dates\n{h['truth_points']} pts", ha="center", va="top", fontsize=9, color="#52514e")
        ax.set_xticks(leads); ax.set_xticklabels([f"+{int(l)} h" for l in leads]); ax.set_xlabel("Lead time")
        ax.set_ylabel(f"Area ≥ {threshold:g} mm / {acc} h, ratio to truth")
        ax.set_title(f"Heavy-rain area by lead time ({REGION_TITLE.get(region, region)})", fontweight="bold")
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.86))
        nd = max(tab[l]["n_dates"] for l in leads)
        _footer(fig, f"n = {nd} dates × {tab[leads[0]]['members']} members per lead. Points = pooled count ratio over dates "
                f"({run_label}: mean over members); bars = 5-95 % bootstrap over dates.\nTop labels: dates with any truth point "
                f"≥ {threshold:g} mm and the truth's point count; few dates carry most heavy points, so the effective n is the number of events.",
                y=-0.02)
        return save_figure(fig, Path(results_dir) / "tp_heavy_area_by_lead", close=True)


# --------------------------------------------------------------------------- station scores
STATION_FORECASTS = (("model", "model"), ("input", "input"), ("truth", "truth"))


def plot_station_scores(results_dir, *, run_label, acc, regions=("all", "europe"), ref="obs"):
    """Scores against station observations by lead: mean over dates of the per-date scores, ±1 SE."""
    import matplotlib.pyplot as plt
    from eval.plotting import eval_style, role_style, save_figure

    path = Path(results_dir) / "station_scores_per_date.csv"
    if not path.exists():
        return []
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["ref"] == ref]
    for r in rows:
        r["forecast"] = "model" if r["forecast"] not in ("input", "truth") else r["forecast"]
    regions = [g for g in regions if any(r["region"] == g for r in rows)]
    leads = sorted({int(r["lead"]) for r in rows})
    metrics = [("bias", f"Bias (mm / {acc} h)", 0.0), ("nrmse", "RMSE / obs SD", None),
               ("corr", "Correlation", None), ("crps", f"CRPS (mm / {acc} h)", None)]
    label = {"model": f"{run_label} (ensemble mean; fair CRPS of members)", "input": "Input (interpolated)",
             "truth": "Truth (o1280 forecast)"}
    offs = {"model": -1.2, "input": 0.0, "truth": 1.2}
    with eval_style():
        fig, axes = plt.subplots(len(metrics), len(regions), figsize=(5.2 * len(regions), 2.7 * len(metrics)),
                                 sharex=True, squeeze=False)
        for j, reg in enumerate(regions):
            sub = [r for r in rows if r["region"] == reg]
            for i, (key, ylabel, refl) in enumerate(metrics):
                ax = axes[i, j]
                for fc, role in STATION_FORECASTS:
                    m, se = [], []
                    for lead in leads:
                        v = np.array([float(r[key]) for r in sub if r["forecast"] == fc and int(r["lead"]) == lead], float)
                        v = v[np.isfinite(v)]
                        m.append(v.mean() if v.size else np.nan)
                        se.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan)
                    st = role_style(role, marker="o", markersize=7)
                    st["linestyle"] = "none"
                    ax.errorbar(np.array(leads) + offs[fc], m, yerr=se, capsize=4, elinewidth=2, label=label[fc], **st)
                if refl is not None:
                    ax.axhline(refl, color="#0b0b0b", lw=1, ls="--")
                if i == 0:
                    nd = len({r["date"] for r in sub})
                    ns = np.mean([float(r["n"]) for r in sub if r["forecast"] == "model"])
                    ax.set_title(f"{REGION_TITLE.get(reg, reg)}\n{nd} dates, ~{ns:,.0f} stations per date", fontsize=11)
                if j == 0:
                    ax.set_ylabel(ylabel)
                if i == len(metrics) - 1:
                    ax.set_xticks(leads); ax.set_xticklabels([f"+{l}" for l in leads]); ax.set_xlabel("Lead time (h)")
                    ax.set_xlim(min(leads) - 12, max(leads) + 12)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955), fontsize=10)
        fig.suptitle(f"tp against station observations ({acc} h accumulations): {run_label}, its input and the truth",
                     fontweight="bold")
        _footer(fig, f"Stations: STVL {acc} h tp (mostly SYNOP; the observations quaver uses), nearest grid point, values outside "
                f"[0, 300] mm dropped. Mean over dates of per-date scores; bars ±1 SE over dates.\nCRPS of a single field = mean "
                f"absolute error, so the ensemble's lower CRPS is partly by construction; compare RMSE and correlation of the "
                f"ensemble mean for like with like. Point scores penalise realistic but misplaced detail.", y=0.0)
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
        return save_figure(fig, Path(results_dir) / "tp_station_scores", close=True)


# --------------------------------------------------------------------------- maps
def _map_cmap():
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    base = plt.get_cmap("YlGnBu")(np.linspace(0.12, 1.0, len(MAP_LEVELS) - 1))
    extra = np.array([[0.42, 0.18, 0.55, 1], [0.62, 0.10, 0.45, 1]])
    cmap = ListedColormap(np.vstack([base[:-1], extra[:1]])).with_extremes(under="white", over=extra[1])
    return cmap, BoundaryNorm(MAP_LEVELS, cmap.N)


def plot_case_maps(results_dir, group, case_names, *, run_label, acc, members=(0, 1, 2), title=None):
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from eval.plotting import eval_style, save_figure

    files = [Path(results_dir) / "cases" / f"{n}.npz" for n in case_names]
    files = [f for f in files if f.exists()]
    if not files:
        LOG.warning("precip_read: no case files for map group %s", group)
        return []
    cmap, norm = _map_cmap()
    proj = ccrs.PlateCarree()
    ncol = 2 + len(members)
    data = [np.load(f) for f in files]
    asp = [(z["box"][1] - z["box"][0]) / (z["box"][3] - z["box"][2]) for z in data]  # PlateCarree: degrees as drawn
    W = 3.2
    top_in, bot_in = 0.95, 1.25
    H = sum(W * a for a in asp) * 1.08 + top_in + bot_in
    with eval_style():
        fig = plt.figure(figsize=(W * ncol + 0.9, H))
        gs = fig.add_gridspec(len(data), ncol, height_ratios=asp, hspace=0.08, wspace=0.04,
                              top=1 - top_in / H, bottom=bot_in / H, left=0.9 / (W * ncol + 0.9), right=0.995)
        im = None
        for r, z in enumerate(data):
            box = z["box"]
            tri = mtri.Triangulation(z["lon"], z["lat"])
            xy = np.c_[z["lon"], z["lat"]][tri.triangles]
            tri.set_mask(np.ptp(xy[..., 0], 1) + np.ptp(xy[..., 1], 1) > 0.5)
            stored = [int(m) for m in z["members"]] if "members" in z else list(range(z["model"].shape[0]))
            mem = [m for m in members if m in stored]
            fields = [("Truth", z["truth"]), ("Input (interpolated)", z["input"])] + \
                     [(f"{run_label} member {m + 1}", z["model"][stored.index(m)]) for m in mem]
            fields = [(n, np.asarray(f, np.float32)) for n, f in fields]
            for c, (name, f) in enumerate(fields):
                ax = fig.add_subplot(gs[r, c], projection=proj)
                ax.set_extent([box[2], box[3], box[0], box[1]], crs=proj)
                im = ax.tripcolor(tri, np.where(f < MAP_LEVELS[0], np.nan, f), cmap=cmap, norm=norm, shading="flat",
                                  transform=proj, rasterized=True)
                ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.6, edgecolor="#3a3a38")
                ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.35, edgecolor="#6b6a66")
                ax.text(0.98, 0.03, f"max {f.max():.0f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                        bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
                if r == 0:
                    ax.set_title(name, fontsize=11.5)
                if c == 0:
                    lab = str(z["label"]) + f"\n{str(z['date'])} +{int(z['step'])} h"
                    ax.text(-0.05, 0.5, lab, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=11,
                            fontweight="bold")
        cax = fig.add_axes([0.25, 0.62 / H, 0.5, 0.16 / H])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max", ticks=MAP_LEVELS)
        cb.ax.set_xticklabels([f"{v:g}" for v in MAP_LEVELS])
        cb.set_label(f"tp (mm / {acc} h)")
        fig.suptitle(title or f"tp maps: {group.replace('_', ' ')}", fontweight="bold", y=1 - 0.25 / H)
        mem_txt = ", ".join(str(m + 1) for m in members)
        _footer(fig, f"One date per row (n = 1 field each: illustration, not statistics). Members {mem_txt} are the first members "
                f"in the file, not chosen. Panel corner: field maximum in the box (mm / {acc} h). White: < {MAP_LEVELS[0]} mm.",
                y=0.1 / H)
        return save_figure(fig, Path(results_dir) / f"tp_maps_{group}", close=True)


# --------------------------------------------------------------------------- entry
def plot(results_dir, lane_config, eval_config, *, output_dir=None, **kwargs):
    results_dir = Path(results_dir)
    meta = _meta(results_dir)
    run_label, acc = _labels(meta, {**eval_config, **({"run_label": kwargs["run_label"]} if kwargs.get("run_label") else {})})
    n_boot = int(eval_config.get("bootstrap", 2000))
    for step in meta["steps"]:
        plot_pdfs(results_dir, int(step), run_label=run_label, acc=acc, n_boot=n_boot)
    if len(meta["steps"]) > 1:
        plot_scores_by_lead(results_dir, run_label=run_label, acc=acc, n_boot=n_boot)
        plot_heavy_area_by_lead(results_dir, run_label=run_label, acc=acc, n_boot=n_boot)
    plot_station_scores(results_dir, run_label=run_label, acc=acc)
    members = tuple(int(m) for m in eval_config.get("map_members", (0, 1, 2)))
    for group, names in (eval_config.get("map_groups") or {}).items():
        titles = eval_config.get("map_titles") or {}
        plot_case_maps(results_dir, group, list(names), run_label=run_label, acc=acc, members=members,
                       title=titles.get(group))
