"""precip_scores evaluator — pointwise + distribution skill for 6h-window tp.

Scores model tp (and the interpolation baseline) against 6h-window truth at
the same step, in mm. Truth comes from the predictions' embedded `y` when the
tp channel is populated, otherwise from the lane's `precip.truth_grib_tpl`
GRIB (the o1280->o2560 main-lane bundles historically carried no tp truth).
The baseline comes from `x_interp` when its tp channel is a real series,
otherwise from the driving o1280 member tp via `precip.baseline_lres_grib_tpl`
(tp is output-only on this lane, so the exported x_interp tp is all zero).

Outputs: scores.json (machine-readable, scoreboard-ingestable), scores_rows.csv,
plots/precip_scores.pdf (skill, tail ratios and a summary table;
``render_from_json`` redraws it from scores.json).
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.evaluators.precip_scores.core import metrics as M
from eval.shared.precip.sources import (
    LresInterpBaseline,
    PrecipTruthSource,
    is_degenerate_channel,
)
from eval.discovery.predictions import find_predictions

LOG = logging.getLogger(__name__)

MM = 1000.0  # metres -> millimetres


def _probe_channel(ds, name: str, tp_idx: int) -> np.ndarray:
    """Cheap two-block sample of one channel for populated/degenerate probes."""
    var = ds[name]
    n = var.shape[2]
    blocks = [var[0, 0, :100_000].values[:, tp_idx]]
    if n > 200_000:
        mid = n // 2
        blocks.append(var[0, 0, mid:mid + 100_000].values[:, tp_idx])
    return np.concatenate(blocks)


def _member_ids(ds, n_members: int) -> list[int]:
    raw = str(ds.attrs.get("member_ids", ""))
    if raw:
        try:
            ids = [int(x) for x in raw.split(",")]
            if len(ids) == n_members:
                return ids
        except ValueError:
            pass
    return list(range(1, n_members + 1))


def run(
    predictions_dir,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir=None,
    overwrite: bool = False,
    checkpoint: str | None = None,
    **kwargs,
) -> Path:
    import xarray as xr

    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "precip_scores"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"precip_scores output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    precip_cfg = dict(lane_config.get("precip", {}))
    var = str(eval_config.get("var", "tp"))
    wet_thr = float(eval_config.get("wet_threshold_mm", M.WET_THRESHOLD_MM))
    only_dates = {str(d) for d in eval_config.get("dates", [])} or None
    only_steps = {int(s) for s in eval_config.get("steps", [])} or None
    max_members = eval_config.get("max_members")

    preds = find_predictions(predictions_dir)
    if only_dates:
        preds = [p for p in preds if p.date in only_dates]
    if only_steps:
        preds = [p for p in preds if p.step in only_steps]
    if not preds:
        raise FileNotFoundError(f"No prediction files to score in {predictions_dir}")
    by_date: dict[str, list] = defaultdict(list)
    for p in preds:
        by_date[p.date].append(p)

    # ---- decide truth + baseline sources from the first file ---------------
    first = preds[0]
    with xr.open_dataset(first.path) as ds0:
        ws = [str(s) for s in ds0["weather_state"].values]
        if var not in ws:
            raise ValueError(f"'{var}' not in weather_state {ws} ({first.path})")
        tp_idx = ws.index(var)
        n_members = int(ds0.sizes["ensemble_member"])
        member_ids = _member_ids(ds0, n_members)
        lat_hres = ds0["lat_hres"].values
        lon_hres = ds0["lon_hres"].values
        y_probe = _probe_channel(ds0, "y", tp_idx)
        xi_probe = (_probe_channel(ds0, "x_interp", tp_idx)
                    if "x_interp" in ds0.variables else np.array([np.nan]))
        ckpt_id = str(ds0.attrs.get("checkpoint_id", checkpoint or ""))

    if max_members:
        n_members = min(n_members, int(max_members))
        member_ids = member_ids[:n_members]

    truth_populated = float(np.isnan(y_probe).mean()) < 0.01
    truth_src = None
    if truth_populated:
        truth_mode = "embedded-y"
        LOG.info("precip_scores: truth = embedded y[%s] channel", var)
    else:
        tpl = precip_cfg.get("truth_grib_tpl")
        if not tpl:
            raise RuntimeError(
                f"predictions carry no {var} truth (y channel is NaN) and the "
                "lane config has no precip.truth_grib_tpl — nothing to score "
                "against. Add the truth GRIB template to the lane's precip block.")
        truth_src = PrecipTruthSource(tpl, var=var)
        truth_mode = f"grib:{tpl}"
        LOG.warning(
            "precip_scores: predictions carry no %s truth — injecting truth "
            "from %s (fix the bundle prepare stage so future runs embed it)",
            var, tpl)

    baseline_src = None
    if not is_degenerate_channel(xi_probe):
        baseline_mode = "x_interp"
        LOG.info("precip_scores: baseline = embedded x_interp[%s]", var)
    else:
        tpl = precip_cfg.get("baseline_lres_grib_tpl")
        if tpl:
            baseline_src = LresInterpBaseline(
                tpl, precip_cfg.get("interp_index_cache"), var=var)
            baseline_src.ensure_index(lat_hres, lon_hres,
                                      probe_date=sorted(by_date)[0])
            baseline_mode = f"lres-nn:{tpl}"
            LOG.warning(
                "precip_scores: x_interp[%s] is degenerate (output-only channel) "
                "— baseline = o1280 member %s nearest-neighbour interpolated", var, var)
        else:
            baseline_mode = "none"
            LOG.warning(
                "precip_scores: no usable baseline (x_interp degenerate, no "
                "precip.baseline_lres_grib_tpl) — scoring model vs truth only")

    # ---- score --------------------------------------------------------------
    rows: list[dict] = []
    for date in sorted(by_date):
        if truth_src is not None:
            truth_src.preload(date)
            truth_src.verify_grid(lat_hres, lon_hres)
        for p in sorted(by_date[date], key=lambda q: q.step):
            with xr.open_dataset(p.path) as ds:
                ws_f = [str(s) for s in ds["weather_state"].values]
                ti = ws_f.index(var)
                if truth_src is not None:
                    truth_mm = truth_src.load(date, p.step).astype(np.float64) * MM
                else:
                    truth_mm = ds["y"][0, 0].values[:, ti].astype(np.float64) * MM

                ens_sum = np.zeros_like(truth_mm)
                bl_sum = np.zeros_like(truth_mm) if baseline_mode != "none" else None
                mem_rows = []
                for mi in range(n_members):
                    yp_mm = ds["y_pred"][0, mi].values[:, ti].astype(np.float64) * MM
                    ens_sum += yp_mm
                    mem = {
                        "member": member_ids[mi],
                        "model": {**M.pair_scores(yp_mm, truth_mm),
                                  **M.field_stats(yp_mm, wet_threshold_mm=wet_thr)},
                    }
                    if baseline_mode == "x_interp":
                        bl_mm = ds["x_interp"][0, mi].values[:, ti].astype(np.float64) * MM
                    elif baseline_src is not None:
                        bl_mm = baseline_src.load(date, p.step, member_ids[mi]).astype(np.float64) * MM
                    else:
                        bl_mm = None
                    if bl_mm is not None:
                        bl_sum += bl_mm
                        mem["baseline"] = {**M.pair_scores(bl_mm, truth_mm),
                                           **M.field_stats(bl_mm, wet_threshold_mm=wet_thr)}
                    mem_rows.append(mem)

            ens_mm = ens_sum / n_members
            row = {
                "date": date,
                "step": p.step,
                "truth": M.field_stats(truth_mm, wet_threshold_mm=wet_thr),
                "model_ens_mean": M.pair_scores(ens_mm, truth_mm),
                "members": mem_rows,
            }
            if bl_sum is not None:
                row["baseline_ens_mean"] = M.pair_scores(bl_sum / n_members, truth_mm)
            rows.append(row)
            LOG.info("scored %s step %03d: model rmse(member-mean)=%.3f mm, "
                     "baseline=%s",
                     date, p.step,
                     M.nanmean([m["model"].get("rmse_mm") for m in mem_rows]),
                     f"{M.nanmean([m.get('baseline', {}).get('rmse_mm') for m in mem_rows]):.3f} mm"
                     if bl_sum is not None else "n/a")
        if truth_src is not None:
            truth_src.release()
        if baseline_src is not None:
            baseline_src.release()

    # ---- aggregate ----------------------------------------------------------
    per_step = aggregate_rows(rows)
    summary = summarize(per_step)

    payload = {
        "meta": {
            "predictions_dir": str(predictions_dir),
            "var": var,
            "unit": "mm / 6h window",
            "truth_source": truth_mode,
            "baseline_source": baseline_mode,
            "checkpoint_id": ckpt_id,
            "n_members": n_members,
            "member_ids": member_ids,
            "n_slices": len(rows),
            "wet_threshold_mm": wet_thr,
            "negative_handling": "raw values in rmse/bias/corr; negatives "
                                 "clipped to 0 only inside quantile histograms",
        },
        "per_step": per_step,
        "summary": summary,
        "rows": rows,
    }
    run_label = str(eval_config.get("run_label") or kwargs.get("run_label")
                    or predictions_dir.parent.name)
    (output_dir / "scores.json").write_text(json.dumps(payload, indent=2))
    _write_csv(output_dir / "scores_rows.csv", rows)
    _render_pdf(output_dir / "plots" / "precip_scores.pdf", payload,
                run_label=run_label)
    LOG.info("precip_scores: %d slices scored -> %s", len(rows), output_dir)
    return output_dir


def aggregate_rows(rows: list[dict]) -> dict:
    """Per-step aggregates over (date, step) rows in the run() row schema.

    Shared with the GRIB-route scorer (eval.evaluators.precip_scores.core.score_gribs), so
    manual-inference NetCDF runs and prepml/FDB GRIB runs report identical
    metric definitions.
    """
    def agg_series(selector) -> dict:
        per_step: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            v = selector(row)
            if v is not None:
                per_step[row["step"]].append(v)
        return {str(s): M.nanmean(vs) for s, vs in sorted(per_step.items())}

    def mem_mean(row, series, key):
        vals = [m.get(series, {}).get(key) for m in row["members"]]
        vals = [v for v in vals if v is not None]
        return M.nanmean(vals) if vals else None

    return {
        "model_rmse_mm": agg_series(lambda r: mem_mean(r, "model", "rmse_mm")),
        "model_bias_mm": agg_series(lambda r: mem_mean(r, "model", "bias_mm")),
        "model_corr": agg_series(lambda r: mem_mean(r, "model", "corr")),
        "model_ens_rmse_mm": agg_series(lambda r: r["model_ens_mean"].get("rmse_mm")),
        "model_p999_mm": agg_series(lambda r: mem_mean(r, "model", "p999_mm")),
        "model_max_mm": agg_series(lambda r: mem_mean(r, "model", "max_mm")),
        "model_wet_frac": agg_series(lambda r: mem_mean(r, "model", "wet_frac")),
        "model_neg_frac": agg_series(lambda r: mem_mean(r, "model", "neg_frac")),
        "truth_p999_mm": agg_series(lambda r: r["truth"].get("p999_mm")),
        "truth_max_mm": agg_series(lambda r: r["truth"].get("max_mm")),
        "truth_wet_frac": agg_series(lambda r: r["truth"].get("wet_frac")),
        "baseline_rmse_mm": agg_series(lambda r: mem_mean(r, "baseline", "rmse_mm")),
        "baseline_corr": agg_series(lambda r: mem_mean(r, "baseline", "corr")),
        "baseline_ens_rmse_mm": agg_series(
            lambda r: r.get("baseline_ens_mean", {}).get("rmse_mm")),
        "baseline_p999_mm": agg_series(lambda r: mem_mean(r, "baseline", "p999_mm")),
        "baseline_max_mm": agg_series(lambda r: mem_mean(r, "baseline", "max_mm")),
        "baseline_wet_frac": agg_series(lambda r: mem_mean(r, "baseline", "wet_frac")),
    }


def summarize(per_step: dict) -> dict:
    """Overall (all steps) summary of aggregate_rows() output."""
    def overall(key):
        vals = [v for v in per_step[key].values() if v is not None]
        return M.nanmean(vals) if vals else None

    summary = {k: overall(k) for k in per_step}
    if summary.get("baseline_rmse_mm") and summary.get("model_rmse_mm"):
        summary["model_over_baseline_rmse_ratio"] = (
            summary["model_rmse_mm"] / summary["baseline_rmse_mm"])
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "step", "member", "series", "rmse_mm", "mae_mm",
                    "bias_mm", "corr", "mean_mm", "max_mm", "p99_mm", "p999_mm",
                    "wet_frac", "neg_frac"])
        for row in rows:
            for mem in row["members"]:
                for series in ("model", "baseline"):
                    s = mem.get(series)
                    if not s:
                        continue
                    w.writerow([row["date"], row["step"], mem["member"], series,
                                s.get("rmse_mm"), s.get("mae_mm"), s.get("bias_mm"),
                                s.get("corr"), s.get("mean_mm"), s.get("max_mm"),
                                s.get("p99_mm"), s.get("p999_mm"),
                                s.get("wet_frac"), s.get("neg_frac")])
            t = row["truth"]
            w.writerow([row["date"], row["step"], "", "truth", "", "", "", "",
                        t.get("mean_mm"), t.get("max_mm"), t.get("p99_mm"),
                        t.get("p999_mm"), t.get("wet_frac"), t.get("neg_frac")])


def _baseline_bias_by_step(rows: list[dict]) -> dict:
    """Member-mean bias of the interpolated input per step (figure only, not in scores.json).

    Same aggregation as ``aggregate_rows``: mean over members, then over the dates of a step.
    """
    per_step: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        vals = [m.get("baseline", {}).get("bias_mm") for m in row["members"]]
        vals = [v for v in vals if v is not None]
        if vals:
            per_step[row["step"]].append(M.nanmean(vals))
    return {str(s): M.nanmean(vs) for s, vs in sorted(per_step.items())}


def _short_source(src: str) -> str:
    kind, sep, rest = str(src).partition(":")
    words = {"grib": "GRIB", "lres-nn": "nearest-neighbour interpolation of GRIB",
             "embedded-y": "embedded y", "x_interp": "embedded x_interp", "none": "none"}
    if sep and "/" in rest:
        return f"{words.get(kind, kind)} {Path(rest).name}"
    return words.get(str(src), str(src))


def render_from_json(scores_json: str | Path, out_pdf: str | Path, *, run_label: str) -> None:
    """Redraw the figures from a saved scores.json (no re-scoring)."""
    payload = json.loads(Path(scores_json).read_text())
    _render_pdf(Path(out_pdf), payload, run_label=run_label)


def _render_pdf(path: Path, payload: dict, *, run_label: str) -> None:
    """Three pages (PDF + PNG per page): skill, tail ratios, summary table.

    Colours by role: model red solid, interpolated input (the baseline) blue dashed, truth
    black. Filled markers are member means (each member scored, then averaged); open markers
    on thinner lines are scores of the ensemble mean. Every number drawn is read from the
    scores.json payload, except the input bias per step, which is averaged from its rows the
    same way.
    """
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eval.plotting import AXIS, FigureBook, eval_style, role_style

    per_step = dict(payload["per_step"])
    per_step["baseline_bias_mm"] = _baseline_bias_by_step(payload.get("rows", []))
    meta = payload["meta"]
    has_baseline = meta["baseline_source"] != "none"
    steps_all = sorted({int(s) for s in per_step.get("model_rmse_mm", {})})
    n_dates = len({r["date"] for r in payload.get("rows", [])})
    thr = meta["wet_threshold_mm"]

    def series(key):
        d = per_step.get(key, {})
        steps = [s for s in sorted(int(s) for s in d) if d[str(s)] is not None]
        return steps, [d[str(s)] for s in steps]

    def plural(n, word):
        return f"{n} {word}{'' if n == 1 else 's'}"

    cases = (f"{plural(meta['n_slices'], 'case')} ({plural(n_dates, 'date')} × "
             f"{plural(len(steps_all), 'lead time')}), {plural(meta['n_members'], 'member')}")
    model_mean = role_style("model", marker="o", markersize=5)
    input_mean = role_style("input", marker="s", markersize=5)
    model_ens = role_style("model", linewidth=1.2, alpha=0.75, marker="o", markersize=5,
                           markerfacecolor="white", linestyle=(0, (1.5, 1.5)))
    input_ens = role_style("input", linewidth=1.2, alpha=0.75, marker="s", markersize=5,
                           markerfacecolor="white", linestyle=(0, (1.5, 1.5)))
    footer = textwrap.fill(
        f"Truth: {_short_source(meta['truth_source'])}. Input (interpolation baseline): "
        f"{_short_source(meta['baseline_source'])}. Checkpoint {meta.get('checkpoint_id', '')}. "
        f"Units: mm per 6 h window; negative values are kept in RMSE, bias and correlation.",
        width=170)

    def finish(fig, axes, handles, labels, name, top_note):
        for ax in axes:
            ax.set_xlabel(AXIS["lead"])
            if steps_all:
                ax.set_xticks(steps_all if len(steps_all) <= 12 else steps_all[::2])
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, -0.075), fontsize=9)
        fig.text(0.5, -0.11, footer, ha="center", va="top", fontsize=7.5, color="0.35")
        fig.suptitle(f"{run_label}: {top_note}\n{cases}", fontsize=12)
        pdf.add(fig, name=name)

    with FigureBook(path, png=True) as pdf, eval_style():
        # ---- page 1: skill against lead time --------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
        panels = [
            ("Root-mean-square error (mm)", "model_rmse_mm", "baseline_rmse_mm",
             ("model_ens_rmse_mm", "baseline_ens_rmse_mm"), None),
            ("Bias, series minus truth (mm)", "model_bias_mm", "baseline_bias_mm", None, 0.0),
            ("Correlation with truth", "model_corr", "baseline_corr", None, None),
        ]
        for ax, (title, mkey, bkey, ens, ref) in zip(axes, panels):
            if ref is not None:
                ax.axhline(ref, color="0.3", linewidth=0.9, zorder=1)
            ax.plot(*series(mkey), **model_mean)
            if has_baseline:
                ax.plot(*series(bkey), **input_mean)
            if ens:
                ax.plot(*series(ens[0]), **model_ens)
                if has_baseline:
                    ax.plot(*series(ens[1]), **input_ens)
            ax.set_title(title)
        handles = [plt.Line2D([], [], **model_mean)]
        labels = ["Model, member mean"]
        if has_baseline:
            handles.append(plt.Line2D([], [], **input_mean))
            labels.append("Input (interpolated), member mean")
        handles.append(plt.Line2D([], [], **model_ens))
        labels.append("Model, ensemble mean (RMSE only)")
        if has_baseline:
            handles.append(plt.Line2D([], [], **input_ens))
            labels.append("Input, ensemble mean (RMSE only)")
        finish(fig, axes, handles, labels, "skill",
               "6 h precipitation skill against lead time")

        # ---- page 2: tail ratios against lead time --------------------------
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
        panels = [
            ("99.9th percentile", "p999_mm", "mm"),
            ("Maximum", "max_mm", "mm"),
            (f"Wet fraction (at least {thr:g} mm)", "wet_frac", ""),
        ]
        for ax, (title, key, unit) in zip(axes, panels):
            ax.axhline(1.0, color=role_style("truth")["color"], linewidth=1.2, zorder=1)
            t_steps, t_vals = series(f"truth_{key}")
            truth_at = dict(zip(t_steps, t_vals))
            for skey, style in (("model", model_mean), ("baseline", input_mean)):
                if skey == "baseline" and not has_baseline:
                    continue
                s, v = series(f"{skey}_{key}")
                pairs = [(st, val / truth_at[st]) for st, val in zip(s, v)
                         if truth_at.get(st)]
                if pairs:
                    ax.plot(*zip(*pairs), **style)
            if t_vals:
                lo, hi = min(t_vals), max(t_vals)
                fmt = (lambda x: f"{x:.0f} {unit}") if unit else (lambda x: f"{100 * x:.0f} %")
                span = fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)} to {fmt(hi)}"
                title = f"{title}, series / truth\n(truth: {span})"
            ax.set_title(title, fontsize=10.5)
            ax.set_ylabel("Ratio to truth")
            lo, hi = ax.get_ylim()
            pad = max(abs(1 - lo), abs(hi - 1), 0.05) * 1.1
            ax.set_ylim(1 - pad, 1 + pad)
        handles = [plt.Line2D([], [], **model_mean)]
        labels = ["Model, member mean"]
        if has_baseline:
            handles.append(plt.Line2D([], [], **input_mean))
            labels.append("Input (interpolated), member mean")
        handles.append(plt.Line2D([], [], color=role_style("truth")["color"], linewidth=1.2))
        labels.append("Truth (ratio 1)")
        finish(fig, axes, handles, labels, "tails",
               "6 h precipitation distribution tails relative to truth")

        # ---- page 3: summary table -----------------------------------------
        summ = payload["summary"]
        b_bias = [v for v in per_step["baseline_bias_mm"].values() if v is not None]
        summ_b_bias = M.nanmean(b_bias) if b_bias else None

        def cell(v, fmt):
            return "" if v is None else format(v, fmt)

        table_rows = [
            ("RMSE, member mean (mm)", "rmse_mm", ".2f", True),
            ("RMSE of the ensemble mean (mm)", "ens_rmse_mm", ".2f", True),
            ("Bias (mm)", "bias_mm", "+.3f", True),
            ("Correlation", "corr", ".3f", True),
            ("99.9th percentile (mm)", "p999_mm", ".1f", False),
            ("Maximum (mm)", "max_mm", ".1f", False),
            (f"Wet fraction (at least {thr:g} mm)", "wet_frac", ".3f", False),
            ("Negative fraction", "neg_frac", ".3f", False),
        ]
        cells = []
        for label, key, fmt, paired in table_rows:
            model_v = summ.get(f"model_{key}")
            base_v = summ_b_bias if key == "bias_mm" else summ.get(f"baseline_{key}")
            truth_v = None if paired else summ.get(f"truth_{key}")
            cells.append([label, cell(model_v, fmt),
                          cell(base_v, fmt) if has_baseline else "", cell(truth_v, fmt)])
        ratio = summ.get("model_over_baseline_rmse_ratio")
        if ratio is not None:
            cells.append(["RMSE ratio, model / input", f"{ratio:.3f}", "", ""])
        n_rows = len(cells) + 1
        fig = plt.figure(figsize=(9.0, 0.34 * n_rows + 1.5))
        ax = fig.add_axes([0.03, 0.9 / (0.34 * n_rows + 1.5), 0.94,
                           0.34 * n_rows / (0.34 * n_rows + 1.5)])
        ax.axis("off")
        tab = ax.table(cellText=cells,
                       colLabels=["Mean over lead times", "Model",
                                  "Input (interpolated)", "Truth"],
                       colLoc="center", cellLoc="right", bbox=[0, 0, 1, 1],
                       colWidths=[0.43, 0.17, 0.24, 0.16])
        tab.auto_set_font_size(False)
        tab.set_fontsize(9.5)
        for (r, c), cl in tab.get_celld().items():
            cl.set_edgecolor("0.8")
            if r == 0:
                cl.set_text_props(weight="bold")
                cl.set_facecolor("0.93")
            if c == 0:
                cl.set_text_props(ha="left")
                cl.PAD = 0.03
        fig.suptitle(f"{run_label}: 6 h precipitation scores, mean over lead times\n{cases}",
                     fontsize=12)
        fig.text(0.5, 0.1 / (0.34 * n_rows + 1.5), footer, ha="center", va="bottom",
                 fontsize=7.5, color="0.35")
        pdf.add(fig, name="summary")
