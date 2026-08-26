"""precip_scores evaluator — pointwise + distribution skill for 6h-window tp.

Scores model tp (and the interpolation baseline) against 6h-window truth at
the same step, in mm. Truth comes from the predictions' embedded `y` when the
tp channel is populated, otherwise from the lane's `precip.truth_grib_tpl`
GRIB (the o1280->o2560 main-lane bundles historically carried no tp truth).
The baseline comes from `x_interp` when its tp channel is a real series,
otherwise from the driving o1280 member tp via `precip.baseline_lres_grib_tpl`
(tp is output-only on this lane, so the exported x_interp tp is all zero).

Outputs: scores.json (machine-readable, scoreboard-ingestable), scores_rows.csv,
plots/precip_scores.pdf.
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval._backends.precip import metrics as M
from eval._backends.precip.sources import (
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

    Shared with the GRIB-route scorer (eval._backends.precip.score_gribs), so
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


def _render_pdf(path: Path, payload: dict, *, run_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    per_step = payload["per_step"]
    meta = payload["meta"]

    def series(key):
        d = per_step.get(key, {})
        steps = sorted(int(s) for s in d)
        return steps, [d[str(s)] for s in steps]

    has_baseline = meta["baseline_source"] != "none"
    C_MODEL, C_BASE, C_TRUTH = "#d95f02", "#555555", "#222222"

    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
        fig.suptitle(f"tp skill vs lead — {run_label} | truth: {meta['truth_source']}"
                     f" | baseline: {meta['baseline_source']}", fontsize=10)
        for ax, (mkey, bkey, ekey, title, unit) in zip(axes, [
            ("model_rmse_mm", "baseline_rmse_mm", "model_ens_rmse_mm",
             "RMSE", "mm / 6h"),
            ("model_bias_mm", None, None, "Bias", "mm / 6h"),
            ("model_corr", "baseline_corr", None, "Correlation", ""),
        ]):
            s, v = series(mkey)
            ax.plot(s, v, "-o", color=C_MODEL, label="model (member mean)", ms=3)
            if bkey and has_baseline:
                s, v = series(bkey)
                ax.plot(s, v, "-s", color=C_BASE, label="interp baseline", ms=3)
            if ekey:
                s, v = series(ekey)
                ax.plot(s, v, "--", color=C_MODEL, alpha=0.6, label="model (ens mean)")
                if has_baseline:
                    s, v = series("baseline_ens_rmse_mm")
                    ax.plot(s, v, "--", color=C_BASE, alpha=0.6,
                            label="baseline (ens mean)")
            if title == "Bias":
                ax.axhline(0, color="k", lw=0.5)
            ax.set_title(title)
            ax.set_xlabel("lead (h)")
            ax.set_ylabel(unit)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
        fig.suptitle(f"tp distribution tails vs lead — {run_label}", fontsize=10)
        for ax, (tkey, mkey, bkey, title) in zip(axes, [
            ("truth_p999_mm", "model_p999_mm", "baseline_p999_mm", "p99.9 (mm/6h)"),
            ("truth_max_mm", "model_max_mm", "baseline_max_mm", "max (mm/6h)"),
            ("truth_wet_frac", "model_wet_frac", "baseline_wet_frac",
             f"wet fraction (> {meta['wet_threshold_mm']:g} mm)"),
        ]):
            s, v = series(tkey)
            ax.plot(s, v, "-", color=C_TRUTH, lw=2, label="truth")
            s, v = series(mkey)
            ax.plot(s, v, "-o", color=C_MODEL, label="model", ms=3)
            if has_baseline:
                s, v = series(bkey)
                ax.plot(s, v, "-s", color=C_BASE, label="interp baseline", ms=3)
            ax.set_title(title)
            ax.set_xlabel("lead (h)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axis("off")
        lines = [f"precip_scores summary — {run_label}",
                 f"checkpoint: {meta['checkpoint_id']}",
                 f"slices: {meta['n_slices']}  members: {meta['n_members']}",
                 f"negative handling: {meta['negative_handling']}", ""]
        for k, v in payload["summary"].items():
            if v is not None:
                lines.append(f"{k:38s} {v:10.4f}")
        ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace",
                fontsize=9, transform=ax.transAxes)
        pdf.savefig(fig)
        plt.close(fig)
