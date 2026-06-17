#!/usr/bin/env python3
"""Side-by-side TC PDFs: cfec83a3 manual_inference vs PrepML on Idalia + Franklin.

Mirrors tc_top7_franklin_idalia_pdfs.py but with only two curves: the manual
baseline (run_20260512T114030) and the per-member-truth-fixed prepml run
(run_20260515T_retrieve_test_idalia). Same checkpoint (cfec83a3), same dates
(2023-08-26..30), same members (1..10), same steps (24..120).

Output: 8 figures (2 events x 2 support modes x 2 plot modes [ratio, log])
in one PDF, plus a JSON summary with the per-curve min/max + tail stats.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval._backends.tc.events import EVENTS
from eval._backends.tc.loading_grib import load_grib_curves, regridded_target_points
from eval._backends.tc.loading_predictions import (
    discover_prediction_files,
    event_days_steps,
    forecast_dates_for_event,
    load_prediction_curves,
    select_prediction_files_for_event,
)
from eval._backends.tc.pdf_plot import plot_pdf_log, plot_pdf_ratios
from eval._backends.tc.plot_config import PLOT_CONFIGS
from eval._backends.tc.workflows import compute_event_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger(__name__)

GRIB_DIR = "/home/ecm5702/perm/reference/o96_o320/tc"
ANALYSIS_EXPID = "OPER_O320_0001"
REFERENCE_EXPIDS = ["ENFO_O320_0001", "EEFO_O96_0001"]
EVENT_NAMES = ["franklin", "idalia"]
SUPPORT_MODES = ["regridded", "native"]

OUTDIR = Path("/home/ecm5702/scratch/eval/o96_o320/tc_cfec83a3_manual_vs_prepml")
OUT_PDF_NAME = "all_tc_distributions.pdf"

CHECKPOINTS: list[dict] = [
    {
        "label": "cfec83a3_manual",
        "display": "cfec83a3 manual_inference",
        "dir": Path("/home/ecm5702/scratch/eval/o96_o320/run_20260512T114030"),
    },
    {
        "label": "cfec83a3_prepml",
        "display": "cfec83a3 PrepML (per-member-truth fix)",
        "dir": Path("/home/ecm5702/scratch/eval/o96_o320/run_20260515T_retrieve_test_idalia"),
    },
]


def _per_event_summary(curves: dict, ckpt_displays: dict[str, str]) -> dict:
    import numpy as np
    out = {}
    for key, curve in curves.items():
        label = ckpt_displays.get(key, key)
        msl_finite = curve.msl[np.isfinite(curve.msl)] if curve.msl.size else curve.msl
        wind_finite = curve.wind[np.isfinite(curve.wind)] if curve.wind.size else curve.wind
        out[key] = {
            "label": label,
            "msl_min": float(msl_finite.min()) if msl_finite.size else None,
            "wind_max": float(wind_finite.max()) if wind_finite.size else None,
            "n_msl": int(curve.msl.size),
            "n_wind": int(curve.wind.size),
        }
    return out


def build_event_stats(event_name: str, support_mode: str) -> tuple[dict, dict, dict]:
    event = EVENTS[event_name]
    plot_cfg = PLOT_CONFIGS[event_name]
    plot_cfg = replace(plot_cfg, plot_title=f"{plot_cfg.plot_title} [{support_mode}] manual vs PrepML")

    first_pred_dir = CHECKPOINTS[0]["dir"] / "predictions"
    all_pred_files = discover_prediction_files(first_pred_dir)
    event_pred_files = select_prediction_files_for_event(all_pred_files, event)
    if not event_pred_files:
        raise FileNotFoundError(f"No prediction files for {event_name} in {first_pred_dir}")
    days_from_preds, steps_from_preds = event_days_steps(event_pred_files)
    forecast_dates = forecast_dates_for_event(event, days_from_preds)
    LOG.info("days=%s steps=%s", days_from_preds, steps_from_preds)

    LOG.info("Loading references for %s [%s] ...", event_name, support_mode)
    ref_curves = load_grib_curves(
        dir_data_base=GRIB_DIR,
        event_name=event_name,
        analysis_expid=ANALYSIS_EXPID,
        analysis_dates=list(event.analysis_dates),
        forecast_dates=forecast_dates,
        reference_expids=REFERENCE_EXPIDS,
        support_mode=support_mode,
        bbox=event.bbox if support_mode == "regridded" else None,
        regrid_resolution=plot_cfg.regrid_resolution,
        steps=steps_from_preds,
        max_pf_members=10,
    )

    target_lon = target_lat = None
    if support_mode == "regridded":
        sample_an = f"{GRIB_DIR}/{event_name}/surface_an_{ANALYSIS_EXPID}_{event.analysis_dates[0]}.grib"
        target_lon, target_lat = regridded_target_points(
            event.bbox, plot_cfg.regrid_resolution, sample_an,
        )

    curves: dict = dict(ref_curves)
    for ckpt in CHECKPOINTS:
        pred_dir = ckpt["dir"] / "predictions"
        pred_files_all = discover_prediction_files(pred_dir)
        ckpt_pred_files = select_prediction_files_for_event(pred_files_all, event)
        if not ckpt_pred_files:
            LOG.warning("No predictions for ckpt=%s event=%s — skipping", ckpt["label"], event_name)
            continue
        LOG.info("Loading predictions: %s [%s] (%d files)", ckpt["display"], support_mode, len(ckpt_pred_files))
        pred_curve = load_prediction_curves(
            ckpt_pred_files,
            bbox=event.bbox,
            support_mode=support_mode,
            target_lon=target_lon,
            target_lat=target_lat,
        )
        curves[ckpt["label"]] = pred_curve

    curve_order = list(REFERENCE_EXPIDS) + [c["label"] for c in CHECKPOINTS if c["label"] in curves]

    event_stats = compute_event_stats(
        curves,
        analysis_key=ANALYSIS_EXPID,
        plot_config=plot_cfg,
        curve_order=curve_order,
    )
    event_stats["event"] = event_name
    event_stats["support_mode"] = support_mode
    event_stats["selected_days"] = days_from_preds
    event_stats["steps_hours"] = steps_from_preds
    return event_stats, plot_cfg, curves


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUTDIR / OUT_PDF_NAME
    out_summary = OUTDIR / f"{out_pdf.stem}.summary.json"

    exp_labels: dict[str, str] = {c["label"]: c["display"] for c in CHECKPOINTS}
    ckpt_displays = dict(exp_labels)
    ckpt_displays[ANALYSIS_EXPID] = "OPER AN"

    summary: dict = {
        "out_pdf": str(out_pdf),
        "support_modes": SUPPORT_MODES,
        "plot_modes": ["ratio", "log"],
        "analysis_expid": ANALYSIS_EXPID,
        "reference_expids": REFERENCE_EXPIDS,
        "events": EVENT_NAMES,
        "checkpoints": [
            {"label": c["label"], "display": c["display"], "dir": str(c["dir"])}
            for c in CHECKPOINTS
        ],
        "figures": [],
    }

    figures_rendered = 0
    with PdfPages(out_pdf) as pdf:
        for mode in SUPPORT_MODES:
            for event_name in EVENT_NAMES:
                LOG.info("=== %s [%s] ===", event_name, mode)
                try:
                    stats, plot_cfg, curves = build_event_stats(event_name, mode)
                except Exception:
                    LOG.exception("Failed to build stats for %s [%s]", event_name, mode)
                    continue

                fig_ratio = plot_pdf_ratios(plot_cfg, event_stats=stats, exp_labels=exp_labels)
                pdf.savefig(fig_ratio, dpi=300)
                plt.close(fig_ratio)
                figures_rendered += 1
                LOG.info("Rendered ratio figure: %s [%s]", event_name, mode)

                fig_log = plot_pdf_log(plot_cfg, event_stats=stats, exp_labels=exp_labels)
                pdf.savefig(fig_log, dpi=300)
                plt.close(fig_log)
                figures_rendered += 1
                LOG.info("Rendered log figure: %s [%s]", event_name, mode)

                slim_vars = {}
                for var_name, var_block in stats["variables"].items():
                    slim_curves = {
                        ck: {kk: vv for kk, vv in cb.items() if kk != "histogram"}
                        for ck, cb in var_block["curves"].items()
                    }
                    slim_vars[var_name] = {
                        "oper": var_block["oper"],
                        "data_range_msl": var_block.get("data_range_msl"),
                        "data_range_wind": var_block.get("data_range_wind"),
                        "curves": slim_curves,
                    }

                summary["figures"].append({
                    "event": event_name,
                    "support_mode": mode,
                    "plot_modes": ["ratio", "log"],
                    "selected_days": stats["selected_days"],
                    "steps_hours": stats["steps_hours"],
                    "analysis_key": stats["analysis_key"],
                    "curve_order": stats["curve_order"],
                    "curves_minmax": _per_event_summary(curves, ckpt_displays),
                    "variables": slim_vars,
                    "extreme_tail": stats.get("extreme_tail"),
                })

    LOG.info("Wrote PDF: %s (%d figures)", out_pdf, figures_rendered)
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    LOG.info("Wrote summary: %s", out_summary)


if __name__ == "__main__":
    main()
