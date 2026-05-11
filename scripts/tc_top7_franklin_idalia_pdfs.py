#!/usr/bin/env python3
"""Top-7 TC checkpoints — Franklin/Idalia normalized PDF overlay (o96_o320 lane).

Mirrors `eval.cli` evaluate output for the o96_o320 lane: produces an 8-figure
combined PDF (2 events × 2 support_modes × 2 plot modes — ratio-vs-OPER and raw
log-scale). Order matches `eval/evaluators/tc/runner.py` (regridded first, then
native) and `eval/evaluators/tc/plotter.py` (ratio first, then log).

Roster ranking source:
  /home/ecm5702/dev/docs/docs/scoreboard_o96_o320/state/source_26_30/scoreboard.csv
Top 7 by tc_rank, with rank #5 (no predictions on disk) substituted by rank #8.
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
SUPPORT_MODES = ["regridded", "native"]  # matches eval/evaluators/tc/runner.py order

OUTDIR = Path("/home/ecm5702/scratch/eval/o96_o320/tc_top7_20260510")
OUT_PDF_NAME = "all_tc_distributions.pdf"

PERM_EVAL = Path("/home/ecm5702/perm/eval")

# Top-7 by tc_rank from scoreboard.csv (rank #5 has no predictions; rank #8 substitutes).
CHECKPOINTS: list[dict] = [
    {
        "rank": 1,
        "label": "manual_61cf1811_new_o96_o320_20260320_piecewise30_h10_l20_sigma100_ref",
        "display": "R1 61cf pw30 exp/k",
        "dir": PERM_EVAL / "manual_61cf1811_new_o96_o320_20260320_piecewise30_h10_l20_sigma100_ref",
    },
    {
        "rank": 2,
        "label": "manual_61cf1811_new_o96_o320_20260320_oldlike100kft",
        "display": "R2 61cf k40 old100kft",
        "dir": PERM_EVAL / "manual_61cf1811_new_o96_o320_20260320_oldlike100kft",
    },
    {
        "rank": 3,
        "label": "manual_61cf1811_new_o96_o320_20260323_karras_n80_sigmax100k",
        "display": "R3 61cf k80 s100k",
        "dir": PERM_EVAL / "manual_61cf1811_new_o96_o320_20260323_karras_n80_sigmax100k",
    },
    {
        "rank": 4,
        "label": "manual_56b6c4e2_old_o96_o320_20260312_piecewise30_h10_l20_sigma100_classic_piecewisegen4_mixedroot",
        "display": "R4 56b6 old pw30",
        "dir": PERM_EVAL / "manual_56b6c4e2_old_o96_o320_20260312_piecewise30_h10_l20_sigma100_classic_piecewisegen4_mixedroot",
    },
    {
        "rank": 6,
        "label": "manual_cfec83a3_new_o96_o320_20260323_piecewise30_h10_l20_sigma100",
        "display": "R6 cfec83 pw30",
        "dir": PERM_EVAL / "manual_cfec83a3_new_o96_o320_20260323_piecewise30_h10_l20_sigma100",
    },
    {
        "rank": 7,
        "label": "manual_59e40596_new_o96_o320_20260422_pw23_t7_h7_l16",
        "display": "R7 59e4 pw23 t7h7l16",
        "dir": PERM_EVAL / "manual_59e40596_new_o96_o320_20260422_pw23_t7_h7_l16",
    },
    {
        # Substitutes rank 5 (manual_cfec83a3..._oldlike200k has no predictions).
        "rank": 8,
        "label": "manual_59e4_300k",
        "display": "R8 59e4 300k pw30",
        "dir": PERM_EVAL / "manual_59e4_300k",
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
    """Load curves + compute stats for one (event, support_mode). Returns (stats, plot_cfg_replaced, curves_for_summary)."""
    event = EVENTS[event_name]
    plot_cfg = PLOT_CONFIGS[event_name]
    plot_cfg = replace(plot_cfg, plot_title=f"{plot_cfg.plot_title} [{support_mode}]")

    # Derive dates/steps from first checkpoint's prediction file inventory.
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
            {"rank": c["rank"], "label": c["label"], "display": c["display"], "dir": str(c["dir"])}
            for c in CHECKPOINTS
        ],
        "figures": [],
    }

    figures_rendered = 0
    with PdfPages(out_pdf) as pdf:
        # Order mirrors eval/evaluators/tc/runner.py: outer loop = mode, inner = event.
        for mode in SUPPORT_MODES:
            for event_name in EVENT_NAMES:
                LOG.info("=== %s [%s] ===", event_name, mode)
                try:
                    stats, plot_cfg, curves = build_event_stats(event_name, mode)
                except Exception:
                    LOG.exception("Failed to build stats for %s [%s]", event_name, mode)
                    continue

                # Two figures per (event, mode): ratio-vs-analysis and raw log.
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

                # Slim copy of stats: drop histogram arrays (huge), keep summary/tail/extreme_tail.
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
