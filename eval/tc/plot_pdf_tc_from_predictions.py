from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from .tc_events import EVENTS
    from .tc_pdf_plot import plot_event_curves
    from .tc_vector_loading import (
        analysis_dates_for_event,
        discover_prediction_files,
        event_days_steps,
        forecast_dates_for_event,
        load_grib_event_curves,
        load_prediction_event_curve,
        regridded_target_points_from_grib,
        select_prediction_files_for_event,
    )
except ImportError:  # allow running as a script
    from tc_events import EVENTS
    from tc_pdf_plot import plot_event_curves
    from tc_vector_loading import (
        analysis_dates_for_event,
        discover_prediction_files,
        event_days_steps,
        forecast_dates_for_event,
        load_grib_event_curves,
        load_prediction_event_curve,
        regridded_target_points_from_grib,
        select_prediction_files_for_event,
    )

LOG = logging.getLogger(__name__)


def run_tc_pdf_from_predictions(
    *,
    predictions_dir: str,
    outdir: str,
    run_label: str,
    display_label: str | None = None,
    out_name: str = "",
    base_tc_dir: str = "/home/ecm5702/hpcperm/data/tc",
    extra_reference_expids: list[str] | None = None,
    support_mode: str = "regridded",
    event_names: list[str] | None = None,
) -> str:
    pred_dir = Path(predictions_dir).expanduser().resolve()
    out_dir = Path(outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = discover_prediction_files(pred_dir)
    if not pred_files:
        raise FileNotFoundError(f"No predictions_*.nc files found in {pred_dir}")

    selected_events = event_names or list(EVENTS.keys())
    extra_reference_expids = extra_reference_expids or []

    if not out_name:
        out_name = f"tc_normed_pdfs_{run_label}_{support_mode}_from_predictions.pdf"
    out_pdf = out_dir / out_name
    out_stats = out_dir / f"{out_pdf.stem}.stats.json"

    payload: dict[str, object] = {
        "run_label": run_label,
        "display_label": display_label or run_label,
        "predictions_dir": str(pred_dir),
        "pdf_file": str(out_pdf),
        "support_mode": support_mode,
        "events": {},
    }

    rendered_events = 0
    with PdfPages(out_pdf) as pdf:
        for event_name in selected_events:
            cfg = EVENTS[event_name]
            event_pred_files = select_prediction_files_for_event(pred_files, cfg)
            if not event_pred_files:
                LOG.info("Skipping event=%s: no matching prediction files", event_name)
                continue

            days, steps = event_days_steps(event_pred_files)
            LOG.info(
                "Running event=%s | support=%s | files=%d | days=%s | steps=%s",
                event_name,
                support_mode,
                len(event_pred_files),
                days,
                steps,
            )

            target_lon = None
            target_lat = None
            if support_mode == "regridded":
                target_lon, target_lat = regridded_target_points_from_grib(
                    cfg,
                    dir_data_base=base_tc_dir,
                    sample_analysis_date=analysis_dates_for_event(cfg, days)[0],
                )
            prediction_curve = load_prediction_event_curve(
                event_pred_files,
                cfg=cfg,
                support_mode=support_mode,
                target_lon=target_lon,
                target_lat=target_lat,
            )
            reference_curves = load_grib_event_curves(
                cfg,
                dir_data_base=base_tc_dir,
                extra_reference_expids=extra_reference_expids,
                support_mode=support_mode,
                forecast_dates=forecast_dates_for_event(cfg, days),
                analysis_dates=analysis_dates_for_event(cfg, days),
                steps=steps,
                max_pf_members=10,
            )

            curves = {
                run_label: prediction_curve,
                **reference_curves,
            }
            reference_curve_order = list(dict.fromkeys([*extra_reference_expids, *cfg.reference_expids]))
            exp_labels = {
                run_label: display_label or run_label,
                **{
                    expid: expid.replace("ENFO_O320_", "")
                    for expid in extra_reference_expids
                },
            }
            fig, event_stats = plot_event_curves(
                cfg,
                curves=curves,
                curve_order=[run_label, *reference_curve_order],
                exp_labels=exp_labels,
                return_stats=True,
            )
            event_stats["selected_days"] = days
            event_stats["steps_hours"] = steps
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
            payload["events"][event_name] = event_stats
            rendered_events += 1

    if rendered_events == 0:
        raise RuntimeError("No matching events found in predictions directory.")

    with open(out_stats, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    LOG.info("Saved TC PDF from predictions: %s", out_pdf)
    LOG.info("Saved TC stats JSON: %s", out_stats)
    return str(out_pdf)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TC normalized PDFs directly from predictions_*.nc files."
    )
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument(
        "--display-label",
        default="",
        help="Optional shorter human-readable label for legends while keeping run-label as the stats key.",
    )
    parser.add_argument("--out-name", default="")
    parser.add_argument("--base-tc-dir", default="/home/ecm5702/hpcperm/data/tc")
    parser.add_argument(
        "--extra-reference-expids",
        default="",
        help="Comma-separated extra reference expids, e.g. ENFO_O320_j138,ENFO_O320_j24v",
    )
    parser.add_argument(
        "--support-mode",
        choices=["native", "regridded"],
        default="regridded",
        help="Use native supports directly or regrid every curve onto the canonical regular TC grid.",
    )
    parser.add_argument(
        "--events",
        default="",
        help="Optional comma-separated subset of events from tc_events.py.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    extra_reference_expids = [
        value.strip()
        for value in args.extra_reference_expids.split(",")
        if value.strip()
    ]
    event_names = [
        value.strip()
        for value in args.events.split(",")
        if value.strip()
    ] or None

    run_tc_pdf_from_predictions(
        predictions_dir=args.predictions_dir,
        outdir=args.outdir,
        run_label=args.run_label,
        display_label=args.display_label or None,
        out_name=args.out_name,
        base_tc_dir=args.base_tc_dir,
        extra_reference_expids=extra_reference_expids,
        support_mode=args.support_mode,
        event_names=event_names,
    )


if __name__ == "__main__":
    main()
