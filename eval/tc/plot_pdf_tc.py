import logging
import os
import argparse
import json

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

try:
    from .tc_events import EVENTS
    from .tc_pdf_plot import plot_event
except ImportError:  # allow running as a script
    from tc_events import EVENTS
    from tc_pdf_plot import plot_event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DIR_DATA_BASE = "/home/ecm5702/hpcperm/data/tc"
def run_tc_pdf(
    *,
    expver: str,
    outdir: str,
    out_name: str = "",
    exp_prefix: str = "ENFO_O320",
    support_mode: str = "regridded",
) -> str:
    exp_key = f"{exp_prefix}_{expver}"
    exp_labels = {exp_key: expver}
    # ip6y is always plotted as a fixed reference in tc_pdf_plot.py.
    # Avoid plotting it twice if user asks for expver=ip6y.
    list_ml_exps = [] if exp_key == "ENFO_O320_ip6y" else [exp_key]
    events_to_run = {
        "dora": list_ml_exps,
        "hilary": list_ml_exps,
        "fernanda": list_ml_exps,
        "idalia": list_ml_exps,
        "franklin": list_ml_exps,
    }

    os.makedirs(outdir, exist_ok=True)
    if not out_name:
        out_name = f"tc_normed_pdfs_all_events_{expver}.pdf"
    out_pdf = os.path.join(outdir, out_name)
    out_stats = os.path.join(outdir, f"{os.path.splitext(out_name)[0]}.stats.json")
    all_stats: dict[str, object] = {
        "expver": expver,
        "exp_prefix": exp_prefix,
        "pdf_file": out_pdf,
        "support_mode": support_mode,
        "events": {},
    }

    with PdfPages(out_pdf) as pdf:
        for event_name, ml_exps in events_to_run.items():
            cfg = EVENTS[event_name]
            logger.info("Running event=%s with ML=%s", event_name, ml_exps)
            fig, event_stats = plot_event(
                cfg,
                dir_data_base=DIR_DATA_BASE,
                out_path="unused",
                include_ml=ml_exps,
                exclude_ml=None,
                exp_labels=exp_labels,
                return_stats=True,
                support_mode=support_mode,
            )
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
            all_stats["events"][event_name] = event_stats

    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, sort_keys=True)

    logger.info("Saved combined PDF to %s", out_pdf)
    logger.info("Saved TC stats JSON to %s", out_stats)
    return out_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TC normalized PDFs for all configured events.")
    parser.add_argument("--expver", default="j24v", help="ML expver (prefix configured by --exp-prefix).")
    parser.add_argument("--outdir", default="/home/ecm5702/dev", help="Output directory for combined PDF.")
    parser.add_argument("--out-name", default="", help="Optional output PDF filename.")
    parser.add_argument(
        "--exp-prefix",
        default="ENFO_O320",
        help="Prefix for ML TC GRIB ids (e.g. ENFO_O320 or ENFO_O1280).",
    )
    parser.add_argument(
        "--support-mode",
        choices=["native", "regridded"],
        default="regridded",
        help="Use native supports directly or regrid every curve onto the canonical regular TC grid.",
    )
    args = parser.parse_args()
    run_tc_pdf(
        expver=args.expver,
        outdir=args.outdir,
        out_name=args.out_name,
        exp_prefix=args.exp_prefix,
        support_mode=args.support_mode,
    )


if __name__ == "__main__":
    main()
