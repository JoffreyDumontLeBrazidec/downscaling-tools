import logging
import os
import argparse

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


def run_tc_pdf(*, expver: str, outdir: str, out_name: str = "") -> str:
    exp_key = f"ENFO_O320_{expver}"
    exp_labels = {exp_key: expver}
    list_ml_exps = [exp_key]
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

    with PdfPages(out_pdf) as pdf:
        for event_name, ml_exps in events_to_run.items():
            cfg = EVENTS[event_name]
            logger.info("Running event=%s with ML=%s", event_name, ml_exps)
            fig = plot_event(
                cfg,
                dir_data_base=DIR_DATA_BASE,
                out_path="unused",
                include_ml=ml_exps,
                exclude_ml=None,
                exp_labels=exp_labels,
            )
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

    logger.info("Saved combined PDF to %s", out_pdf)
    return out_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TC normalized PDFs for all configured events.")
    parser.add_argument("--expver", default="j24v", help="ML expver (without ENFO_O320_ prefix).")
    parser.add_argument("--outdir", default="/home/ecm5702/dev", help="Output directory for combined PDF.")
    parser.add_argument("--out-name", default="", help="Optional output PDF filename.")
    args = parser.parse_args()
    run_tc_pdf(expver=args.expver, outdir=args.outdir, out_name=args.out_name)


if __name__ == "__main__":
    main()
