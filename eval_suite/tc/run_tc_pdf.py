import logging
import os

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from tc_events import EVENTS
from tc_pdf_plot import plot_event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DIR_DATA_BASE = "/home/ecm5702/hpcperm/data/tc"
OUTDIR = "/home/ecm5702/dev"

list_ml_exps = [
    "ENFO_O320_ip6y",
    "ENFO_O320_j0ys",
    "ENFO_O320_iz2r",
]

list_ml_exps = [
    "ENFO_O320_iz2q",
    "ENFO_O320_iytd",
    "ENFO_O320_iysd",
    "ENFO_O320_iytc",
    "ENFO_O320_iz2p",
    "ENFO_O320_iz2o",
]

EVENTS_TO_RUN = {
    "dora": list_ml_exps,
    "hilary": list_ml_exps,
    "fernanda": list_ml_exps,
    "idalia": list_ml_exps,
    "franklin": list_ml_exps,
}

os.makedirs(OUTDIR, exist_ok=True)
out_pdf = os.path.join(OUTDIR, "tc_normed_pdfs_all_events.pdf")

with PdfPages(out_pdf) as pdf:
    for event_name, ml_exps in EVENTS_TO_RUN.items():
        cfg = EVENTS[event_name]
        logger.info("Running event=%s with ML=%s", event_name, ml_exps)

        fig = plot_event(
            cfg,
            dir_data_base=DIR_DATA_BASE,
            out_path="unused",  # not used anymore
            include_ml=ml_exps,
            exclude_ml=None,
        )

        pdf.savefig(fig, dpi=300)
        plt.close(fig)
        # or: import matplotlib.pyplot as plt; plt.close(fig)

logger.info("Saved combined PDF to %s", out_pdf)
