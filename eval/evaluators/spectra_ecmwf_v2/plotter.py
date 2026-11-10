"""ECMWF spectra evaluator — PDF plots."""
from __future__ import annotations

import json
import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


def plot(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Build the spectra PDFs for this run.

    `spectra_ecmwf_grid.png` / `.pdf` is the standard figure (eval.plotting.spectra_grid):
    every variable in one figure, the amplitude spectrum of truth, input and model above and
    the amplitude ratio to the truth below. Two more files are written side by side. `spectra_ecmwf.pdf` holds the
    mean power spectra (the square of the stored amplitudes, in the
    variable's display unit squared), with the model, truth and input
    curves on log-log axes. `spectra_ecmwf_ratio.pdf` holds the same model
    and input spectra divided by the truth spectrum, so that a perfect
    match is the horizontal line at one and departures of a few percent
    are readable. The ratio file needs the truth curves and is skipped
    without them. Each PDF also gets one PNG per page in
    `spectra_ecmwf_pages/` and `spectra_ecmwf_ratio_pages/`.
    """
    from ._plotter import build_pdf_ecmwf_ratio, build_pdf_ecmwf_with_references, build_spectra_grid

    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_amp_dir = results_dir / "spectra"
    if not pred_amp_dir.exists():
        LOG.warning("spectra_ecmwf plotter: spectra dir not found: %s", pred_amp_dir)
        return output_dir

    # References are addressed by evaluation window, so their directory name
    # cannot be reconstructed here. The evaluator records where it put them.
    truth_amp_dir: Path | None = None
    input_amp_dir: Path | None = None
    recorded = ""
    summary_path = results_dir / "spectra_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            recorded = str(summary.get("reference_spectra_dir", "") or "").strip()
        except (OSError, ValueError):
            recorded = ""

    # Rule 41: the runner records where the input curve lives (input_native/ for the native x,
    # input/ for x_interp) and where it came from in run_meta.json.
    recorded_input = ""
    meta_path = results_dir / "run_meta.json"
    if meta_path.exists():
        try:
            recorded_input = str(json.loads(meta_path.read_text(encoding="utf-8"))
                                 .get("input_reference_dir", "") or "").strip()
        except (OSError, ValueError):
            recorded_input = ""

    if recorded:
        truth_amp_dir = Path(recorded)
        # truth and the x_interp input sit side by side under the same window key
        input_amp_dir = Path(recorded.replace("/truth/", "/input/", 1))
        if input_amp_dir == truth_amp_dir:
            input_amp_dir = None
    else:
        reference_dir = eval_config.get("reference_dir", "")
        if reference_dir:
            LOG.info(
                "spectra_ecmwf plotter: no reference_spectra_dir recorded in %s; "
                "falling back to the pre-window-key layout under %s",
                summary_path, reference_dir,
            )
            truth_amp_dir = Path(reference_dir) / "truth" / "spectra"
            input_amp_dir = Path(reference_dir) / "input" / "spectra"
    if recorded_input:
        input_amp_dir = Path(recorded_input)

    for label, candidate in (("truth", truth_amp_dir), ("input", input_amp_dir)):
        if candidate is not None and not candidate.exists():
            LOG.warning(
                "spectra_ecmwf plotter: %s reference not found at %s; "
                "that curve will be missing from the PDF",
                label, candidate,
            )
    truth_label = eval_config.get("truth_label", "truth")
    input_label = eval_config.get("input_label", "input")

    # The standard figure (owner, 9 Nov 2026): all variables in one figure, amplitude spectrum
    # above and amplitude ratio to truth below, truth / input / model.
    try:
        written = build_spectra_grid(
            pred_amp_dir,
            output_dir / "spectra_ecmwf_grid",
            truth_amp_dir=truth_amp_dir,
            input_amp_dir=input_amp_dir,
            truth_label=truth_label,
            input_label=input_label,
            model_label=eval_config.get("model_label", "model"),
            title=eval_config.get("title"),
        )
        LOG.info("spectra_ecmwf: wrote standard grid figure: %s", written)
    except (FileNotFoundError, ValueError) as exc:
        LOG.warning("spectra_ecmwf grid plotter: %s", exc)

    out_pdf = output_dir / "spectra_ecmwf.pdf"
    try:
        n = build_pdf_ecmwf_with_references(
            pred_amp_dir,
            out_pdf,
            truth_amp_dir=truth_amp_dir,
            input_amp_dir=input_amp_dir,
            truth_label=truth_label,
            input_label=input_label,
        )
        LOG.info("spectra_ecmwf: wrote %d-page PDF: %s", n, out_pdf)
    except FileNotFoundError as exc:
        LOG.warning("spectra_ecmwf plotter: %s", exc)

    # Same curves again, but divided by the truth curve, so that departures of
    # a few percent are visible instead of being hidden inside the thickness of
    # a log-log line. Written to its own file so the absolute-amplitude PDF
    # above stays exactly as it was.
    ratio_pdf = output_dir / "spectra_ecmwf_ratio.pdf"
    if truth_amp_dir is None:
        LOG.info(
            "spectra_ecmwf plotter: no truth reference available, so the "
            "ratio-to-truth PDF was not written"
        )
    else:
        try:
            n = build_pdf_ecmwf_ratio(
                pred_amp_dir,
                ratio_pdf,
                truth_amp_dir=truth_amp_dir,
                input_amp_dir=input_amp_dir,
                truth_label=truth_label,
                input_label=input_label,
            )
            LOG.info("spectra_ecmwf: wrote %d-page ratio PDF: %s", n, ratio_pdf)
        except FileNotFoundError as exc:
            LOG.warning("spectra_ecmwf ratio plotter: %s", exc)

    return output_dir
