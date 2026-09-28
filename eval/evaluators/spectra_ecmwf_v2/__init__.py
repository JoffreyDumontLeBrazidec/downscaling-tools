"""ECMWF spectra evaluator, version two — gptosp pipeline (AC-only).

Only one difference changes the numbers: fields are staged onto the complete
grid, rather than onto templates with 28 latitude rows dropped near the poles.
Measured 2026-08-25 on a real O1280 field at a fixed truncation, that pole mask
shifts the spectrum by 2.0% at the median and 13.2% at worst, inside the scored
band above wavenumber 100.

The rest are internal and were each verified to leave the numbers alone. The
truncation is passed to gptosp explicitly with -T and read back off every file
it produces, which is bitwise identical to the old -l derivation over the
retained coefficients. Amplitudes come from the GRIB coefficients via eccodes
rather than Metview, which agrees to about 5e-12 across all four lanes and all
fields, over their full range. Curve names are written in the six-field form the
scoreboard actually reads. The reference cache is addressed by evaluation window
and staging template, so a run cannot silently reuse a reference computed for a
different month or a different grid.

Since 2026-09-28 this is the only spectra evaluator: the HEALPix proxy
(`spectra`) and version one (`spectra_ecmwf`) are retired. Its scorer compares
the run's mean curves with the truth reference above wavenumber 100 and emits
scoreboard rows named `spectra_v2_<variable>_*` and `spectra_v2_mean_*`, which are
deliberately not the proxy's `spectra_*` names (see scorer.py).
"""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "spectra_ecmwf_v2",
    "requires": ["predictions"],
    "outputs": [
        "spectra_summary.json, staging_summary.json, spectra_curve_summary.json: what was staged and the mean curves.",
        "spectra/: the mean spectrum curves per variable.",
        "spectra_v2_scores.json: per-variable relative L2 detail (written by score).",
        "spectra_ecmwf.pdf and spectra_ecmwf_ratio.pdf: spectra and model-to-truth ratio figures (written by plot).",
    ],
}


__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
