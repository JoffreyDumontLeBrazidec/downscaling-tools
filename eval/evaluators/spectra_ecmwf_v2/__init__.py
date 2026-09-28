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

`spectra_ecmwf` is kept unchanged beside this as the reference implementation,
so the two can be run against each other.
"""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "spectra_ecmwf_v2",
    "default_enabled": False,
    "scoreboard": False,
    "requires": ["predictions"],
    "host_constraint": "ac",
}
