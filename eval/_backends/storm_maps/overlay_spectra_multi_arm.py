"""RETIRED 2026-07-13 — bespoke multi-arm box-FFT spectra overlay.

Consolidation call (owner): keep exactly ONE regional per-box spectra path (BoxSpectra in
render.py) plus the two eval.cli global evaluators — `spectra` (HEALPix fast proxy) and
`spectra_ecmwf` (ECMWF GRIB). This standalone multi-arm overlay produced the runtime-confounded
b785 "3.5-4.2x over-power" figures (unified-runtime grid-scale noise floor read through a
20-100 km near-Nyquist band); it is retired to avoid a third, easily-misused spectra path.

For per-box spectra use eval/_backends/storm_maps/render.py (BoxSpectra), PRISTINE-only,
40-150 km band. Historical outputs preserved at ~/scratch/eval/testbed_validation_plots and
~/perm/eval-rescue-20260709/. See tc-o320-o1280 20260711_lane_soundness_audit.md CORRECTION.
"""
import sys

sys.exit(
    "overlay_spectra_multi_arm.py is RETIRED (2026-07-13). Use render.py BoxSpectra "
    "(regional, pristine-only, 40-150 km band) or the `spectra` / `spectra_ecmwf` evaluators."
)
