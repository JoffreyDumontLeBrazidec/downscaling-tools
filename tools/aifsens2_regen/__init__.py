"""Regeneration of AIFS ENS version 2 forecasts for early 2026.

This package rebuilds the ECMWF machine-learned ensemble, public checkpoint
aifs-ens-crps-2.0, from each IFS ensemble member's own perturbed initial
conditions.  The campaign covers 2026-01-01 00 UTC to 2026-05-12 00 UTC, every
twelve hours, ten perturbed members, lead times of six and twelve hours.

The stages are meant to be run in order and each one is safe to re-run:

    retrieve       fetch the initial-condition fields from MARS
    assemble       build one 222-field file per start and member
    run_forecasts  run the model on one GPU, writing the full native output
    regrid         select 68 variables and regrid N320 to O320
    manifest       gather the records for a block and for the campaign
    verify_block   decide PASS or FAIL for a block

calendar.py says which starts belong to which block, and gribspec.py is the one
place where the composition of a file is written down.  See README.md for the
command sequence.
"""

__all__ = [
    "calendar",
    "gribspec",
    "common",
    "retrieve",
    "assemble",
    "run_forecasts",
    "regrid",
    "manifest",
    "verify_block",
]
