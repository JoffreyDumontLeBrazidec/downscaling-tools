"""Multi-row aggregator of the few-step big eval (o96->o320, 15 Oct 2026).

One module per protocol part: ``structure`` (1), ``spectra`` (2), ``extremes`` (3),
``scores`` (4, with the paired per-date test and the ``fcrps`` fallback), ``costs`` (5),
and ``summary`` (the one-page summary). ``common`` holds the row file, the result-directory
discovery and the table writers. The command-line entry is ``../aggregate.py``.
"""
