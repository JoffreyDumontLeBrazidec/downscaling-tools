"""Map videos of downscaled fields (``python -m eval.tools.map_video``).

Global or box maps over lead time, one regular-grid cell per video pixel, interpolated from the
native octahedral grid with ring-wise bilinear weights. Not the retired ``videogen`` (storm
scenes tied to one checkpoint, quarantined 2026-09-29): this tool takes any prediction files.
See ``__main__.py`` for the command line.
"""
