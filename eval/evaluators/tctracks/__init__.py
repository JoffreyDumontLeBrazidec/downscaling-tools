"""Track-based TC comparison layer over tctracker outputs (eval.cli tccompare).

Distinct from ``eval/evaluators/tc`` (box-based raw extremes on prediction
NetCDFs — the scoreboard/verdict instrument). This package compares TRACKS
from multiple sources (model expver, ctrl expver, target ENFO, input EEFO)
that were all produced by ``eval.cli tctracker`` on ONE tracking support.
Diagnostic panel only: no composite score, no scoreboard ingestion.
"""
