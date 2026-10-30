"""Probabilistic ensemble scoring: the computation behind the probabilistic evaluator."""

from .scoring import compute_probabilistic_scores, crps_ensemble_components
from .calibration import (
    PooledMoments,
    correction_factor,
    dry_floor,
    dry_wet_disagreement,
    histogram_shape,
    rank_counts,
    spread_skill_ratio,
    summarize_histogram,
    truth_ranks,
)
from .scale_split import LowPassOperator, inner_mask, sigma_km_for_cut
from .verdict import calibration_state, paired_difference, verdict_label
from .plotting import plot_probabilistic_summary, plot_rank_histograms, plot_band_scores

__all__ = [
    "compute_probabilistic_scores",
    "crps_ensemble_components",
    "PooledMoments",
    "correction_factor",
    "dry_floor",
    "dry_wet_disagreement",
    "histogram_shape",
    "rank_counts",
    "spread_skill_ratio",
    "summarize_histogram",
    "truth_ranks",
    "LowPassOperator",
    "inner_mask",
    "sigma_km_for_cut",
    "calibration_state",
    "paired_difference",
    "verdict_label",
    "plot_probabilistic_summary",
    "plot_rank_histograms",
    "plot_band_scores",
]
