"""Local ML-vs-ENFO ensemble-spread comparison (spread proxy)."""

from .plotting import plot_all, plot_spread_curves, plot_spread_maps, plot_spread_spectra
from .scoring import compute_spread_proxy

__all__ = [
    "compute_spread_proxy",
    "plot_all",
    "plot_spread_curves",
    "plot_spread_maps",
    "plot_spread_spectra",
]
