"""Observation CRPS tool. Its evaluator, obs_crps, was retired on 2026-09-28; the code stays
because tools/station_head refers to it.

The compute itself lives in obs_crps_compute.py and is executed as a standalone
script under the ECMWF module environment (vtb for the observation database,
mars for the fields), not imported into the eval venv.
"""

from .plotting import plot_obs_crps_summary

__all__ = ["plot_obs_crps_summary"]
