"""tc_obs evaluator: the run's tropical cyclones against the IBTrACS best track and the operational trackers.

It reads ``tc_structure``'s per-case table (``cases.csv``: centre, central pressure and winds of every
model member, of the truth and of the input, per event, date and lead) and, for every storm case
(``storm_ok`` = 1, the truth-based first guess of tc_structure), compares them at valid time with the
IBTrACS best-track position and central pressure (linear interpolation between 6-hourly fixes when the
valid time falls between two) and with the ECMWF operational tracker at the same valid time: the HRES
track (a single field) and the ENS tracks (51 members, an ensemble reference). The headline is ensemble
scores: fair CRPS of the members' central pressures against the best-track pressure, the members' mean
position error and the spread of their centres. Truth, input and HRES are single-value references.
With few storms, every output states how many cases on how many dates it rests on: they are cases,
not a verdict. ``plot`` draws the figures from the saved tables alone.

The reference cache comes from ``eval.cli obs fetch-tc`` on Atos (IBTrACS rows of the configured storms,
the decoded tracker files, tc_meta.json); copy it to the cluster that holds the run. The anemoi IBTrACS
zarr has no wind; a NOAA IBTrACS CSV, when one is configured at fetch time, adds the best-track wind and
then ``wind_crps`` / ``wind_err`` compare ``wind_col`` with it (the definitions differ: grid 10 m wind
against a sustained-wind estimate).

Lane configuration (the ``tc_obs:`` block):
  cache          The tc_obs reference cache directory, or a list or mapping of candidates (first existing).
  cases          tc_structure's cases.csv; ``{run_label}`` is replaced by the run label. Default
                 <run>/evaluators/tc_structure/cases.csv.
  only_storm_ok  Score only tc_structure's storm cases. Default true.
  max_gap_h      Largest gap between two best-track fixes to interpolate across. Default 6.
  wind_col       tc_structure column compared with the best-track wind when there is one. Default maxwind300_ms.
Diagnostic only: no scoreboard row.
"""
from eval.evaluators.base import no_score

from .plotter import plot
from .runner import run

score = no_score

EVALUATOR_SPEC = {
    "name": "tc_obs",
    "requires": ["predictions"],
    "outputs": [
        "cases_obs.csv: per storm case and row (model, ens, truth, input, hres) the best track and the scores "
        "(pmin_crps, pmin_err, pmin_mae, pos_err_member_km, pos_err_mean_km, pos_spread_km).",
        "members_obs.csv: per member of model and ENS, the pressure and position errors.",
        "summary.csv (mean and SE over cases by lead and pooled), paired.csv (model minus each reference, per case).",
        "summary.md, run_meta.json (states the number of cases and dates).",
        "tc_obs_by_lead.png, tc_obs_cases.png (plot).",
    ],
}

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
