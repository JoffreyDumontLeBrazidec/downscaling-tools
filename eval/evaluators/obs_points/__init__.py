"""obs_points evaluator: the run's ensemble scored against surface station observations.

For every prediction file it takes the station observations valid at date + lead from an observation
cache (written on Atos by ``eval.cli obs fetch`` from STVL, the database quaver reads), matches each
station to its nearest grid point (dropped beyond ``max_km``), and scores, per variable, lead, region
and date, four rows on the same station sample: ``model`` (the ensemble), and the single fields
``input`` (x_interp), ``truth`` (y) and ``hres`` (the operational HRES at the station, where STVL has
it). The headline scores are ensemble scores: fair CRPS, spread and the spread/skill ratio corrected by
sqrt((M+1)/M), rank-histogram counts and, for tp, the fair threshold-weighted CRPS and the fair Brier
score at each threshold with the observed exceedance counts. The deterministic scores of the ensemble
mean (bias, RMSE, MAE, correlation) and the member-mean MAE are kept as extra columns. A single
field's fair CRPS is its MAE and its spread is zero; every table labels these rows "single field".
2 m temperature is also scored with quaver's lapse-rate height correction (truth and model at the
O1280 orography, input at the O320 orography, HRES at its own). tp is also scored as disc means
around the station's grid point (``radii``). ``plot`` draws the figures from the saved tables
alone, so ``evaluate --plot-only`` replots without reading a prediction again. Two runs are compared
date by date with ``eval.cli obs compare``.

Variables: 2t [K], 10ff [m/s, from 10u/10v], msl [Pa], tp [mm per 12 h ending at valid time; file
metres x 1000, clipped at 0].

Lane configuration (the ``obs_points:`` block); only ``cache`` is required:
  cache          The observation cache directory, or a list or mapping of candidate directories (one
                 per cluster); the first that exists on this machine is used.
  vars           Default [2t, 10ff, msl, tp].
  regions        Mapping name -> [lat0, lat1, lon0, lon1] (lon -180..180, half-open) or null for every
                 station. Default: all, europe (35-75 N, 12.5 W-42.5 E, quaver's), 40-70N, 25-40N, 10-25N.
  height_classes Mapping name -> [lo, hi) in metres of |O1280 orography - station height|, scored as
                 extra regions. Default all_dz<100m [0, 100] and all_dz>=300m [300, 1e9].
  thresholds     tp thresholds in mm per 12 h for twCRPS, Brier and counts. Default [5, 10, 20].
  radii          tp disc radii in km (0 = nearest grid point). Default [0, 18, 36].
  max_km         Largest station-to-grid-point distance kept. Default 15.
  lapse_rate     K/m for the 2t height correction; 0 turns the corrected variant off. Default 0.0065.
  min_stations   A region is scored on a date only with at least this many stations. Default 30.
  weighting      none (plain station mean, the default) or station_density (quaver's Gaussian-kernel
                 station-density weights, eval/tools/obs_crps).
  dates, steps   Restrict the files read (YYYYMMDD strings and lead hours).
  plot_regions   Regions drawn in the score figures. Default [all, europe].
Diagnostic only: no scoreboard row. Point scores at stations penalise realistic but misplaced detail,
and observation error widens the outer bins of the rank histograms.
"""
from eval.evaluators.base import no_score

from .plotter import plot
from .runner import run

score = no_score

EVALUATOR_SPEC = {
    "name": "obs_points",
    "requires": ["predictions"],
    "outputs": [
        "scores_per_date.csv: per var, variant, radius, region, lead, date and row (model, input, truth, hres), "
        "the ensemble scores first (fcrps, spread, spread_rms, ssr_corrected, twcrps<t>, brier<t>, n_obs_gt<t>), "
        "then the deterministic ones (bias, rmse_mean, mae_mean, mae_member, corr_mean).",
        "rank_hist.csv: rank-histogram counts of the observation among the members, per date.",
        "summary.csv: mean and SE over dates of every score; paired.csv: paired differences between rows "
        "(mean, SE over dates, date wins).",
        "summary.md, run_meta.json, obs_points_log.txt.",
        "obs_<var>_<variant>[_r<km>km]_scores.png, obs_tp_tail[_r<km>km].png, obs_rank_hist_<var>_<variant>.png (plot).",
    ],
}

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
