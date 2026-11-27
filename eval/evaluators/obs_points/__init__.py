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
output grid's orography, ``model_height_column``, default O1280; input at its own grid's orography; HRES
at its own). tp is also scored as disc means
around the station's grid point (``radii``). When the cache holds 24 h tp and the SEEPS station
climatology (``obs fetch --tp-periods 12,24 --seeps``), 24 h tp (the sum of two consecutive 12 h files of
the same date, leads 24 h and later) is scored as variable ``tp24`` on the stations that pass quaver's
SEEPS filter (dry-day probability in [0.1, 0.85)), with a ``seeps`` column: SEEPS of every member at
every station (quaver's categories, 0.1 mm rounding and 3 x 3 matrix), averaged over members and
stations; lower is better (quaver stores 1 - SEEPS). ``plot`` draws the figures from the saved tables
alone, so ``evaluate --plot-only`` replots without reading a prediction again. Two runs are compared
date by date with ``eval.cli obs compare``.

Variables: 2t [K], 10ff [m/s, from 10u/10v], msl [Pa], tp [mm per 12 h ending at valid time; file
metres x 1000, clipped at 0], tp24 [mm per 24 h, SEEPS only].

quaver conformity (owner rule, 27 Nov 2026: every station score equals quaver's): observations pass quaver's
toss hard limits (2t 173-333 K, 10ff 0-100 m/s, msl 800-1100 hPa, tp 0-900 mm) and, per row, its
analysis-departure check on the row's own grid; physically implausible reports that pass (tp 301-703 mm in
2023) are kept, as quaver keeps them. fair CRPS per station with M members (Ferro), spread = root of the
station-density-weighted mean of the member variance (ddof 0), ensemble-mean RMSE = root of the weighted mean
square error; density weights over every observation of the valid time; each row on its own stations. Means
over dates as quaver's scorecard (averaging "wmo"): RMS over dates for rmse_mean and spread, plain mean for
fair CRPS and the rest. One departure is kept on purpose: the box-edge strip (``box_edge_deg``, default 1 on
regional grids; 2-3 of ~3,200 e2box stations per date on the 9 km lane, < 0.05 points), which guards against
the slot-head edge artefact; set ``box_edge_deg: 0`` for exact quaver station sets.

Lane configuration (the ``obs_points:`` block); only ``cache`` is required:
  cache          The observation cache directory, or a list or mapping of candidate directories (one
                 per cluster); the first that exists on this machine is used.
  vars           Default [2t, 10ff, msl, tp].
  regions        Mapping name -> [lat0, lat1, lon0, lon1] (lon -180..180, bounds included as in quaver's
                 domains) or null for every station. Default: all, e2box (11-69 N, 79 W-39 E) and europe
                 (35-75 N, 12.5 W-42.5 E), quaver's two card domains, 40-70N, 25-40N, 10-25N.
  height_classes Mapping name -> [lo, hi) in metres of |model orography - station height|, scored as
                 extra regions. Default all_dz<100m [0, 100] and all_dz>=300m [300, 1e9].
  model_height_column
                 Cache column of the output grid's orography at the station, used for the model and truth
                 rows' 2t height correction and for the height classes. Default h_o1280 (9 km output); the 4 km
                 lane sets h_o2560 (``obs fetch --add-height o2560=<orography> --out <cache>``). A column the
                 cache lacks is an error.
  tp_obs_var     The cache series the model tp is scored against: tp (12 h, the default) or tp6 (6 h windows
                 ending at valid time, the 4 km lane); anything but tp turns SEEPS and the 12 h window audit off.
  thresholds     tp thresholds in mm per 12 h for twCRPS, Brier and counts. Default [5, 10, 20].
  radii          tp disc radii in km (0 = nearest grid point). Default [0, 18, 36].
  max_km         Largest station-to-grid-point distance kept for the model and truth rows. Default 15.
  input_max_km   The same for the input row on its native grid. Default: unlimited on a global grid, the grid's
                 coarsest point spacing on a regional one.
  box_edge_deg   On a regional grid, drop stations whose nearest grid point is within this many degrees of the
                 box edge (slot-head edge artefact, docs ablation note s21). Default 1; 0 keeps them; global grids keep all.
  lapse_rate     K/m for the 2t height correction; 0 turns the corrected variant off. Default 0.0065.
  min_stations   A region is scored on a date only with at least this many stations. Default 30.
  density_population  all (default: the density kernel runs over every observation of the valid time in the
                 cache that passes the row's checks, scored or not, as quaver does; exact only with a global
                 cache) or scored (only the scored stations, the pre-26 Nov 2026 behaviour).
  weighting      station_density (the default since 27 Nov 2026: quaver's Gaussian-kernel station-density
                 weights, eval/tools/obs_crps) or none (plain station mean).
  station_set    per_row (default, quaver): each row on its own stations, those where it has a value and whose
                 observation passes quaver's analysis-departure check on that row's own grid; common: the
                 intersection over rows (the pre-27 Nov 2026 behaviour).
  analysis_columns  Row -> cache column of the analysis of that row's grid. Default model/truth an (O1280; an_o2560
                 when model_height_column is h_o2560), input an_o320 when input_height_column is h_o320*, hres an.
                 A column the cache lacks means no analysis check for that row (caches before 27 Nov 2026).
  seeps          Score 24 h SEEPS when the cache allows it. Default true.
  competitor_ensembles
                 Score the input and the truth as ensembles when the files carry their members
                 (``x_interp_members`` / ``y_members``, or a member axis on ``x_interp`` / ``y`` whose members
                 differ, as the ENS-member files of 4 Nov 2026 do). Default true. Otherwise, and with false,
                 they are the control fields, labelled "single field": their fair CRPS is the MAE of one field
                 and is compared with the ensemble mean only, never with an ensemble's fair CRPS (rule 34(e)).
  dates, steps   Restrict the files read (YYYYMMDD strings and lead hours).
  plot_regions   Regions drawn in the score figures. Default [all, europe].
Diagnostic only: no scoreboard row. Point scores at stations penalise realistic but misplaced detail,
and observation error widens the outer bins of the rank histograms.

Provenance (audit 5 Nov 2026, rule 34(e)): ``run_meta.json`` carries a ``provenance`` block (verifying field,
how the model ensemble was built from what the file says: one draw per input member, or generative draws from
one input, which is a single-input diagnostic and not an ensemble; the kind of every competitor row; weight
form; spread form; region bounds; date spacing) and ``summary.md`` prints it above the tables.
``eval.cli evaluate`` collects the blocks of a run into ``evaluators/provenance.md``.
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
        "the ensemble scores first (fcrps, spread, spread_rms, ssr_corrected, twcrps<t>, brier<t>, n_obs_gt<t>, "
        "seeps for tp24), "
        "then the deterministic ones (bias, rmse_mean, mae_mean, mae_member, corr_mean).",
        "rank_hist.csv: rank-histogram counts of the observation among the members, per date.",
        "summary.csv: mean and SE over dates of every score; paired.csv: paired differences between rows "
        "(mean, SE over dates, date wins).",
        "tp_window_check.csv / tp_window_summary.csv: the A0 accumulation-window audit (correlation of truth, input "
        "and ensemble mean with the station tp of the window ending at valid time and of the window before; a WARN "
        "line when the window before correlates better on average at a lead).",
        "summary.md, run_meta.json (WARN lines, also for HRES tp that looks like metres), obs_points_log.txt.",
        "obs_<var>_<variant>[_r<km>km]_scores.png, obs_tp_tail[_r<km>km].png, obs_rank_hist_<var>_<variant>.png (plot).",
    ],
}

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
