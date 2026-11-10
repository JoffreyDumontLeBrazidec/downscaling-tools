"""obs_imerg evaluator: the run's 12 h precipitation against IMERG Final V07B, point by point on O1280.

IMERG is a gridded, satellite-based precipitation estimate, used here as a third reference next to the
truth (y) and the stations of obs_points. The IMERG store is on our O1280 grid in our point order, so
there is no interpolation: ``eval.cli obs fetch-imerg`` (Atos) cuts the E2 points for the needed valid
times into one small netCDF file (12 h = the two 6 h IMERG steps ending at valid time, in mm), which is
copied next to the predictions. For every prediction file and region it scores, on the same points, the
model ensemble and the single fields input and truth against IMERG: fair CRPS, spread and corrected
spread/skill, rank-histogram counts, fair twCRPS and fair Brier at each threshold with the IMERG
exceedance counts (``n_obs_gt<t>``), and the deterministic scores of the ensemble mean as extra columns.
It also keeps the tp histograms of IMERG, truth, input and all members (precip_read's 0.05-dex bins)
for PDFs. The tables have the obs_points layout, so ``eval.cli obs compare`` pairs two runs by date.

Known IMERG caveats: less drizzle and a heavier tail than the IFS, weak poleward of 60 N and over snow.
The default regions therefore stay within 25-60 N.

Lane configuration (the ``obs_imerg:`` block):
  imerg        The IMERG cut file, or a list or mapping of candidate paths (the first existing is used).
  regions      Mapping name -> [lat0, lat1, lon0, lon1], inclusive. Default 25-60N, europe (35-60 N,
               12.5 W-40 E), 25-40N, 40-60N.
  thresholds   mm per 12 h. Default [5, 10, 20].
  min_points   A region is scored on a date only with at least this many valid IMERG points. Default 1000.
  dates, steps Restrict the files read.
  plot_regions Regions drawn. Default the first two regions.
  native_input   rule 41: the input row from the native x (nearest native point of each IMERG point), never
                 x_interp when the file has x. Default true; false reproduces old tables.
  competitor_ensembles
               As in obs_points: the input and truth rows are ensembles when the files carry their members
               (default true); otherwise single fields, labelled so (rule 34(e)).
Diagnostic only: no scoreboard row. ``run_meta.json`` carries the provenance block (see obs_points).
"""
from eval.evaluators.base import no_score

from .plotter import plot
from .runner import run

score = no_score

EVALUATOR_SPEC = {
    "name": "obs_imerg",
    "requires": ["predictions"],
    "outputs": [
        "scores_per_date.csv, rank_hist.csv, summary.csv, paired.csv, summary.md: obs_points layout, variant 'imerg', "
        "rows model (ensemble), input and truth (single fields).",
        "hist.npz: tp counts per lead, region and row (imerg, truth, input, model) in 0.05-dex bins.",
        "run_meta.json.",
        "obs_tp_imerg_scores.png, obs_imerg_tail_<region>.png, obs_imerg_rank_hist_<region>.png, obs_imerg_tp_pdfs.png (plot).",
    ],
}

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
