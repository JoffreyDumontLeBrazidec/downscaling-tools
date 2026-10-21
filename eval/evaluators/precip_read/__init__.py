"""precip_read evaluator: the total-precipitation read of a run, in one place, replottable.

For every lead time it streams the prediction files once and keeps, per date and per region,
the tp histogram of the truth, the interpolated input and every member; the sums behind bias,
normalised RMSE, correlation and fair CRPS of the ensemble against the truth (the input scored
the same way as a one-member forecast); and the counts above the heavy-rain thresholds. It
also cuts the configured case boxes (truth, input, all members) for maps. ``plot`` draws the
figures from those files alone, so ``evaluate --plot-only`` replots without reading a
prediction again. Per-bin tp spectra stay in ``box_spectra``.

Lane configuration, all keys optional (the ``precip_read:`` block):
  var          The precipitation channel. Default "tp".
  accum_hours  Accumulation window written on the axes. Default 6.
  dates, steps Restrict the files read (lists of YYYYMMDD strings and lead hours).
  regions      Mapping name -> [lat0, lat1, lon0, lon1] (lon in -180..180). Default: the whole
               file, 10-25N, 25-40N, 40-70N and europe (35-70 N, 12 W-40 E).
  thresholds   Heavy-rain thresholds in mm. Default [20, 50, 100, 150, 200].
  cases        List of {name, label, date, step, box: [lat0, lat1, lon0, lon1]} for maps.
  map_groups   Mapping figure name -> list of case names, one map figure per group.
  map_members  Members drawn on the maps (0-based). Default [0, 1, 2]: the first three, not chosen.
  bootstrap    Bootstrap draws over dates for the bands. Default 2000.
Diagnostic only: no scoreboard row.
"""
from eval.evaluators.base import no_score

from .plotter import plot
from .runner import run

score = no_score

EVALUATOR_SPEC = {
    "name": "precip_read",
    "requires": ["predictions"],
    "outputs": [
        "hist_step<NNN>.npz: per date, member and region, tp counts in 0.05-dex bins for truth, input and model.",
        "scores_rows.csv: per lead, date and region, the sums behind bias, normalised RMSE, correlation, "
        "fair CRPS and spread, and the counts above each threshold.",
        "cases/<name>.npz: truth, input and every member on the case box.",
        "run_meta.json: files, members, regions, thresholds.",
        "tp_pdf_step<NNN>.png, tp_scores_by_lead.png, tp_heavy_area_by_lead.png, tp_maps_<group>.png (plot).",
    ],
}

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
