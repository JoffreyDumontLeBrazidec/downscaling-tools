"""Probabilistic spread, CRPS and calibration evaluator (local, from the prediction files).

Computes, from the members in the prediction files, per variable, region (n.hem, tropics,
s.hem, europe) and lead time: the CRPS and Ferro's fair CRPS, the ensemble spread, the
ensemble-mean RMSE, the calibration read of the owner's rule (spread and RMSE as roots of
pooled area-weighted mean squares over points and dates, their ratio, and the ratio times
sqrt((m+1)/m) so a calibrated m-member ensemble reads 1, with a 95 % bootstrap interval of
that corrected ratio over dates (fixed seed, 2000 resamples); rank histograms with m+1 bins,
ties broken at random with a fixed seed, the two outer-bin fractions reported separately;
for precipitation a dry floor of `rank_dry_threshold_mm` (0.1 mm; the files hold metres)
before ranking, so trace members over a dry truth are ties, plus a false-drizzle rate (truth
dry, every member wet) and a missed-rain rate (truth wet, every member dry)), and, with
`scale_split: true`, the same scores for the large part L = P_L(field) and the fine part
F = field - L of members and truth, P_L being one isotropic Gaussian low-pass with its
half-power wavelength at `scale_cut_km` (400 km by default; see core/scale_split.py), and
for the full field on the same inner points those bands use (band `full_inner`, so full,
large and fine are comparable). The bands are NOT additive (P_L is a taper): full_inner is
not large + fine. When the files carry the input members (x_interp), their spread per band
is reported too, so the read can say how much large-scale spread comes from the input.
Precipitation (tp) is in the default variables and is clipped at zero before scoring.

The scores are read with the owner's rule 32 v2 (core/verdict.py, applied by `eval.cli
judge`) on the full, full_inner and fine bands; the large band is reported, not judged (its
scales come from the input). A CRPS win is tested on the centre (ensemble-mean RMSE better)
and on the calibration (the bootstrap interval of the corrected spread/RMSE fully inside
[0.95, 1.05]; fully outside is "outside", crossing an edge is "undecided"): "better model"
needs both; the other outcomes are "better centre, calibration outside 0.05 (closer to /
further from 1 than reference)", "better centre, calibration undecided", "better
calibration, not better centre", "CRPS win, neither centre nor calibration" and "CRPS win,
calibration undecided"; "worse CRPS", "tie" (within 2 paired standard errors over dates, a
harness label) and "n/a" complete the set. Realism and spectra come first; scores never
outrank them. The scale split is off by default because it costs about three times the
full-field scoring (the judge turns it on). This is the local, cheap probabilistic check;
the canonical probabilistic verdict is the `quaver` scorecard. No scoreboard row.
"""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "probabilistic",
    "requires": ["predictions"],
    "outputs": [
        "probabilistic_summary.json: CRPS, spread and ensemble-mean error per variable and region; "
        "the settings (members m, correction factor, P_L cut, tie seed, whether the files carry input members).",
        "scores_by_lead.csv, summary_by_lead.csv: the historical metrics crps, fcrps, spread, rmse_ens_mean "
        "per date and lead, and their mean, std, stderr and n_dates by lead (unchanged names and meanings).",
        "calibration_by_lead.csv: per (step, weather_state, domain, band) the pooled spread_rms, rmse_ens_mean, "
        "spread_skill_ratio, spread_skill_ratio_corrected (= ratio * sqrt((m+1)/m)) with its bootstrap interval "
        "over dates (spread_skill_ratio_corrected_lo95, _hi95, _se; fixed seed), fcrps_mean and stderr, "
        "n_members, n_dates, n_points_total, and spread_input_rms when the files carry input members. Bands: "
        "full (every point of the region minus the box-edge strip), and with scale_split full_inner, large and "
        "fine (the points inside the P_L edge margin). Bands are NOT additive: P_L is a taper, full_inner is not "
        "large + fine.",
        "band_scores_by_lead.csv: the per-date values behind calibration_by_lead.csv (fcrps, crps, spread_rms, "
        "rmse_ens_mean, ms_spread, ms_err, spread_input_rms) per band full / full_inner / large / fine; only full "
        "without scale_split.",
        "rank_histograms.csv: per (step, weather_state, domain) the m+1 bin counts (bin_0..bin_m), the fractions "
        "in bin 0 and in bin m against 1/(m+1) each (and their sum against 2/(m+1)), flatness_rms, rank_mean_norm, "
        "a shape word (U needs both outer bins high; a one-sided pile is 'biased high' / 'biased low'), and for "
        "tp/cp the dry threshold (mm), the false_drizzle_rate and the missed_rain_rate.",
        "plots/probabilistic_scores.pdf: lead-time curves of the historical metrics (written by plot).",
        "plots/rank_histograms.pdf: one page per variable, rank histograms by lead and region (written by plot).",
        "plots/band_scores.pdf: fair CRPS, spread, RMSE and corrected spread/RMSE by band (written by plot, "
        "when scale_split ran).",
    ],
}


__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
