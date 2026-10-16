"""Temporal consistency of one member across its lead times (TD-227).

The lane downscales every lead on its own. This read measures whether the generated part of
the o1280 field flickers between consecutive leads, band by band, against the truth and the
interpolated input, from ``predictions_<date>_step<SSS>.nc`` files that hold consecutive
steps of the same dates and members.

Instruments (docs ``epics/research-ideas/in-progress/20261016_temporal_consistency_methods_survey.md`` s2):

1. per-band lead-to-lead difference: the area-weighted mean of |f(L+dt) - f(L)| of the field
   band-passed to 31-56, 56-125 and >125 km, for ``y_pred``, ``y`` (truth) and ``x_interp``
   (input), per variable and member, pooled over dates and lead pairs (the metric of Andrae et
   al. 2410.05431, made per band);
2. per-band lag-1 correlation between lead L and L+dt of the band-passed field (EnScale's
   autocorrelation, 2509.26258);
3. storm continuity: the truth's msl minimum is found in a search box at every step and the
   model's and input's minimum msl and maximum 10 m wind are read within a radius of it; the
   statistic is the standard deviation of the lead-to-lead change, model over truth.

Band-pass: linear sampling of the points on a regular 0.07-degree grid (the rule-14 sampler),
mean removed, FFT without taper, mask on the isotropic wavelength band, inverse FFT; statistics
on the interior (a margin of ``edge_frac`` of each side dropped) with cos(lat) weights.
The same operator is applied to prediction, truth and input, so ratios are fair.

Outputs: ``bands.csv`` (one row per run, variable, band, quantity), ``storms.csv``,
``read.md`` (ratios model/truth and, with two runs, run/run).
"""
