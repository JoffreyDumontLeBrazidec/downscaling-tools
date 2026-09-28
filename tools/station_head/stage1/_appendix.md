
## A step down in the network from 25 August 2026

The number of reporting stations is not quite constant across the window, and the
departure from constancy is not noise. From the start of the window until 25 August
2026 at 12 UTC the count of 2 m temperature reports sits at about 16,000 at every
valid time. From 25 August at 18 UTC to the end of the window it sits at about
13,300, a step down of roughly 2,800 stations, that is about 17 per cent, and it
never recovers. The same step is visible in every parameter. There are also two
shorter dips of about the same depth, from 27 to 29 June and from 8 to 9 July, after
which the count returns to its usual level.

This matters for the head because the validation split fixed in section 4 of the
design note is the initialisations of 18 to 31 August 2026, so a little over half of
the validation period falls after the step while the whole training period falls
before it. Whether that is a real change in what the database holds or a matter of
observations still arriving is not something this build can tell from the counts
alone; it should be re-checked before any validation number from that period is
believed, and re-running the retrieval for the last week of August at a later date
would settle it.

Separately, the final valid time of the window, 1 September 2026 at 00 UTC, is
markedly thinner than every other: 7,652 reports for 2 m temperature against a
median of 15,947, and 3,667 for precipitation against a median of 11,289. It is the
last valid time of the calendar and it is the only one that is halved, so any
per-valid-time statistic should treat it as an outlier rather than as a typical case.

## The analysis at the station, section 5 of the design note

The optional section 5 work was done in full for the planned one-in-eight subsample.
For every eighth synthetic index, that is 56 of the 446 valid times, the 2 m
temperature, 2 m dewpoint, 10 m wind components and mean sea level pressure were read
out of the built target store, which is the O1280 analysis at the valid time, and the
value at each station nearest O1280 point was attached to the observation as the
column `analysis_value`. The 10 m wind speed was formed from the two components the
same way quaver forms it. The result is
`/home/ecm5702/agent-work/20260909-station-head-adapter/outputs/analysis_at_stations_subsample.parquet`,
2,687,152 rows over 56 valid times.

The job was far cheaper than the design note feared. The five variables the head
needs are all inside the first 34-variable chunk of the store, so one chunk read of
about 0.9 GB serves a valid time rather than the two the note assumed, and the whole
subsample took about 80 seconds of wall time rather than hours. Extending this to all
446 valid times would cost roughly ten minutes and is worth doing in stage 2.

The departure statistics below are the observation minus the analysis at the nearest
point, with no quality control applied and no lapse-rate correction, which is exactly
the raw quantity the design note asked to record. Temperature and dewpoint sit within
a few tenths of a kelvin of the analysis in the median and within about 3.5 K at the
fifth and ninety-fifth percentiles, which is the representativeness spread the head is
meant to reduce. The wind speed observation is biased low against the analysis by
about a third of a metre per second, which is the usual sign of stations sheltered
relative to the model. The pressure departure is in pascals.

| parameter | rows | median departure | 5th percentile | 95th percentile |
|---|---|---|---|---|
| 2t (K) | 878,272 | +0.075 | -3.172 | +3.529 |
| 2d (K) | 600,886 | +0.084 | -3.373 | +3.534 |
| 10ff (m/s) | 685,169 | -0.324 | -3.016 | +2.832 |
| msl (Pa) | 522,825 | -12.625 | -346.550 | +188.188 |
