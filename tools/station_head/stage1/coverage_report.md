# Stage 1 coverage report: station observations for the 2026 AIFS ENS version 2 calendar

This report describes what the stage 1 build of the station head adapter actually
retrieved. The calendar is the 446 synthetic samples of the AIFS ENS version 2
validtime store, which are 223 initialisations at 00 and 12 UTC from 12 May 2026
12 UTC to 31 August 2026 12 UTC, each with a 6 and a 12 hour lead. The valid times
run from 12 May 2026 18 UTC to 1 September 2026 00 UTC, exactly six hours apart,
with no gap and no repetition, so the pairing manifest is a clean one to one map
between a store index and a valid time.

## Station counts per parameter

The table gives, over the 446 valid times, the median, smallest and largest number
of stations that reported the parameter, and how many distinct stations were seen at
least once.

| parameter | period (h) | median | min | max | distinct stations | total rows |
|---|---|---|---|---|---|---|
| 2t | 0 | 15947 | 7652 | 16724 | 17973 | 6999598 |
| 2d | 0 | 10745 | 7044 | 11307 | 12318 | 4805193 |
| 10ff | 0 | 12311 | 7011 | 13054 | 15259 | 5469584 |
| msl | 0 | 9327 | 5886 | 9838 | 13263 | 4137835 |
| tp | 6 | 11289 | 3667 | 12144 | 13940 | 5006239 |

The valid times with the fewest reports, for each parameter, are listed below. A low
count usually means a late-arriving observation batch rather than a failed request.

- 2t: 2026-09-01 00Z (7652), 2026-06-28 12Z (12536), 2026-07-09 00Z (12582), 2026-06-28 00Z (12763), 2026-06-28 18Z (12770)
- 2d: 2026-09-01 00Z (7044), 2026-05-24 12Z (9947), 2026-08-24 00Z (9987), 2026-07-24 00Z (10072), 2026-05-25 00Z (10133)
- 10ff: 2026-09-01 00Z (7011), 2026-06-28 12Z (10381), 2026-06-28 18Z (10665), 2026-06-28 00Z (10711), 2026-06-29 00Z (10769)
- msl: 2026-09-01 00Z (5886), 2026-07-24 00Z (8215), 2026-08-24 00Z (8240), 2026-05-25 00Z (8587), 2026-07-25 00Z (8645)
- tp: 2026-09-01 00Z (3667), 2026-06-28 12Z (8518), 2026-06-28 18Z (8852), 2026-06-28 00Z (9683), 2026-07-09 00Z (9910)

## Stability of the network

For 2t, 17973 distinct stations were seen at least once and 11469 of them, that is 63.8 per cent, reported at more than 90 per cent of the 446 valid times. The median station reported at 98.4 per cent of valid times.

For tp, 13940 distinct stations were seen at least once and 7745 of them, that is 55.6 per cent, reported at more than 90 per cent of the 446 valid times. The median station reported at 94.4 per cent of valid times.

Across all five parameters, 23138 distinct station identifiers were seen. This is the row count of the static station table.

## Stations, terrain and the hold-out split

Every station was matched to its nearest point of the O1280 output grid and given the
static fields the analysis archive carries there: the sub-grid orography standard
deviation, the slope, the surface height and the land-sea fraction within 15 km. The
terrain class follows the design note, mountain above 100 m of sub-grid orography,
hilly between 30 and 100 m, flat below. The hold-out flag was assigned by the
following method: hash: sha1(stnid) modulo 100 below 5.

| stratum | stations | held out | held out (%) |
|---|---|---|---|
| terrain_class = flat | 11383 | 604 | 5.31 |
| terrain_class = hilly | 6030 | 293 | 4.86 |
| terrain_class = mountain | 5725 | 309 | 5.40 |
| region = europe | 10177 | 542 | 5.33 |
| region = n.hem.other | 7904 | 404 | 5.11 |
| region = s.hem | 1608 | 80 | 4.98 |
| region = tropics | 3449 | 180 | 5.22 |
| all | 23138 | 1206 | 5.21 |

The nearest-point distance has a median of 3.60 km and a maximum of 6.8 km. The station minus model height difference has a median of -14.5 m overall and -178.3 m in the mountain class, which is the representativeness gap the head is meant to close.

Coastal stations, meaning those whose 15 km neighbourhood mixes land and sea, number 7135, which is 30.8 per cent of the network.

## Failed and empty requests

No request failed and no request returned an empty result. All 2230 requests, that is five parameters at each of the 446 valid times, returned stations.

## Method and wall time

- retrieval method: SLURM batch job on the nf queue, job 34850096, one process, module environment python3 + vtb + ecmwf-toolbox; vtb works on a batch node, so no login-node fallback was needed
- retrieval wall time: 27 minutes 23 seconds for 2230 requests, that is 5 parameters at each of 446 valid times, about 0.73 s per request
- static station table: SLURM batch job 34859566 on the nf queue, 22 seconds
- analysis at stations (section 5): SLURM batch job 34859882 on the nf queue, one in eight subsample



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
