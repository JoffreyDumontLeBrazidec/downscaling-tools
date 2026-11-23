"""The classic plot suite: the fixed set of pages the owner gets every time he asks for plots.

Owner, 23 Nov 2026: "define the classic eval.suite ... so I have always the same stuff". The pages,
in this order, are fixed; only the case list (``eval/config/suite/<name>.yaml``) changes per lane:

    01 spectra +24 h       truth (target), input, model(s): eval.plotting.spectra_grid on box_spectra reads
    02 spectra +72 h       same; exactly two spectra pages, no more
    03 TC members          10 members of 10 m wind speed around a storm: input / target / model rows (zoom_maps)
    04 rain members        10 members of tp on an intense-rain case: input / target / model rows (zoom_maps)
    05 areas               member 1, input | target | model(s), one row per area (zoom_maps)
    06 scores              fair CRPS, spread and spread/skill by lead against STATIONS, every headline variable
                           on one page (obs_points summary.csv); the target is a competitor row
    07 global tp           tp stills, input | target | model(s), one row per case (map_video)
    08 tp distribution     tp PDF over all members, model(s) against input and target (precip_dist streaming)
    09 TC distribution     storm central pressure and maximum wind, members of model(s), input, target and
                           IBTrACS at the same cases (tc_structure cases.csv + tc_obs cache)

Two more pages of the agent's choice for the case are added by hand (``10_choice_*``, ``11_choice_*``),
drawn with eval.cli tools; the suite leaves those slots empty.

Ensembles are one generative draw per input member (rule 36): the suite takes the ``_m`` prediction rows as
they are and never draws several samples from one input member.
"""
