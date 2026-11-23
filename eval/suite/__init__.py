"""The classic plot suite: the fixed set of pages the owner gets every time he asks for plots.

Owner, 23 Nov 2026: "define the classic eval.suite ... so I have always the same stuff". The pages,
in this order, are fixed; only the case list (``eval/config/suite/<name>.yaml``) changes per lane:

    01 spectra +24 h       truth (target), input, model(s): eval.plotting.spectra_grid on box_spectra reads
    02 spectra +72 h       same; the standard read (0.07 deg sampling), dotted input and target limits
    01b-02b fine spectra   the same two leads read at the case list's ``spectra.fine_res_deg`` (0.035 deg on the
                           4 km lane), drawn whenever the list sets it. Owner rule, 23 Nov 13:21: BOTH reads, as
                           a rule. The standard read rightly stops near the target limit; the fine read shows the
                           grain below it, which he wants to see (and ideally remove).
    03 TC members          10 members of 10 m wind speed (0 to a top fitted to the box) in a 10-degree box on a
                           storm: input | target | model side by side (zoom_maps)
    04 rain members        10 members of tp on an intense-rain case: input | target | model side by side (zoom_maps)
    05 areas               member 1, input | target | model, one row per area, 10-degree boxes, colours fitted to
                           the box (zoom_maps)
    05z zooms              member 1, input | target | model, one page per area, one row per zoom level (simple
                           smooth maps, eval/suite/simple_maps.py): close level = the tightest allowed, 230 output
                           grid spacings wide (~1000 km at 4.4 km, ~2000 km at 9 km), then x2, x4 (rule 53)
    06 scores              fair CRPS, spread and spread/skill by lead against STATIONS, every headline variable
                           on one page (obs_points summary.csv); the target is a competitor row
    07 global tp           tp stills, input | target | model(s), one row per case (map_video)
    08 tp distribution     tp PDF over all members, model(s) against input and target (precip_dist streaming)
    09 TC distribution     storm central pressure and maximum wind, members of model(s), input, target and
                           IBTrACS at the same cases (tc_structure cases.csv + tc_obs cache)

Map pages (03, 04, 05, 07) show one model, the first --run (owner 23 Nov 09:07: "dont put both 150 and 300, just
300"); ``--maps-all-runs`` adds the others. The input panels are the native input member (rule 41).

Two more pages of the agent's choice for the case are added by hand (``10_choice_*``, ``11_choice_*``),
drawn with eval.cli tools; the suite leaves those slots empty.

Ensembles are one generative draw per input member (rule 36): the suite takes the ``_m`` prediction rows as
they are and never draws several samples from one input member.
"""
