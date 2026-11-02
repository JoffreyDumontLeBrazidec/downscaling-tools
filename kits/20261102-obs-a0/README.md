# A0: station sanity check and tp audits (2 Nov 2026)

Record: docs `epics/fine-scale-o320-o1280/in-progress/20261101_observation_based_eval_inventory_and_plan.md` (sections 4b.3, 4b.4).
Purpose: before any observation evaluator is built, check that the station pipeline gives the expected order on 2t
(O1280 truth beats the O320 input, orography) and audit the 21 Oct tp result (truth level with input at stations).

1. Atos, login node, reads only: `python3 fetch_obs_atos.py <atos_out>` (STVL obs 2t/10ff/msl/tp, HRES at stations,
   O1280/O320 orography heights per station). Copy `<atos_out>/*.csv` to Jupiter (a few tens of MB).
2. Jupiter, CPU: `python score_a0.py <obs dir> /e/data1/jureap-data/ecmwf/users/jdumont/outputs/seedfix_wide_20261030/pred <out>`
   Smoke first with `--dates 2`. Outputs: `a0_summary.md`, `a0_summary.csv`, `a0_paired.csv`, `a0_scores_per_date.csv`,
   `a0_tp_window_check.csv`, `a0_log.txt` (WARN lines must be read).

Pass criteria (pre-registered): 2t lapse-corrected, all stations, truth MAE below input MAE at every lead with
>= 25/30 date wins. If not, the pipeline is suspect and no station score is quoted.
