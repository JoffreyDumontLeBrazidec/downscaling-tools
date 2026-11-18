# Final-model battery (`python -m eval.cli final_battery`)

Run ONCE per completely trained model (final EMA-2000 export of a finished run, or a fine-tune at its end step).
Never per step, never as an arm read, never inside `judge` (owner, 18 Nov 2026 13:35 UTC). The command refuses to
start without `--final-model "<name of the finished checkpoint>"`.

It is a realism audit, not a score: three kits from the 17 Nov 2026 investigations of GL512 450k, kept as they were
(only paths, row name, labels and variable lookup by name changed; see the two vendor commits). Specs and reference
numbers (docs master, `epics/fine-scale-o320-o1280/in-progress/`):

| part | ids | note (section) | kit here |
|---|---|---|---|
| terrain | T1-T8, Z1-Z6, D (2d/skt/tcw), B11, maps m01-m11 | `20261117_realism_t2m_terrain.md` s6, `support/20261117_realism_t2m/BATTERY.md` | `kits/terrain/` |
| consistency | C1-C14, member maps | `20261117_realism_consistency_ensemble_physics.md` s6 | `kits/consistency/` |
| wind | W1-W9 (= D1-D9 of the note) | `20261117_realism_wind_cyclones.md` s6 | `kits/wind/` |

## Input

Box member-mode predictions, one draw per input member (rule 31 seeds), `<pred-root>/pred/<row>_L<LLL>/predictions_<date>_step<LLL>.nc`
with `y_pred`, `x_interp_members`, `y`, `x_interp`, `weather_state`, on the 30 dates of 2023 of `ens_member_box_20261104`,
leads 12-72 h. Save ALL model outputs (predictor `SAVE_VARS` with 2d, skt, tcw): the 2d/skt part (T4, D, C14) switches
on only then. Truth = the 10 enfh pf members (`stores/`, `stores_leads/` under `--member-stores`), never the control alone.
Defaults point at Jupiter (`/e/data1/jureap-data/ecmwf/users/jdumont`).

## Run

```bash
python -m eval.cli final_battery plan --final-model "FS-245 100k EMA-2000" --row FS245e2_m \
    --pred-root /e/data1/jureap-data/ecmwf/users/jdumont/outputs/ens_member_box_20261104 --out <OUT>   # prints env + commands
python -m eval.cli final_battery run  --final-model "FS-245 100k EMA-2000" --row FS245e2_m \
    --pred-root ... --out <OUT> [--workers 6] [--consistency-workers 3] [--parts terrain wind] [--no-maps]
python -m eval.cli final_battery summary --out <OUT>     # re-render summary.md from the tables
```

CPU only. 17 Nov costs on the Jupiter login node for 30 dates x 6 leads: terrain 5-12 min (6 workers), consistency
42 min per mask (3 workers, about 5 GB each; the 5-degree mask rerun doubles it, `--no-edge5` skips it), wind about
19 s per date-lead plus about 7 GB of reduced fields per lead (`<OUT>/wind/fields<LLL>/`, deletable after). Heavier
runs go in a CPU job. Steps resume from `<OUT>/.done/`; logs in `<OUT>/logs/`.

## Output

`<OUT>/summary.md`: one table per part, this row beside the GL512 450k number of 17 Nov and the spec's pass line
(`OUT` = outside it). The kit tables stay the record (layout in `layout.py`). `tests/test_final_battery.py` checks that
the summary of the committed 17 Nov tables reproduces the quoted reference numbers (needs the docs checkout:
`DOCS_ROOT`, `~/dev/docs` or a sibling `docs/`); `FB_SYNTHETIC=1` adds a full run on a synthetic copy of the stores.

## Not ported

- W6 (native-grid amplitude ratio against the pf truth, `box_spectra_in3_pf`): its script `amp_spectra.py --truth pf
  --margin 3` lives in the Jupiter gldual read kit, not in a repository. Use `eval.cli spectra` on the row (control
  truth) until it is vendored.
- W9 (XSET-1 hurricane structure) needs an XSET-1 row of the model on Atos: `--xset-rows <root> --xset-cases tc_cases.csv`.
