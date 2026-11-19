# MI-S25: month inference set, September 2025 (17 Nov 2026)

One input dataset for global 36 -> 9 km inference without prepml: 30 start dates (1-30 Sep 2025, 00 UTC) x 20 leads
(+12..+240 h every 12 h) x 10 members (od eefo pf 1-10, O320, the 70 GL512 input variables). Same pipeline as XSET-1
(`kits/20261116-extremes-set`, gated bit for bit on 16 Nov), longer lead list, full-field output.

Atos (work dir `/ec/res4/scratch/ecm5702/agent-work/20261117-month-inference`, kit checkout in `downscaling-tools/`):
1. `mi_probe.sh` read-only: MARS list of eefo steps 12-240, quotas, /ec/ai writability.
2. `mi_template.sh` once: O1280 GRIB key template (od enfo pf 1, step 12).
3. `bash mi_chain_submit.sh dates_s25.txt 2 ret` -> `mi_retrieve.sbatch` (MARS -> grib/, counts checked) -> `mi_build_in.sbatch`
   (`stores/in/<date>.zarr`, statistical check, `stores/check/in_<date>.done`).
4. `mi_measure.sh` sizes and times; `mi_copy_to_jupiter.sh` (on ac6-100) -> Jupiter
   `/e/data1/jureap-data/ecmwf/users/jdumont/datasets/mi_s25_eefo_o320_pf10_l012-240/in/`.

Jupiter (kit checkout in `/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261117-month-inference/downscaling-tools`):
5. `sbatch jupiter/mi_xcheck.sbatch` gate: month path = XSET-1 GL512 500k row bit for bit, packing within bound.
6. `sbatch --job-name=mi_GL512_450k_<i> jupiter/mi_predict.sbatch GL512_450k <dates...>` (3 dates per job, ~6 h on 4 GH200).
   Output `outputs/month_inference_s25/rows/<ROW>/<date>/pred_<date>_step<LLL>.npz` (uint16 packed, all 10 outputs) + hist/.
   Seed base 1234 = every XSET-1 / S25 paper row (rule 31 pairing); prepml jdyb uses 756 and is not paired.

Back on Atos: `mi_encode_grib.py --rowdir ... --date ... --template grib/tmpl_o1280.grib --out ... --expver <label>` -> GRIB for
quaver source="marsfile" (scorecards kit `q_compute_ctl.py`). Never FDB (rule 30).
`grib/in_sfc_pf_<date>.grib` (native O320 input, 10 sfc params, steps 12-240) is kept: it is the input row for quaver.

## Sets (19 Nov 2026)
MI-S25 = Sep 2025 (`dates_s25.txt`), MI-J25 = Jan 2025 (`dates_j25.txt`, 31 dates, owner pick 19 Nov 2026: other season, Eowyn/Herminia).
`mi_set.sh` maps a date to its set; Jupiter input `datasets/mi_<set>_eefo_o320_pf10_l012-240/in`, output `outputs/month_inference_<set>/rows/<ROW>/`.
`mi_predict.sbatch` takes the set from its dates (one set per job); the xcheck gate file stays in the S25 folder. Atos work dir is shared (dates differ).
Probe: `bash mi_probe.sh 20250101 20250131`. Chain: `bash mi_chain_submit.sh dates_j25.txt 2 ret` with SBATCH_ACCOUNT=ecaifs (nf limit 100/account).
