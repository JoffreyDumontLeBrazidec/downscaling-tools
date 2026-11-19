# MI4-S26: 4 km lane month inference input set, September 2026 (full globe)

Input-only store for running the global o1280 -> o2560 model (G150e2, sandbox db_o2560_global_chain_20261116, code
exp/o2560-global-chain-20261116 982ebb3b) on one month, and scoring at stations against the input ensemble (quaver, prepml style).
No 4 km truth (owner 19 Nov 2026). Record: docs note epics/o1280-o2560-lane-defects/in-progress/
20261027_ca300_finetune_o1280_o2560_smoke.md, sections 22, 22.1, 22.2.

- Dates: 20260901..20260930 (dates_s26.txt), 00 UTC; od enfo pf 1-10; leads 12..240 h every 6 h (39); full O1280 globe.
- Variables: the 37 in_lres variables of Jupiter datasets/o1280_o2560_global/in_lres_o1280.zarr (order, points and statistics
  read from its metadata copy ref/in_lres_o1280_meta.zarr), plus tp/cp 6 h windows in array `tp`.
- Recipe = s20 member store builder (20261109-o2560-tc): every MARS request at grid=O1280 (pl t/u/v/w/z spectral -> MIR).
- Atos: /ec/res4/scratch/ecm5702/agent-work/20261119-mi4k-s26 (settings in mi4_set.sh, never env). Jupiter:
  /e/data1/jureap-data/ecmwf/users/jdumont/datasets/mi4_s26_enfo_o1280_pf10_l012-240/{in,check}.
- Kept native GRIB per date in keep/: sfq_<d>.grib (10u 10v 2d 2t msl, steps 12..240/6) and tp_<d>.grib (tp cp 6..240/6).

Files: mi4_retrieve_date.sh (+ .sbatch), mi4_build_in.py (+ mi4_build.sbatch), mi4_check_s20.py (+ .sbatch: processing
reproduces the s20 box rows bit for bit), mi4_chain_submit.sh (lanes), mi4_copy_driver.sh (nohup on the login node with the
Jupiter ssh master; deletes the Atos store copy only after the name+size md5 matches). Stop everything: touch state/STOP.

Progress: grep -h "DATE_DONE\|FAIL\|BUILT\|GRIB_CLEANED" logs/mi4_*.out; ls state/copied_*.ok | wc -l; grep -h "^OK\|MISMATCH" logs/cp_*.log
