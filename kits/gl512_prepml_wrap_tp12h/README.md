# GL512 prepml wrap gl512-20261118-prepml-wrap-fast-tp12h (18 Nov 2026)

Live copy: /home/ecm5702/dev/anemoi-inference-configs/prepml_envs/gl512-20261118-prepml-wrap-fast-tp12h (Atos). This folder
holds its source files under version control. Lane: eval/config/lanes/o320_o1280_prepml_gl512_450k_sept2025_tp12h.yaml.

Built from gl512-20261111-prepml-wrap-fast (TF32, one-row ConditionalLayerNorm, edge padding; global-noise and rule-31 seed
patches in gl512_runner_patches.py unchanged) plus two fixes:
1. gl512_tp_patches.py: anemoi-inference Accumulate gets tp = cp = 0 at step 0. The lane steps every 12 h, because the
   checkpoint tp/cp are 12-h accumulations (out_hres variables_metadata period [0h, 12h]) and Accumulate sums every step.
2. FORCING_DATE_FIX.diff: the frozen runner copy season-scale-tools-frozen-20261118-forcingfix (sandbox
   20261109-gl512-prepml) uses input_date = date in UnifiedDownscalingRunner.predict_step, so the high-res time forcings
   are those of the valid date. Same change as anemoi-inference commit 39d5371 (agent-work/20261117-forcing-date). The
   fork DownscalingRunner.predict_step is not called by this runner.
Files: activate (= bin/activate), shim_anemoi-inference (= shim_bin/anemoi-inference), sitecustomize.py and the three patch
modules (= site/), build_tp12h_wrap_step2.sh (how the runner copy was made). Record: docs
epics/misc/20261117_quaver_scorecards_for_paper.md s5.8-s5.9.
