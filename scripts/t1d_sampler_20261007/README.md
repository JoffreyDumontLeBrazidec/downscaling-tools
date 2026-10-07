# T1d stage A: sampler screen on the 1.2M parent (2026-10-07)

Box screen (T1 stage-1 protocol) of four Heun churn-off samplers on the 1.2M parent (run `551dfd1e`, step 400,000),
Franklin/Idalia box, 5 dates x leads 24/120 x 10 members = 100 draws per run, one A100 on Atos AC per run.
How to run it: `LAUNCH.md`. Other builders own `dp/` and `interp/`; nothing here touches them.

| file | what it is |
|---|---|
| `LAUNCH.md` | the executor's runbook: preparation, gates G1-G3, the owner's go, submission order, dependency graph, GPU-hours, sacct |
| `t1d_env.sh` | shared paths and settings; `T1D_HOST` = `ac` (default) or `ag` picks the cluster of the gate and the predictions (work folder, code worktree, sandbox, certified venv, input bundles, run-root naming); every script sources it |
| `make_lanes_t1d.py` | writes the four arm lanes `tc_o320_o1280_p12m_*` (sampler_overrides on `tc_o320_o1280_p12m_ctrl`), prints the arm table and every applied sigma list with log10 and ln steps; `--verify-fork` checks the lists bit for bit against the fork's own schedulers (anemoi-core 27391c1); refuses to overwrite without `--force` |
| `resolve_lane_t1d.py` | loads a lane through `eval/config/loader.py`, writes the sampler JSON, scope JSON and base-checkpoint path the job passes, refuses any checkpoint but the 1.2M parent (G2 part 1) |
| `se_tc_predict_t1d.sbatch` | one box prediction run (`python -m eval.predict.main`, the command `eval.cli predict` builds) on 1 A100, sandbox or certified runtime, seed 756 or 757000, probe and nvidia-smi sidecar |
| `submit_gates_t1d.sh` | gate G1: the one-draw job under the sandbox and under the certified venv (`TEST=1` = `sbatch --test-only`) |
| `g1_compare.py` | gate G1: compares two prediction files field by field (max abs diff per variable, relative, bitwise) |
| `check_attrs_t1d.py` | gate G2 part 2: the files' sampler attribute, checkpoint and members, and the probe's schedule and calls in the job log |
| `submit_post_t1d.sh` | AG route only: on hpc-login, after the five AG predictions completed, the AC half (5 evaluations, then intensity, spectra, read) |
| `submit_stageA_t1d.sh` | stage A: five predictions, each with its evaluation (afterok), then intensity, spectra and read (afterok on all five evaluations); refuses existing run roots; `TEST=1` = gate G3 |
| `se_tc_eval_t1d.sbatch` | `eval.cli evaluate --only tc,texture,wind_extremes,probabilistic,surface,shape --steps 24,120` on one run |
| `box_post_t1d.sbatch`, `tc_intensity_t1d.py` | per-draw storm intensity table (T1 stage 3 `tc_intensity.py` with the runs as arguments) + checks + summary |
| `spectra_t1d.sbatch` | v3 box amplitude spectra (`--skip-maps --spectra-method linear`, leads 24/120, all 100 draws, truth and interpolated input as references) |
| `read_t1d.sbatch` | `../fewstep_sampler_20260930/read_stage1.py`: baseline p12m_pw30_c0 s756, one replicate (s757000), T1 section 6 floors |
| `build_bundle_t1d.py` | copies metrics, intensity, spectra, read, gates, sacct/timings, jobs.tsv and lanes into `<work>/bundle/stageA/` for docs `in-progress/20261007_T1d_sampler_beat_1p2M_results/stageA/` |
| `common/plot_sampler_texture_v3.py`, `common/tc_intensity_summary.py` | unchanged copies from the T1 stage 3 docs bundle (`20260930_T1_stage1_results/stage3/scripts/common/`) |
| `common/probe_sampler.py` | unchanged copy of T1 `stage2/scripts/probe_sampler.py` (print-only wrappers of the schedulers and the Heun sampler) |

Lanes (in `eval/config/lanes/`): `tc_o320_o1280_p12m_ctrl` (base; differs from `tc_o320_o1280_ft400k_gmass_ctrl` by one added
key, `predict.checkpoint`), `tc_o320_o1280_p12m_pw30_c0`, `..._p12m_c0_pw16_s1k`, `..._p12m_st2`, `..._p12m_st4`.
