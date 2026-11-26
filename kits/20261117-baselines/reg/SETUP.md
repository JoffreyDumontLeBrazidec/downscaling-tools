# REG baseline: Jupiter sandbox build (17 Nov 2026)

Row in the paper: "GL512 450k warm start, deterministic objective, 30k steps". Same network, data, graph, statistics and
10 outputs as GL512; `training.training_approach: deterministic` (sigma fixed at 5e5, so MSE on the residual, optimum =
conditional mean). Read with one forward per input member (`SAMPLER_MODE=det` in `xset_predict_global.py`).

Cost: 2 nodes x 4 GH200, ~0.7 s/step (FS-242 r4000 measured 0.676 s/step on 8 GPUs), 30k steps = ~6 h = ~47 GH200-h;
one 12 h segment. Disk: checkpoint every 1,000 kept 1 (~6 GB) + 6 slim EMA exports per half-life (~70 GB).

Needs his typed word in the baselines thread (rule 9: a new training launch). Everything before the `sbatch` line is
login-node work.

## Paths

```
EXP=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261117-regression-baseline
GS=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261111-gs-global-crop          # FS-242, the template
GL=/e/data1/jureap-data/ecmwf/users/jdumont/outputs/checkpoint_nm50_global/db_global_512_300k/7bb77ac91f5f444e9132cd133713c0c3
GLCFG=~/dev/db_global_20261027/configs/db_global_512_300k.yaml                         # GL512's config (run 7bb77ac9)
K=<downscaling-tools checkout at branch claude/project-thread-jqfcjw>/kits/20261117-baselines/reg
```

## Steps (Jupiter login node, via ac6-100)

1. Sandbox skeleton, copied from FS-242 (never edit `$GS`):
   ```
   mkdir -p $EXP/{jobs,tools,configs,chains,logs,warm,checks}
   cp -r $GS/jobs/chain $GS/jobs/stall_watchdog.py $GS/jobs/train_with_peakmem_h9.py $EXP/jobs/
   cp $GS/tools/*.py $GS/GRAPH_SHA256 $GS/DATASETS_PIN $GS/env.sh $EXP/
   mv $EXP/*.py $EXP/tools/
   ```
   `env.sh` is GL512's venv activation as FS-242 uses it. Check `diff -r` of `jobs/chain` against `$GS` = empty.
2. Code: own worktree at FS-242's pin (fixed kernel, deterministic approach, dynamic crops, loader_read):
   ```
   R=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261109-global-o2560-code/anemoi-core
   git -C $R worktree add --detach $EXP/code/anemoi-core 8c12d4dec1f61da50834cfe2e4880ed644c33b83
   echo 8c12d4dec1f61da50834cfe2e4880ed644c33b83 > $EXP/CODE_PIN
   md5sum $EXP/code/anemoi-core/models/src/anemoi/models/layers/conv/gt.py   # (real path: models/src/anemoi/models/triton/gt.py) must be 4cf07c3e422218a3d3aab7fc769fed7f (blob 88b03616)
   ```
   Check that `training_approach` "deterministic" exists at this pin:
   `grep -rn '"deterministic"\|deterministic' $EXP/code/anemoi-core/training/src/anemoi/training/train/tasks/ | head`.
   If it is absent, stop and report (do not add code on the cluster; fixes go on ds-trunk first).
3. Warm file (450k EMA-2000 weights in a Lightning frame). Look first for an existing one:
   `ls /e/data1/jureap-data/ecmwf/users/jdumont/sandbox/*/warm*/*450000*ema2000*warm*.ckpt 2>/dev/null`.
   If none, build it (CPU, ~10 min, ~12 GB RAM; frame = GL512's own 450k training checkpoint, so the EMA shadow
   cross-check runs):
   ```
   source $EXP/env.sh
   python3 $GS/reads/valcap/make_ema_warm.py \
     $GL/anemoi-by_step-epoch_449-step_450000.ckpt \
     $GL/inference-ema2000-anemoi-by_step-epoch_449-step_450000.ckpt \
     $EXP/warm/gl512_step450000_ema2000_warm.ckpt | tee $EXP/warm/make_ema_warm.log
   sha256sum $EXP/warm/gl512_step450000_ema2000_warm.ckpt > $EXP/warm/WARM_SHA256
   ```
   The log must show every `model.*` key set from the export and the shadow cross-check passing.
4. Config and job files (from the kit; reviewed generators, exact-once substitutions):
   ```
   python3 $K/make_reg_config.py $GLCFG $EXP/configs $EXP/warm/gl512_step450000_ema2000_warm.ckpt
   python3 $K/make_reg_job.py $GS/jobs/gs_job.sh $GS/jobs/gs_accept.sh $EXP/jobs
   cp $K/sandbox_jobs/reg_gl512_450k_det.sbatch $EXP/jobs/
   cp $K/sandbox_jobs/read_hook.sh $EXP/tools/
   bash -n $EXP/jobs/reg_job.sh && bash -n $EXP/jobs/reg_accept.sh
   diff <(python3 -c 'import yaml,sys;print(yaml.safe_dump(yaml.safe_load(open(sys.argv[1])),sort_keys=True))' $GLCFG) \
        <(python3 -c 'import yaml,sys;print(yaml.safe_dump(yaml.safe_load(open(sys.argv[1])),sort_keys=True))' $EXP/configs/reg_gl512_450k_det.yaml) \
        > $EXP/checks/config_vs_gl512.diff
   ```
   The diff may hold only: warm_start, transfer_learning, load_weights_only, training_approach, lr block, max_steps,
   ema export keys, checkpoint frequency, run_name, num_nodes, num_gpus_per_model, num_workers, prefetch_factor,
   limit_batches.validation, num_sanity_val_steps, recompile_limit, model.compile, training.crop. Anything else: stop.
5. Chain state and dry checks:
   ```
   python3 $EXP/jobs/chain/chain_state.py init $EXP/chains/reg_gl512_450k_det --tag reg_gl512_450k_det --end 30000
   CFG_ONLY=1 bash $EXP/jobs/reg_gl512_450k_det.sbatch    # renders the config, prints the resolved keys
   DRY_RUN=1  bash $EXP/jobs/reg_gl512_450k_det.sbatch    # prints the srun line, checks warm sha, graph sha, pins
   ```
   (`chain_state.py` arguments: copy the exact `init` line from `$GS/HANDOFF.md` if they differ.)
6. Submit (his word): `sbatch $EXP/jobs/reg_gl512_450k_det.sbatch`. Record the job id in `$EXP/HANDOFF.md` and in the
   docs note `epics/paper-o320-o1280/in-progress/20261117_baselines.md`.

## Gate (in-chain smoke, fresh segment, steps 0-200; `reg_accept.sh`)

Warm-start lines > 0 and the frozen QK gains equal the warm file's; `training_approach deterministic` line; 10 outputs;
lr groups 1e-4 wd 0, lr inside warm-up; finite loss; last-50 <= 1.3 x first-50; memory under the FS-242 limits; no crop
fallback; no recompile-limit warning. FAIL -> `<chain root>/SMOKE_FAILED.txt`, exit 1, no successor.

Step-0 loss = GL512's own MSE at sigma 5e5, a level its noise law (NM50) almost never sampled, so it may be high;
record it. A loss that rises over the first 200 steps is a FAIL.

## Reads

Exports `$EXP/chains/reg_gl512_450k_det_r2/b8f583a2.../inference-ema2000-*-step_{20000,30000}.ckpt` (run of job 2968869; only the 30k raw by-step file is kept, the 20k row builds its trainer from it). Rows `REG_GL512_30k` (paper)
and `REG_GL512_20k` (convergence check) in `kits/20261116-extremes-set/jupiter/xset_runs.sh`; same for S25 with the
S25 env (`kits/20261117-baselines/README.md`). If 20k and 30k differ by more than 1 % in station CRPS, report it.
