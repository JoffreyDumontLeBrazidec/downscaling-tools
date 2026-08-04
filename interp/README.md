# interp — mechanistic interpretability for AIFSDD

Tools to explain how the Anemoi diffusion downscaler creates storms,
precipitation extremes and small-scale structure: which inputs matter (also
specifically for extremes), which layers/stages are involved, and whether the
explanations are physically plausible.

All metrics are **per surface target** ({10u, 10v, 2t, msl, tp}; tp only in the
o48→o96 lane), and all data comes from **real event bundles** — never
`val_dataloader()` (broken lres grid post 2026-06-11 reorg, and no storm at the
probe). Named events live in `interp/core/data.py:EVENTS`.

## Question → tool

| Question | Tool | Output JSON |
|---|---|---|
| Which inputs matter (per noise level / end-to-end)? | `permutation` (`--mode sigma\|sampling`) | `permutation_importance[_full_sampling].json` |
| Which inputs matter **for the extremes**? | `permutation --extreme-percentile 99`, `ig --functionals tail` | same + `integrated_gradients.json` |
| Which inputs create the **small scales**? | `ig --functionals spectral` | `integrated_gradients.json` |
| Do TC min-MSLP / max-10m-wind extremes emerge cheaply enough for 10k-step sweeps? | `tc_emergence`, `tc_emergence_sweep` | `tc_emergence.json`, `tc_emergence_sweep_{rows,by_sigma,summary,ranked}.csv`, `tc_emergence_sweep_by_sigma_ranked.csv` |
| How much does each conditioning pathway carry vs σ? | `ablation` | `conditioning_ablation.json` |
| Which block/stage causally commits each field? | `patching` (residual / grid_region / stage) | `activation_patching.json` |
| What drives the storm core, and is it local? | `ig --functionals eye,box` | `integrated_gradients.json` (maps + coherence) |
| Layer activity / representation structure vs σ? | `activations` (norms + CKA, one pass) | `activation_profiling.json`, `cka_analysis.json` |

## Running

```bash
cd ~/dev/downscaling-tools

# locally (login node, CPU, smoke):
python -m interp permutation --checkpoint $CKPT --output-dir /tmp/out \
    --event franklin_o96_o320_m4 --device cpu --sigmas 5.0 --n-repeats 1

# exact TC min/max probe for one checkpoint (Franklin + Idalia; default selects top-20 input MSLP-min and top-20 input wind-max cases per TC):
python -m interp tc_emergence --checkpoint $CKPT --output-dir /tmp/tc_emergence \
    --case-preset o96_o320_idalia_franklin --sigmas 120 80 50 20 10 5 1 --modes free teacher

# ladder comparison: TC-good cfec83a3 vs TC-poor 0c446b41, by-step checkpoints and the same strong-input TC case set:
python -m interp tc_emergence_sweep --output-dir /tmp/tc_emergence_sweep \
    --run good_cfec='/home/ecm5702/scratch/aifs/checkpoint/cfec83a3cd0644778e2bfcbacfa9f4fc/anemoi-by_step-*.ckpt' \
    --run poor_0c446='/home/ecm5702/scratch/aifs/checkpoint/0c446b4118b94ec2bbec56c00409d664/anemoi-by_step-*.ckpt' \
    --case-preset o96_o320_idalia_franklin --sigmas 120 80 50 20 10 5 1 --modes free teacher

# on SLURM (GA100 + env_ref.sh by default):
sbatch interp/slurm/run.sbatch tier1 $CKPT cfec_200k
sbatch interp/slurm/run.sbatch patching $CKPT cfec_200k
sbatch interp/slurm/run.sbatch ig $CKPT cfec_200k --functionals global_mean,eye,tail,spectral

# GH200 instead (aarch64 .ds-ag venv):
sbatch --gres=gpu:gh200:1 --export=ALL,INTERP_ENV=env_gh200 \
    interp/slurm/run.sbatch ig $CKPT cfec_200k
```

Results land in `~/perm/interp/<ckpt_id>/<tool_subdir>/`, each with a
`run_meta.json` (args, checkpoint, git sha) for later comparison.

## Report & comparison

```bash
# ONE multi-page PDF per checkpoint (storm/extreme maps first):
python -m interp.viz --run-dir ~/perm/interp/cfec_200k          # -> plots/report.pdf

# summary-scalar diff across checkpoints (open report.pdf only if a scalar moves):
python -m interp compare --run-dirs ~/perm/interp/59e4_300k ~/perm/interp/cfec_200k \
    --output-dir ~/perm/interp/compare_59e4_cfec
```

## Layout

```
core/      model.py (load/denoise/sample) · data.py (bundles, EVENTS) ·
           geometry.py · regions.py (masks + differentiable functionals) ·
           hooks.py (block discovery/capture) · runmeta.py
tools/     permutation.py · ablation.py · activations.py · patching.py · ig.py
viz/       panels.py (heatmap/loglog/bars/geo templates) · report.py
compare.py · cli.py (python -m interp <tool>) · slurm/ (env_ref, env_gh200, run.sbatch)
```

## Gotchas

- `denoise_at_sigma` loops batch-1 internally: anemoi-core-ref mis-assembles the
  encoder input at batch>1. IG's grad path can't loop, so run IG with ONE bundle.
- `patching`: full-block patching is degenerate for this architecture (see the
  module docstring); `residual` is the per-block localizer, sanity checks
  `patch_none≈0 / patch_all≈1 / enc_both≈1` must hold.
- Epic + findings: `~/dev/docs/epics/mechanistic-interpretability/`.
