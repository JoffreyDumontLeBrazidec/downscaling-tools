# health: how much of a checkpoint's capacity is dead or misused

`python -m health audit <ckpt>` reads one checkpoint and says how much of the network does nothing: heads and MLP
units with zero weights, units that never get a gradient, saturated or linearised units, blocks that add almost
nothing to the residual stream, inputs the model never reads. It gives **readings, not gates**: nothing stops a run,
the exit code is always 0 (owner, 30 Oct 2026). Use it to decide whether a checkpoint could be better, for example
by fine-tuning after reviving dead units.

The design and the literature behind it are in the docs repo: `epics/training-diagnostics/network-health/in-progress/20261030_nn_sanity_lane_review.md`
(s4 lists the 12 checks; s8 records the owner's answers; s9 holds the first sweep).

## When to run it

- When a checkpoint lands and is read (the thread that reads it owns the reaction; the headline goes in the run
  status note next to the read).
- Before a capacity claim ("width bought nothing", "A = B"): CA300's 64 dead heads voided one such claim.
- When an optimizer misbehaves on a converged checkpoint, or a count smells wrong.

## Commands

```bash
cd ~/dev/downscaling-tools
# CPU checks 1, 3, 6 (numpy + yaml, no torch; any login node or a small job; ~1 min per 2 GB file)
python -m health audit CKPT [--lane o320_o1280_box] [--prev PREV/audit.json] [--out DIR] [--id NAME]
# + GPU checks 2, 4, 7, 9 and the graph audit 8 (torch + the model code + one GPU; 2 bundles x 4 sigmas = 8 cases)
python -m health audit CKPT --gpu --bundles B1_input_bundle.nc B2_input_bundle.nc [--local-scope-json BOX.json]
python -m health diff A/audit.json B/audit.json      # check 12 between any two audits
python -m health registry                            # the reference checkpoint of each lane
python -m health revive SRC DST --audit A/audit.json # BY HAND only: write a revived copy (needs torch)
sbatch health/slurm/audit.sbatch CKPT [CKPT ...]     # several files in one job (header: per-cluster env)
```

Python: the cluster venv (`~/dev/.ds-ag-260616/bin/python3` on AG, `.ds-260612` on AC, the `.ds` venv after
`module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3` on Jupiter). The login nodes' system python is too old.

Output: `$HEALTH_ROOT/<ckpt id>/audit.json` and `SUMMARY.txt` (default `~/perm/health`; id = `<run8>_<step>k[_emaN]`).
`SUMMARY.txt` opens with the headline, then one line per check (`FAIL` = exact zero found, `WARN` = outside a
provisional band, `ok`, `info` = no band yet), the non-finite and EMA-alias lines, the optimizer mapping line, the
revival commands and the check-12 diffs.

## What each check reads

Every metric is a FRACTION or RATIO per block, keyed by ROLE (`encoder`, `processor.<i>`, `decoder`, `mapper:<name>`),
so models of different widths read on the same bands.

| # | Check | Needs | Reads |
|---|---|---|---|
| 1 | zero-content heads and units; NaN/Inf; EMA `*_dest` alias mismatch | file | head dead = q, k, v, self, edge rows and projection columns all exactly 0; unit dead = W1 row and W2 column 0 |
| 3 | gradient-dead from the optimizer state | training file (by_step) | unit dead = second moment exactly 0 on its bias, and on its W1 row and W2 column (AdamW: `exp_avg_sq`; SOAP: the diagonals of `GG`, since its `exp_avg_sq` is in a rotated basis; Muon: bias only). A zero moment means no gradient for at least the WINDOW printed in the summary (the moment underflows: ~1.5k steps at beta2 0.95, ~70k at 0.999), not "for the whole run". Head = q, k, v bias moments 0. Zero-content units are not counted twice |
| 6 | scale ladders | file | weight RMS over nn.Linear init RMS; QK gains; edge-logit scale per head (s11 ladder: healthy < 0.2, growing > 0.5); LayerNorm gains (exact 0 = dead channel); low-norm units (row norm < 0.2 x median) |
| 2 | gradient-dead on K batches (GraMa) | GPU | per unit / head mean abs gradient per case; dead = exactly 0 on all K, dormant = 0 or < 1e-3 of the layer mean on >= 3/4 of the cases |
| 4 | pre-activation saturation | GPU | per unit over all nodes: max < -5.55 (GELU output exactly 0), max < -14 (gradient exactly 0), min > 0 (linear) |
| 7 | residual-stream budget | GPU | per processor block: update / stream norm, attention and MLP parts, max/median abs activation, Dirichlet energy |
| 9 | information flow | GPU | sigma 0.2, 5, 80: every input variable gets a gradient, every output variable moves when the inputs move |
| 8 | graph | GPU run | per edge type: destination nodes with no in-edge, degree range, components of the processor graph, edge-index sha1 |
| 12 | trend | two audits | each reading vs the previous checkpoint (`--prev`) and the lane reference (`--lane`, registry.yaml); exact zeros that appear or vanish always listed |

Not here: check 5 (attention N_eff per head; the pyg ladder in docs `support/20261028_db_attention_health/`), the
smoke tier S, the in-training callback tier T (deferred by the owner: no chain restarts, no kernel change).

## Bands and the reference registry

`bands.yaml`: exact-zero findings are `FAIL`; the `WARN` bands are provisional and are set from the first sweep
(note s9). `registry.yaml`: one reference checkpoint per lane (the cleanest on file, with the reason); every audit with
`--lane` diffs against it. Replace an entry when a cleaner checkpoint lands.

## Reading the numbers

- `grad_dead_unit_frac` needs the optimizer state: inference and EMA files carry none, so audit the by_step file.
  "0 zero-weight units" does not mean no dead units (DB300: 0 zero-weight, 125 gradient-dead in processor block 2).
- The optimizer state is matched to names by position and shape, after removing buffers, aliases and what the config
  freezes (`submodules_to_freeze`, `freeze_qk_gains`). `optimizer map: ... exact` is the proof the labels are right;
  `AMBIGUOUS` means a tensor without a state shares a shape with a state (a frozen block the config does not name), so
  every name after it may be shifted: do not quote gradient-dead indices from such a file. The bias / W1-row /
  W2-column counts agreeing (`agreement` in the JSON) is a second check.
- A dead unit costs nothing in the forward (its output is exactly 0) but is capacity paid for and unused.
- Revival is by hand: `python -m health revive` redraws the input side (W1 row, head input rows) from nn.Linear's
  init and zeroes the output side, then warm start from the new file. The function is unchanged only where the unit's
  output was exactly 0: always for zero-weight units and heads; for gradient-dead units only with check-4 evidence
  (`never_fires`, from an audit with `--gpu`), else they are skipped unless `--unproven`. Gradient-dead heads are not
  revived. It refuses when the optimizer mapping is not exact. Never revive inside a running chain.

## Origin

Consolidates the docs kits: `audit_dead_units.py` (zero-weight heads, EMA aliases; fine-scale `support/20261027_dead_heads`),
`adam_state_dead_units.py` and `zero_grad_units.py` (optimizer-state dead units; `support/20261030_gradient_dead_units`,
`support/20261030_zero_grad_units`), `m2/all_blocks_scale.py` (edge-logit ladder). anemoi-core's
`anemoi.training.utils.dead_units` (fork 18a5415af) is the same criterion as check 1 with revival at load time.
Tests: `python -m pytest health` (a toy checkpoint with planted faults; CPU, seconds).
