#!/usr/bin/env python3
"""Config of the regression baseline REG (17 Nov 2026, paper thread "Score classical, regression and CorrDiff baselines").

What the paper states: "GL512 450k warm start, deterministic objective, 30k steps" = the same network and data as the
generative model, fine-tuned with training_approach deterministic (EDM preconditioning with sigma fixed at 5e5, so the
loss is the plain MSE of the residual and the optimum is the conditional mean), read with one forward per input member.

Source: GL512's config ~/dev/db_global_20261027/configs/db_global_512_300k.yaml (run 7bb77ac9). Changes, all of them:
  - warm start = GL512 step 450,000 with every model tensor from its EMA-2000 export (LL_WARM_FILE; make_ema_warm.py),
    transfer_learning true, load_weights_only true (fresh optimizer, step 0)
  - training.training_approach: deterministic
  - lr warm-up 1000 to 1e-4 then constant (rule 22, continuation of converged weights); 30,000 steps
  - dynamic 4,000 km caps + 800 km halo, one GPU per model, 2 nodes x 4 GH200 = global batch 8 (rule 3; FS-242's crop
    block and compile settings, make_configs.py crop()); validation on 16 caps of 3,000 km
  - EMA [2000, 5000] (GL512's), slim exports every 5,000 steps; checkpoint every 1,000 steps, latest kept
Kept from GL512: class, 512 x 16 x 4 heads, 10 outputs, NL35, AdamW wd 0, frozen QK gains, data, graph, statistics, seed.
Usage: python3 make_reg_config.py <gl512.yaml> <out_dir> <warm_file>
"""
import copy
import sys

import yaml

src, out, warm = sys.argv[1], sys.argv[2], sys.argv[3]
base = yaml.safe_load(open(src))
OUT10 = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "cp", "tp"]
STEPS = 30000


def reg(c, run_name, max_steps, nodes, radius, smoke=False):
    c = copy.deepcopy(c)
    t = c["training"]
    for split in ("training", "validation", "test"):
        assert c["dataloader"][split]["datasets"]["out_hres"]["reorder"] == OUT10, "GL512's 10 outputs expected"
    c["system"]["input"]["warm_start"] = warm
    t["transfer_learning"] = True
    t["load_weights_only"] = True
    t["training_approach"] = "deterministic"
    t["lr"].update({"warmup": 50 if smoke else 1000, "rate": 1.0e-4, "min": 1.0e-4, "iterations": max_steps, "schedule": "constant"})
    t["max_steps"] = max_steps
    assert t["ema"]["halflife_steps"] == [2000, 5000] and t["ema"]["enabled"] is True
    t["ema"]["export_keep_every"] = 5000
    t["ema"]["skip_raw_export"] = True
    c["diagnostics"]["checkpoint"]["every_n_train_steps"] = {"save_frequency": 100 if smoke else 1000, "num_models_saved": 1}
    c["diagnostics"]["log"]["mlflow"]["run_name"] = run_name
    c["system"]["hardware"]["num_nodes"] = nodes
    c["system"]["hardware"]["num_gpus_per_model"] = 1
    c["dataloader"]["num_workers"] = {"training": 6, "validation": 2, "test": 1}
    c["dataloader"]["prefetch_factor"] = 2
    c["dataloader"]["limit_batches"]["validation"] = 2
    t["num_sanity_val_steps"] = 2
    t["recompile_limit"] = 128
    c["model"]["compile"] = [
        {"module": "anemoi.models.layers.conv.GraphTransformerConv", "options": {"dynamic": True}},
        {"module": "anemoi.models.layers.normalization.ConditionalLayerNorm", "options": {"dynamic": True}},
    ]
    t["crop"] = {
        "enabled": True, "mode": "dynamic", "seed": None,
        "dynamic": {
            "radius_mode": "fixed", "radius_km": float(radius), "halo_km": 800.0, "centre_spacing_km": 25.0,
            "capacity": "fixed", "capacity_probes": 256, "pad_multiple": 8,
            "loader_read": True, "loader_margin_km": 300.0,
            "validation_caps": 16, "validation_radius_km": 3000.0,
        },
    }
    return c


def dump(c, name):
    with open(f"{out}/{name}.yaml", "w") as f:
        yaml.safe_dump(c, f, sort_keys=False, default_flow_style=False, width=4096)
    print(f"wrote {out}/{name}.yaml")


dump(reg(base, "reg_gl512_450k_det", STEPS, 2, 4000), "reg_gl512_450k_det")
