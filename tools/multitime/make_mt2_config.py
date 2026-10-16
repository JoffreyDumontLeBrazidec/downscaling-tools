#!/usr/bin/env python3
"""Write the multi-time E2 config mt2_e2_20k.yaml for Vera Gahlen's spatial-downscaler branch.

The run mt2_e2_20k trains a multi-in multi-out diffusion downscaler (two time slots 12 h
apart, o320 -> o1280, Europe box E2) on the fork branch claude/project-thread-n1zaq3 of
anemoi-core (Vera's feat/spatial_downscaler 6587cc679 plus HindcastBlockDataset, the
log_normal training noise and the heun_window sampler), warm-started from our single-time
donor 12dcefea.

The LAYOUT (key names, group defaults, structure) comes from Vera's o48 -> o96 config
  /home/ecm5702/hpcperm/sandbox/20260917-vera-spatial-downscaler/config/o48_o96_100k_atos.yaml
and the VALUES from our single-time E2 recipe
  /ec/res4/scratch/ecm5702/sandbox/20261007-pl25-ac/configs/do0_e2_50k_ac.yaml
through the mapping of the 17 Sep loss-match note (diagnostic: [tp, cp] for our
direct_prediction, trainable node and edge parameters 0, 512 channels, 4 heads, 16 layers,
the same graph, scalers and variable weights, residual statistics as .npz, the projection
matrix through CrossGridProjector).

Vera's config validation is off, so a mistyped key is silently ignored. Every edit below is
therefore asserted: the key exists in the layout file, or a named source file of the
anemoi-core checkout reads it (the check greps the code). The script refuses to write if any
assertion fails.

Outputs (in --out-dir):
  mt2_e2_20k.yaml                                   the flat config (Hydra primary config)
  model/graphtransformer_transport_residual_edm_mt2.yaml   the model group it selects
  CHECK.md                                          the line-by-line check list
  mt2_e2_20k.layout_diff.txt                        unified diff against the layout file

Rendering (the effective, Hydra-composed config) is a separate step done with the branch's
own composition (see --render).

Usage (Atos, any python3 with PyYAML):
  python3 make_mt2_config.py --out-dir <S>/configs --code-root <S>/code/anemoi-core \
      --sandbox <S> [--render]
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

LAYOUT = Path("/home/ecm5702/hpcperm/sandbox/20260917-vera-spatial-downscaler/config/o48_o96_100k_atos.yaml")
LAYOUT_MODEL = Path(
    "/home/ecm5702/hpcperm/sandbox/20260917-vera-spatial-downscaler/config/model/"
    "graphtransformer_transport_residual_edm_lossmatch.yaml"
)
RECIPE = Path("/ec/res4/scratch/ecm5702/sandbox/20261007-pl25-ac/configs/do0_e2_50k_ac.yaml")

E2V4 = "/ec/res4/scratch/ecm5702/o320_o1280_europe_e2_v4/data"
LRES_ZARR = f"{E2V4}/in_lres_o320_halo1p5.zarr"
HRES_ZARR = f"{E2V4}/out_hres_o1280.zarr"
FORCINGS_ZARR = "/ec/res4/scratch/ecm5702/o320_o1280_europe_e2/data/in_hres_o1280_forcings.zarr"
LANE = "/home/ecm5702/hpcperm/lanes/o320_o1280_europe_e2"
# The E2-box O320 -> O1280 matrix of the do0 recipe (system.input.truncation there). The global
# grid-box-average matrix interpol_o320_to_o1280_grid-box-average.mat.npz is 6,599,680 x 421,120 and
# cannot multiply the box stores (55,871 -> 814,810 points).
TRUNCATION = f"{LANE}/inter_mat/interpol_O320_to_O1280_linear-europe_e2_10_70N_80W_40E_halo1p5.mat.npz"
TRUNCATION_INV = f"{LANE}/inter_mat/interpol_O1280_to_O320_linear-europe_e2_10_70N_80W_40E_halo1p5.mat.npz"

STEPS = [12, 24, 36, 48, 60, 72]
FORK_DATASET_PY = "/home/ecm5702/dev/anemoi-datasets/src/anemoi/datasets/data/dataset.py"  # lane fork, read only
RUN_NAME = "mt2_e2_20k"
MAX_STEPS = 20000
N_NODES, N_GPUS_PER_NODE, N_GPUS_PER_MODEL = 2, 4, 1
TRUE_LR = 1.0e-4  # the rate the optimiser must see (our lr.rate, not scaled by the GPU count)
EMA_HALFLIFE = 2000  # the branch has ONE EMA callback; our recipe has two (2000, 10000), see CHECK.md

def _levels(p):
    return [f"{p}_{lev}" for lev in (100, 1000, 200, 300, 400, 50, 500, 700, 850, 925)]


# Variable orders. in_lres: the store order without cp and tp (68), which is also the order of our
# donor's 68 outputs; out_hres: the do0 reorder list (the same 68, then cp, tp), so the 70 outputs
# have the do0 model's output order and cp, tp sit at the end, where the 68 -> 70 extension put them.
OUT68 = (["10u", "10v", "2d", "2t", "msl"] + _levels("q") + ["skt", "sp"] + _levels("t") + ["tcw"]
         + _levels("u") + _levels("v") + _levels("w") + _levels("z"))
IN_LRES = list(OUT68)
OUT_HRES = OUT68 + ["cp", "tp"]
IN_HRES = ["cos_julian_day", "cos_latitude", "cos_local_time", "cos_longitude", "insolation", "lsm",
           "sin_julian_day", "sin_latitude", "sin_local_time", "sin_longitude", "z"]
HRES_FORCING = ["cos_latitude", "cos_longitude", "sin_latitude", "sin_longitude", "cos_julian_day",
                "cos_local_time", "sin_julian_day", "sin_local_time"]


class Editor:
    """Applies edits to a nested dict and asserts that each key is known."""

    def __init__(self, layout: dict, code_root: Path):
        self.layout = copy.deepcopy(layout)
        self.cfg = copy.deepcopy(layout)
        self.code_root = code_root
        self.log: list[tuple[str, str, str]] = []  # (path, value, evidence)
        self.errors: list[str] = []

    def _code_has(self, rel: str, needle: str) -> bool:
        path = self.code_root / rel
        if not path.exists():
            return False
        return needle in path.read_text()

    @staticmethod
    def _get(d: dict, keys: list[str]):
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                raise KeyError(k)
            d = d[k]
        return d

    def _in_layout(self, keys: list[str]) -> bool:
        try:
            self._get(self.layout, keys)
            return True
        except KeyError:
            return False

    def set(self, path: str, value, code: tuple[str, str] | None = None, note: str = "") -> None:
        keys = path.split(".")
        if self._in_layout(keys):
            evidence = "layout"
        elif code is not None and self._code_has(*code):
            evidence = f"code: {code[0]} contains {code[1]!r}"
        else:
            self.errors.append(f"{path}: not in the layout and no code evidence {code}")
            return
        d = self.cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = copy.deepcopy(value)
        self.log.append((path, json.dumps(value, default=str) if not isinstance(value, str) else value,
                         evidence + (f"; {note}" if note else "")))

    def delete(self, path: str, why: str) -> None:
        keys = path.split(".")
        if not self._in_layout(keys):
            self.errors.append(f"delete {path}: not in the layout")
            return
        d = self._get(self.cfg, keys[:-1])
        del d[keys[-1]]
        self.log.append((path, "(deleted)", why))


def dataset_entry(dataset: str, select: list[str], fake: dict, start, end) -> dict:
    return {
        "dataset_config": {"dataset": dataset, "frequency": "${data.frequency}", "select": select,
                           "fake_hindcasts": fake},
        "start": start,
        "end": end,
        "hindcast_blocks": {"steps_per_sequence": 6, "step": "12h"},
    }


def build(code_root: Path, sandbox: Path) -> tuple[dict, dict, Editor, Editor]:
    layout = yaml.safe_load(LAYOUT.read_text())
    recipe = yaml.safe_load(RECIPE.read_text())
    ed = Editor(layout, code_root)
    R = "training/src/anemoi/training/"
    M = "models/src/anemoi/models/"

    # ---- defaults: our model group (Vera's current layout of the residual EDM group) and the
    # existing-graph group (the graph is a converted file; nothing is rebuilt) -------------------
    defaults = copy.deepcopy(layout["defaults"])
    new_defaults = []
    for item in defaults:
        if isinstance(item, dict) and "model" in item:
            new_defaults.append({"model": "graphtransformer_transport_residual_edm_mt2"})
        elif isinstance(item, dict) and "graph" in item:
            new_defaults.append({"graph": "existing"})
        else:
            new_defaults.append(item)
    assert (code_root / R / "config/graph/existing.yaml").exists()
    ed.set("defaults", new_defaults, note="model group -> ./model/graphtransformer_transport_residual_edm_mt2.yaml; "
           "graph group -> existing (config/graph/existing.yaml: overwrite False, no nodes/edges = load the file)")

    # ---- system -------------------------------------------------------------------------------
    ed.set("system.input.dataset", HRES_ZARR, note="HRES store, used for out_hres")
    ed.set("system.input.dataset_lres", LRES_ZARR)
    ed.set("system.input.residual_statistics", str(sandbox / "assets/o1280_dict_0_72.npz"))
    ed.set("system.input.truncation", TRUNCATION, note="E2-box matrix of the do0 recipe (rows sum to 1)")
    ed.set("system.input.truncation_inv", TRUNCATION_INV)
    ed.set("system.input.graph", str(sandbox / "assets/graph/europe_e2_m600rmknn13_vera_fused.pt"),
           note="our do0 graph (graph_m600_ringmask_knn13) converted to the branch's fused node sets")
    ed.set("system.input.warm_start", None, code=(R + "config/system/input/example.yaml", "warm_start"),
           note="set by the job (weights-only warm start of the first segment)")
    ed.set("system.hardware.num_gpus_per_model", N_GPUS_PER_MODEL)
    ed.set("system.hardware.num_gpus_per_node", N_GPUS_PER_NODE)
    ed.set("system.hardware.num_nodes", N_NODES)

    # ---- data ---------------------------------------------------------------------------------
    norm = "anemoi.models.preprocessing.normalizer.InputNormalizer"
    in_lres = copy.deepcopy(layout["data"]["datasets"]["in_lres"])
    in_lres["forcing"] = []
    in_lres["diagnostic"] = []
    in_lres["processors"]["normalizer"]["config"] = {"default": "mean-std", "std": None, "min-max": None,
                                                     "max": None, "none": None}
    ed.set("data.datasets.in_lres", in_lres, note="do0 in_lres normalizer (mean-std for all 68); no forcing in the store")
    in_hres = copy.deepcopy(layout["data"]["datasets"]["in_hres"])
    in_hres["forcing"] = list(HRES_FORCING)
    in_hres["diagnostic"] = []
    in_hres["processors"]["normalizer"]["config"] = copy.deepcopy(
        recipe["data"]["datasets"]["in_hres"]["processors"]["normalizer"]["config"])
    ed.set("data.datasets.in_hres", in_hres, note="do0 in_hres forcing list and normalizer")
    out_hres = copy.deepcopy(layout["data"]["datasets"]["out_hres"])
    out_hres["forcing"] = []
    out_hres["diagnostic"] = ["tp", "cp"]
    out_hres["processors"]["normalizer"]["config"] = copy.deepcopy(
        recipe["data"]["datasets"]["out_hres"]["processors"]["normalizer"]["config"])
    ed.set("data.datasets.out_hres", out_hres,
           note="diagnostic [tp, cp] replaces do0 direct_prediction [tp, cp]; do0 out_hres normalizer (std for tp, cp)")
    assert norm == out_hres["processors"]["normalizer"]["_target_"]

    # ---- dataloader ---------------------------------------------------------------------------
    ed.set("dataloader.num_workers", {"training": 8, "validation": 4, "test": 4}, note="do0 values")
    ed.set("dataloader.limit_batches", {"training": 1000, "validation": 144, "test": 20})
    ed.set("dataloader.steps", STEPS, note="anchor holder only (not read by code)")
    ed.set("dataloader.select_matched_input", IN_LRES, note="anchor holder: the 68 in_lres variables")
    ed.set("dataloader.select_forcings", IN_HRES, note="anchor holder: the 11 forcings-store variables")
    ed.set("dataloader.select_matched_output", OUT_HRES, note="anchor holder: 68 + cp, tp")
    hb_code = (R + "data/data_reader.py", '"hindcast_blocks"')
    splits = {
        "training": ({"end": 2021, "steps": STEPS}, None, 1901),
        "validation": ({"start": 2022, "end": 2022, "steps": STEPS}, None, None),
        "test": ({"start": 2023, "end": 2023, "steps": STEPS}, None, None),
    }
    for split, (fake, start, end) in splits.items():
        for name, store, sel in (("in_lres", "${system.input.dataset_lres}", IN_LRES),
                                 ("in_hres", FORCINGS_ZARR, IN_HRES),
                                 ("out_hres", "${system.input.dataset}", OUT_HRES)):
            base = f"dataloader.{split}.datasets.{name}"
            ed.set(f"{base}.dataset_config.dataset", store)
            ed.set(f"{base}.dataset_config.select", sel)
            ed.delete(f"{base}.dataset_config.fake_forecasts", "the E2 stores carry fake_hindcasts, not fake_forecasts")
            ed.set(f"{base}.dataset_config.fake_hindcasts", fake,
                   code=(FORK_DATASET_PY, "def _fake_hindcasts_to_indices"),
                   note="lane fork anemoi-datasets (/home/ecm5702/dev/anemoi-datasets) _subset")
            ed.set(f"{base}.start", start)
            ed.set(f"{base}.end", end)
            ed.set(f"{base}.hindcast_blocks", {"steps_per_sequence": 6, "step": "12h"}, code=hb_code)

    # ---- task ---------------------------------------------------------------------------------
    ed.delete("task.offsets", "the branch tip reads input_offsets / output_offsets (task/spatial_downscaler.yaml)")
    task_code = (R + "tasks/spatial_downscaler.py", "input_offsets")
    ed.set("task.input_offsets", ["0h", "12h"], code=task_code)
    ed.set("task.output_offsets", ["0h", "12h"], code=(R + "tasks/spatial_downscaler.py", "output_offsets"))

    # ---- training -----------------------------------------------------------------------------
    ed.delete("training.transport.residual_reference",
              "moved to model.residual_reference on the branch tip (models/.../transport_encoder_processor_decoder.py)")
    ed.set("training.transport.prediction_mode", "residual")
    ed.set("training.max_steps", MAX_STEPS)
    ed.set("training.num_sanity_val_steps", recipe["training"]["num_sanity_val_steps"], note="do0 value")
    ed.set("training.precision", "bf16-mixed")
    ed.set("training.gradient_clip", {"val": 1.0, "algorithm": "norm"})
    n_inst = N_NODES * N_GPUS_PER_NODE / N_GPUS_PER_MODEL
    local_lr = TRUE_LR / n_inst
    ed.set("training.optimization.lr", local_lr,
           note=f"train/methods/base.py: effective_lr = num_nodes*num_gpus_per_node*lr/num_gpus_per_model = "
                f"{n_inst:g} x {local_lr:g} = {TRUE_LR:g}")
    ed.set("training.optimization.optimizer.weight_decay", 0.0)
    ed.set("training.optimization.optimizer.betas", [0.9, 0.95])
    ed.set("training.optimization.optimizer.eps", 1.0e-8)
    ed.set("training.optimization.lr_scheduler.lr_min", TRUE_LR,
           note="timm CosineLRScheduler with lr_min = the effective rate: linear warm-up then constant")
    ed.set("training.optimization.lr_scheduler.warmup_t", 1000)
    ed.set("training.optimization.lr_scheduler.t_initial", "${training.max_steps}")
    decay = 0.5 ** (1.0 / EMA_HALFLIFE)
    ed.set("training.weight_averaging",
           {"_target_": "anemoi.training.diagnostics.callbacks.weight_averaging.EMAWeightAveraging", "decay": decay},
           code=(R + "train/train.py", 'getattr(self.config.training, "weight_averaging", None)'),
           note=f"decay 0.5**(1/{EMA_HALFLIFE}) = {decay:.10f} (half-life {EMA_HALFLIFE} steps); ONE EMA only")
    gv = copy.deepcopy(recipe["training"]["scalers"]["datasets"]["out_hres"]["general_variable"]["weights"])
    ed.set("training.scalers.datasets.out_hres.general_variable.weights", gv, note="do0 NL35 weights")
    ed.set("training.scalers.datasets.out_hres.node_weights.nodes_attribute_name", "area_weight_ring2",
           note="do0 ring-masked area weight (attribute of the out_hres nodes of the converted graph)")
    ed.set("training.metrics.datasets.out_hres", ["z_500", "t_850", "u_850", "v_850", "tp"], note="do0 list")

    # ---- diagnostics --------------------------------------------------------------------------
    ed.set("diagnostics.log.mlflow.run_name", RUN_NAME)
    ed.set("diagnostics.log.interval", recipe["diagnostics"]["log"]["interval"], note="do0 value")
    ed.set("diagnostics.checkpoint.every_n_train_steps", {"save_frequency": 1000, "num_models_saved": -1})
    ed.set("diagnostics.checkpoint.every_n_epochs", {"save_frequency": None, "num_models_saved": 0},
           note="do0 value (an epoch is 1000 steps here; the step checkpoints already cover it)")
    ed.set("diagnostics.checkpoint.every_n_minutes", {"save_frequency": None, "num_models_saved": 0}, note="do0 value")

    if ed.errors:
        raise SystemExit("config edits failed:\n  " + "\n  ".join(ed.errors))

    # ---- model group --------------------------------------------------------------------------
    mlay = yaml.safe_load(LAYOUT_MODEL.read_text())
    upstream = yaml.safe_load((code_root / R / "config/model/graphtransformer_transport_residual_edm.yaml").read_text())
    med = Editor(upstream, code_root)
    med.set("num_channels", 512, note="do0 / lossmatch")
    med.set("keep_batch_sharded", True, note="False crashes when a model is sharded")
    med.set("residual_reference", {"out_hres": "in_lres"})
    rt = recipe["model"]["model"]["diffusion"]
    ninf = rt["inference_defaults"]["noise_scheduler"]
    med.set("model.transport.sigma_max", ninf["sigma_max"], note="do0 inference sigma range")
    med.set("model.transport.sigma_min", ninf["sigma_min"])
    med.set("model.transport.rho", ninf["rho"])
    med.set("model.transport.training_condition",
            {"distribution": "log_normal", "mean": rt["log_normal_mean"], "std": rt["log_normal_std"]},
            note="replaces the karras training draw (models/.../transport/schedules.py registers log_normal)")
    med.set("model.transport.inference_defaults.sampling_schedule",
            {"schedule_type": "karras", "sigma_max": ninf["sigma_max"], "sigma_min": ninf["sigma_min"],
             "rho": ninf["rho"], "num_steps": 30})
    med.set("model.transport.inference_defaults.sampler",
            {"sampler": "heun_window", "window": 2, "stride": 1, "S_churn": 0.0})
    for part in ("processor",):
        med.set(f"{part}.num_heads", 4)
        med.set(f"{part}.num_layers", 16)
        med.set(f"{part}.num_chunks", recipe["model"]["processor"]["num_chunks"], note="do0 (compute only)")
    med.set("encoders.0.mapper.num_heads", 4)
    med.set("decoders.0.mapper.num_heads", 4)
    for k in ("in_lres", "in_hres", "out_hres", "hidden"):
        med.set(f"node_trainable_parameters.{k}", 0)
    for k in ("data2hidden", "hidden2hidden", "hidden2data"):
        med.set(f"edge_trainable_parameters.{k}", 0)
    if med.errors:
        raise SystemExit("model group edits failed:\n  " + "\n  ".join(med.errors))
    # Keys INSIDE the values set above, which the path check cannot see: each must be read by the code.
    inner = [
        (R + "data/data_reader.py", "steps_per_sequence: int"),            # hindcast_blocks.steps_per_sequence
        (R + "data/data_reader.py", 'step: str | datetime.timedelta = "12h"'),  # hindcast_blocks.step
        (M + "transport/schedules.py", '"log_normal"'),                    # training_condition.distribution
        (M + "transport/schedules.py", "std: float"),                      # training_condition.mean / std
        (M + "samplers/transport_samplers.py", '"heun_window": EDMHeunWindowAveragedSampler'),
        (M + "samplers/transport_samplers.py", "def __init__(self, window: int, stride: int = 1"),
        (M + "samplers/transport_samplers.py", "S_churn: float = 0.0"),
        (R + "diagnostics/callbacks/weight_averaging.py", "decay: float = 0.999"),
        (R + "train/methods/base.py", "* config.training.optimization.lr"),
    ]
    missing = [f"{f}: {n!r}" for f, n in inner if not ed._code_has(f, n)]
    if missing:
        raise SystemExit("value-level keys without code evidence:\n  " + "\n  ".join(missing))
    ed.log.append(("(inner keys)", "; ".join(n for _, n in inner), "code grep, all found"))
    # sanity: the values the 17 Sep loss-match group used, where the two layouts share a key
    assert mlay["num_channels"] == 512 and mlay["processor"]["num_heads"] == 4
    return ed.cfg, med.cfg, ed, med


def dump(d: dict) -> str:
    return yaml.safe_dump(d, sort_keys=False, default_flow_style=False, width=120)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--code-root", required=True)
    p.add_argument("--sandbox", required=True)
    args = p.parse_args()
    out = Path(args.out_dir)
    (out / "model").mkdir(parents=True, exist_ok=True)
    cfg, mcfg, ed, med = build(Path(args.code_root), Path(args.sandbox))

    head = ("# GENERATED by downscaling-tools tools/multitime/make_mt2_config.py; do not edit by hand.\n"
            "# Layout: " + str(LAYOUT) + "\n# Values: " + str(RECIPE) + " (via the 17 Sep loss-match mapping)\n"
            "# Launch: anemoi-training train --config-path=<this folder> --config-name=mt2_e2_20k\n")
    targets = {out / "mt2_e2_20k.yaml": head + dump(cfg),
               out / "model/graphtransformer_transport_residual_edm_mt2.yaml":
                   "# GENERATED by make_mt2_config.py from the branch's graphtransformer_transport_residual_edm.yaml\n"
                   + dump(mcfg)}
    for path, text in targets.items():
        if path.exists() and path.read_text() != text:
            path.rename(path.with_name(path.name + ".bak-" + __import__("datetime").datetime.utcnow().strftime("%Y%m%dT%H%M%S")))
        path.write_text(text)
        print(f"wrote {path}")

    layout_text = dump(yaml.safe_load(LAYOUT.read_text())).splitlines(keepends=True)
    diff = difflib.unified_diff(layout_text, dump(cfg).splitlines(keepends=True),
                                fromfile=str(LAYOUT) + " (yaml load/dump)", tofile="mt2_e2_20k.yaml")
    (out / "mt2_e2_20k.layout_diff.txt").write_text("".join(diff))
    with open(out / "mt2_e2_20k.edits.tsv", "w") as fh:
        for prefix, editor in (("", ed), ("model:", med)):
            for path, value, ev in editor.log:
                fh.write(f"{prefix}{path}\t{value[:300]}\t{ev}\n")
    print(f"{len(ed.log)} config edits and {len(med.log)} model-group edits asserted; "
          f"diff and edit log in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
