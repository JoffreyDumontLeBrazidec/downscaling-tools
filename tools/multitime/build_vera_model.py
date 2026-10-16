#!/usr/bin/env python3
"""Build Vera Gahlen's downscaler from an mt2 config on the CPU, optionally warm-started, and save it.

Used for the mt2_e2_20k warm start (stages b and c) and its checks:

- with no warm start it writes a "container": a checkpoint of the freshly initialised model
  ({"state_dict", "hyper_parameters": {"data_indices", "metadata"}, "pytorch-lightning_version",
  "global_step": 0, "epoch": 0}), the shape template that transplant_to_vera.py (single slot) and
  tile_warm_start.py (two slots) fill, and also the randomly initialised contrast of the forward
  check;
- with --warm-start FILE it builds the model through the branch's own weights-only warm start
  (system.input.warm_start=FILE, training.load_weights_only=True, training.transfer_learning=False,
  i.e. LightningModule.load_from_checkpoint(strict=False), as the training job will) and, with
  --verify, checks that every tensor of FILE's state_dict landed in the model unchanged and lists
  the model tensors FILE does not provide.

Either way --interface writes the pickled AnemoiModelInterface for warm_start_fwdcheck.py.

The model is built exactly as AnemoiTrainer.model builds it (same composition, same datamodule,
graph, statistics), with MLflow off and the graph loaded on the CPU.

Usage (Vera x86 venv, never its activate.sh):
  PYTHONPATH=<core>/models/src:<core>/training/src:<core>/graphs/src:/home/ecm5702/dev/anemoi-datasets/src \
  OUTPUT=/tmp/... python build_vera_model.py --config-dir <S>/configs --config-name mt2_e2_20k \
      [--override task.input_offsets=[0h] ...] [--warm-start F --verify] [--container C] [--interface I]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-dir", required=True)
    p.add_argument("--config-name", default="mt2_e2_20k")
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--warm-start", default=None)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--container", default=None, help="write the model as a container checkpoint here")
    p.add_argument("--interface", default=None, help="write the pickled AnemoiModelInterface here")
    p.add_argument("--keys", default=None, help="write the state_dict keys and shapes here (json)")
    p.add_argument("--seed", type=int, default=756)
    args = p.parse_args()
    for path in (args.container, args.interface):
        if path and Path(path).exists():
            log(f"refusing: {path} exists (move it aside first)")
            return 2

    os.environ.setdefault("ANEMOI_BASE_SEED", str(args.seed))
    os.environ.setdefault("ANEMOI_GRAPHS_FORCE_CPU", "1")
    import torch
    import pytorch_lightning as pl
    import anemoi.datasets
    import anemoi.models
    import anemoi.training

    log(f"anemoi.training {anemoi.training.__file__}; anemoi.models {anemoi.models.__file__}; "
        f"anemoi.datasets {anemoi.datasets.__file__}; torch {torch.__version__}; pl {pl.__version__}")
    from hydra import compose, initialize_config_dir

    from anemoi.training.train.train import AnemoiTrainer

    overrides = ["diagnostics.log.mlflow.enabled=False", "training.run_id=null", "training.fork_run_id=null",
                 *args.override]
    if args.warm_start:
        overrides += [f"system.input.warm_start={args.warm_start}", "training.load_weights_only=True",
                      "training.transfer_learning=False"]
    with initialize_config_dir(config_dir=os.path.abspath(args.config_dir), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    log(f"composed {args.config_name} with {overrides}")
    torch.manual_seed(args.seed)
    trainer = AnemoiTrainer(cfg)
    model = trainer.model
    net = model.model.model
    log(f"built {type(model).__name__} / {type(net).__name__}: n_step_input {model.n_step_input}, "
        f"n_step_output {model.n_step_output}")
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    roots = ("encoder", "decoder", "processor", "noise_cond_mlp", "noise_embedder")
    net_keys = [k for k in sd if k.startswith("model.model.") and k.split(".")[2] in roots]
    log(f"state_dict: {len(sd)} tensors, {len(net_keys)} network tensors "
        f"({sum(sd[k].numel() for k in net_keys)} parameters)")
    for k in net_keys:
        if ".proc." in k and ".proc.0." not in k and "processor" in k:
            continue
        log(f"  {k} {tuple(sd[k].shape)}")
    if args.keys:
        Path(args.keys).write_text(json.dumps({k: list(v.shape) for k, v in sd.items()}, indent=0))

    di = model.data_indices
    for name, ix in di.items():
        log(f"data indices {name}: model.input {len(ix.model.input)} {list(ix.model.input.name_to_index)}")
        log(f"data indices {name}: model.output {len(ix.model.output)} {list(ix.model.output.name_to_index)}")

    result = {"config": args.config_name, "overrides": overrides, "n_tensors": len(sd), "n_network": len(net_keys)}
    if args.warm_start and args.verify:
        ws = torch.load(args.warm_start, map_location="cpu", weights_only=False)["state_dict"]
        equal, differ, missing_in_model = 0, [], []
        for k, v in ws.items():
            if k not in sd:
                missing_in_model.append(k)
            elif tuple(v.shape) != tuple(sd[k].shape):
                differ.append((k, "shape", tuple(v.shape), tuple(sd[k].shape)))
            elif torch.equal(v.to(sd[k].dtype), sd[k]):
                equal += 1
            else:
                differ.append((k, "values", float((v.double() - sd[k].double()).abs().max())))
        not_provided = [k for k in sd if k not in ws]
        net_differ = [d for d in differ if d[0] in net_keys]
        result.update({"warm_start": args.warm_start, "equal": equal, "differ": differ,
                       "warm_start_keys_not_in_model": missing_in_model, "model_keys_not_in_warm_start": not_provided,
                       "network_tensors_equal": sum(1 for k in net_keys if k in ws and torch.equal(ws[k].to(sd[k].dtype), sd[k]))})
        log(f"warm start check: {equal} of {len(ws)} tensors land unchanged; {len(differ)} differ "
            f"({len(net_differ)} of them network tensors); {result['network_tensors_equal']} of {len(net_keys)} network "
            f"tensors equal the warm-start file; {len(missing_in_model)} file keys not in the model; "
            f"{len(not_provided)} model keys not in the file")
        for d in differ[:40]:
            log(f"  differs: {d}")
        log(f"  model keys not in the file: {not_provided[:20]}")
        log("WARM START LOAD " + ("OK" if not net_differ and result["network_tensors_equal"] == len(net_keys) else "FAILED"))

    if args.container:
        hp = {"data_indices": di}
        try:
            hp["metadata"] = model.hparams.get("metadata")
        except Exception as exc:  # noqa: BLE001
            log(f"no metadata in hparams ({exc})")
        torch.save({"state_dict": sd, "hyper_parameters": hp, "pytorch-lightning_version": pl.__version__,
                    "global_step": 0, "epoch": 0,
                    "provenance": {"built_by": "build_vera_model.py", "config": args.config_name, "overrides": overrides,
                                   "seed": args.seed, "date": datetime.datetime.utcnow().isoformat() + "Z"}},
                   args.container)
        log(f"wrote container {args.container}")
    if args.interface:
        torch.save(model.model, args.interface)
        log(f"wrote interface {args.interface}")
    print("RESULT " + json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in result.items()}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
