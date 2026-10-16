#!/usr/bin/env python
"""Run `anemoi-training train <overrides>` in this process and print step time and peak GPU memory per rank.

The mt2_e2_20k entry point (Vera Gahlen's branch, fork claude/project-thread-n1zaq3). It adds one observer
callback to every Lightning Trainer and changes nothing in the training:
  [timing]  wall seconds per optimiser step on each rank (a line every 50 steps; at the end the median and
            mean over the steps after the first 10, validation and checkpoint pauses inside the mean only);
  [peakmem] torch.cuda.max_memory_allocated and max_memory_reserved of the rank's device, every 50 steps,
            after each validation and at exit;
  [lr]      the applied learning rate at steps 1, 500, 1000, 1001 and every 1000 steps.
Modelled on the do0 sandbox's jobs/train_with_peakmem_h9.py.

Usage (inside srun, usually behind tools/chain/chain_guard_launch.py):
  python train_with_stats.py --config-path=<dir> --config-name=mt2_e2_20k <hydra overrides...>
"""
import atexit
import os
import shutil
import statistics
import sys
import time

import pytorch_lightning as pl
import torch

RANK = os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0"))
GIB = 2**30


def _peak(tag: str) -> None:
    if torch.cuda.is_available():
        print(f"[peakmem] rank {RANK} {tag}: max_memory_allocated={torch.cuda.max_memory_allocated() / GIB:.2f} GiB "
              f"max_memory_reserved={torch.cuda.max_memory_reserved() / GIB:.2f} GiB device={torch.cuda.get_device_name()}",
              flush=True)


atexit.register(_peak, "at exit")


class StatsObserver(pl.Callback):
    def __init__(self) -> None:
        self._t_last = None
        self._dt: list[tuple[int, float]] = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if self._t_last is None:
            self._t_last = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        now = time.perf_counter()
        step = int(trainer.global_step)
        if self._t_last is not None:
            self._dt.append((step, now - self._t_last))
        self._t_last = now
        if step % 50 == 0 and self._dt:
            recent = [d for _, d in self._dt[-50:]]
            print(f"[timing] rank {RANK} step {step}: last {len(recent)} steps median {statistics.median(recent):.3f} "
                  f"s/step, mean {statistics.mean(recent):.3f} s/step", flush=True)
            _peak(f"step {step}")
        if step in (1, 500, 1000, 1001) or (step > 0 and step % 1000 == 0):
            lrs = [f"{g['lr']:.4e}" for opt in trainer.optimizers for g in opt.param_groups]
            print(f"[lr] rank {RANK} global_step {step}: {lrs}", flush=True)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        _peak(f"after validation at step {trainer.global_step}")
        self._t_last = time.perf_counter()

    def on_train_end(self, trainer, pl_module) -> None:
        d = [x for s, x in self._dt if s > 10]
        if d:
            print(f"[timing] rank {RANK} SUMMARY steps {self._dt[0][0]}-{self._dt[-1][0]} (first 10 excluded): n={len(d)} "
                  f"median {statistics.median(d):.3f} s/step, mean {statistics.mean(d):.3f} s/step", flush=True)
        _peak("at train end")


_orig_init = pl.Trainer.__init__


def _init_with_observer(self, *args, **kwargs):
    cbs = list(kwargs.get("callbacks") or [])
    cbs.append(StatsObserver())
    kwargs["callbacks"] = cbs
    _orig_init(self, *args, **kwargs)


pl.Trainer.__init__ = _init_with_observer

from anemoi.training.__main__ import main  # noqa: E402

sys.argv = [shutil.which("anemoi-training") or "anemoi-training", "train", *sys.argv[1:]]
main()
