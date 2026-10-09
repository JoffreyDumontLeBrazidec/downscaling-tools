#!/usr/bin/env python
"""Run a training entry point with a guard that refuses to train from the wrong step.

    srun python chain_guard_launch.py --expect-step N [--root R] -- <entry> [args...]

<entry> is a Python script (`train_with_peakmem_h9.py`), a console script on PATH (`anemoi-training`), or
`-m <module>`. Before running it, Lightning's `Trainer.fit` is wrapped so that every trainer gets one more
callback. At `on_train_start`, after Lightning has restored the checkpoint, the callback compares
`trainer.global_step` with N, the step `chain_state.py resolve` read from the run's `last.ckpt` (0 for a fresh
start). Any difference stops the job on every rank before the first optimiser step. This catches what the
resolver cannot see from outside: a lane YAML that forces `load_weights_only` or a `warm_start`, or anemoi's
MLflow dry-run path that silently starts from scratch.

With --root, rank 0 writes `<root>/.chain_guard/<job>.ok` once the step matches, so the job body can prove the
guard ran. No change to anemoi; works with pytorch_lightning and lightning.pytorch.
"""
import argparse
import os
import runpy
import shutil
import sys


class ResumeStepMismatch(RuntimeError):
    pass


def _rank():
    for k in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if os.environ.get(k) is not None:
            return int(os.environ[k])
    return 0


def _make_guard(base, expect_step, root):
    class ChainResumeGuard(base):
        def on_train_start(self, trainer, pl_module):
            step = int(trainer.global_step)
            if step != expect_step:
                msg = ("[chain-guard] REFUSED: training starts at global_step {} but this segment expected {} "
                       "(from the run's last.ckpt; 0 = fresh start). The checkpoint did not load as a full "
                       "resume. Stopping before the first optimiser step.").format(step, expect_step)
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
                raise ResumeStepMismatch(msg)
            sys.stderr.write("[chain-guard] ok: training starts at global_step {} as expected\n".format(step))
            if root and _rank() == 0:
                d = os.path.join(root, ".chain_guard")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "{}.ok".format(os.environ.get("SLURM_JOB_ID", "local"))), "w") as fh:
                    fh.write("{}\n".format(step))

    return ChainResumeGuard


def install(expect_step, root=None):
    """Wrap Trainer.fit of every Lightning flavour that imports, so each fit carries the guard."""
    patched = []
    for mod_name in ("pytorch_lightning", "lightning.pytorch"):
        try:
            mod = __import__(mod_name, fromlist=["Trainer", "Callback"])
        except ImportError:
            continue
        trainer_cls, callback_cls = mod.Trainer, mod.Callback
        if getattr(trainer_cls.fit, "_chain_guarded", False):
            patched.append(mod_name)
            continue
        guard_cls = _make_guard(callback_cls, expect_step, root)
        orig_fit = trainer_cls.fit

        def fit(self, *args, _orig=orig_fit, _guard=guard_cls, **kwargs):
            if not any(isinstance(cb, _guard) for cb in self.callbacks):
                self.callbacks.append(_guard())
            return _orig(self, *args, **kwargs)

        fit._chain_guarded = True
        trainer_cls.fit = fit
        patched.append(mod_name)
    if not patched:
        raise SystemExit("[chain-guard] neither pytorch_lightning nor lightning.pytorch imports; refusing to run unguarded")
    return patched


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: chain_guard_launch.py --expect-step N [--root R] -- <entry> [args...]")
    cut = argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--expect-step", type=int, required=True)
    p.add_argument("--root", default=None)
    a = p.parse_args(argv[:cut])
    entry = argv[cut + 1:]
    if not entry:
        raise SystemExit("[chain-guard] no entry point after --")
    patched = install(a.expect_step, a.root)
    if _rank() == 0:
        sys.stderr.write("[chain-guard] armed on {} : expect global_step {}\n".format(", ".join(patched), a.expect_step))
    if entry[0] == "-m":
        sys.argv = [entry[1]] + entry[2:]
        runpy.run_module(entry[1], run_name="__main__", alter_sys=True)
        return
    path = entry[0] if os.path.isfile(entry[0]) else shutil.which(entry[0])
    if not path:
        raise SystemExit("[chain-guard] entry point not found: {}".format(entry[0]))
    sys.argv = [path] + entry[1:]
    if path.endswith(".py"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(path)))  # as `python script.py` would
    runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    main()
