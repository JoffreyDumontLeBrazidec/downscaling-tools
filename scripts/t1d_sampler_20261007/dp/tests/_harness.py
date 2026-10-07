"""Test harness: the REAL capture/writer code of interp/tools/trajectory.py (extracted by
ast, so the heavy imports of that module are not needed) and the fork's samplers module
(imported normally when anemoi-models is installed, else loaded from a file with a stub
for its one anemoi import)."""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
import logging
import os
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
TRAJ = REPO / "interp" / "tools" / "trajectory.py"
FORK_CANDIDATES = [
    os.environ.get("T1D_FORK_SAMPLERS", ""),
    str(REPO.parent / "anemoi-core" / "models" / "src" / "anemoi" / "models" / "samplers" / "diffusion_samplers.py"),
    "/home/claude/wt/ac-ncf9gh/models/src/anemoi/models/samplers/diffusion_samplers.py",
]


def load_trajectory_funcs(names=("capture_denoiser", "_write_trajectory_states")):
    import torch
    tree = ast.parse(TRAJ.read_text())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "TRAJECTORY_STATES_FORMAT"
                                                  for t in node.targets):
            keep.append(node)
    mod = ast.Module(body=keep, type_ignores=[])
    ns = {"np": np, "torch": torch, "os": os, "Path": Path, "contextlib": contextlib, "json": json,
          "LOGGER": logging.getLogger("trajectory-under-test")}
    exec(compile(mod, str(TRAJ), "exec"), ns)
    missing = [n for n in names if n not in ns]
    if missing:
        raise RuntimeError(f"not found in {TRAJ}: {missing}")
    return ns


def load_fork_samplers():
    try:
        from anemoi.models.samplers import diffusion_samplers as ds  # type: ignore
        if "custom" in ds.NOISE_SCHEDULERS:
            return ds, ds.__file__
    except Exception:
        pass
    for cand in FORK_CANDIDATES:
        if cand and Path(cand).is_file():
            for mname in ("anemoi", "anemoi.models", "anemoi.models.distributed"):
                sys.modules.setdefault(mname, types.ModuleType(mname))
            shapes = types.ModuleType("anemoi.models.distributed.shapes")
            shapes.DatasetShardSizes = dict
            sys.modules.setdefault("anemoi.models.distributed.shapes", shapes)
            spec = importlib.util.spec_from_file_location("fork_diffusion_samplers", cand)
            ds = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ds)
            if "custom" not in ds.NOISE_SCHEDULERS:
                raise RuntimeError(f"{cand} has no custom scheduler (not the patched fork)")
            return ds, cand
    raise RuntimeError("fork diffusion_samplers.py not found; set T1D_FORK_SAMPLERS")
