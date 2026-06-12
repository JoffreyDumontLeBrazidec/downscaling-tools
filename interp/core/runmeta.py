"""Run metadata: ckpt ids + run_meta.json (what makes runs comparable later).

Every tool driver calls :func:`write_run_meta` into its output dir so
``python -m interp compare`` can line runs up without archaeology.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path


def ckpt_id_from_path(checkpoint_path: str) -> str:
    """Build the ckpt_id used in the perm output convention.

    parent-dir "59e40596..." + filename ".../step_300000.ckpt"
        -> "59e4_300k"  (first 4 hex chars + "_" + step//1000 + "k")
    """
    p = Path(checkpoint_path)
    hex4 = p.parent.name[:4]
    step = None
    stem = p.stem
    if "step_" in stem:
        tail = stem.split("step_")[-1]
        digits = "".join(c for c in tail if c.isdigit())
        if digits:
            step = int(digits)
    if step is None:
        for tok in stem.replace("-", "_").split("_"):
            if tok.startswith("step") and tok[4:].isdigit():
                step = int(tok[4:])
    if step is None:
        return hex4
    return f"{hex4}_{step // 1000}k"


def _git_sha() -> str | None:
    try:
        repo = Path(__file__).resolve().parent.parent.parent
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=repo, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def write_run_meta(output_dir, tool: str, args, extra: dict | None = None) -> Path:
    """Write <output_dir>/run_meta.json describing this run.

    `args` is the parsed argparse Namespace (serialized via vars()).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = getattr(args, "checkpoint", None)
    meta = {
        "tool": tool,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_sha": _git_sha(),
        "checkpoint": ckpt,
        "ckpt_id": ckpt_id_from_path(ckpt) if ckpt else None,
        "args": {k: v for k, v in vars(args).items()},
    }
    if extra:
        meta.update(extra)
    path = out / "run_meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path
