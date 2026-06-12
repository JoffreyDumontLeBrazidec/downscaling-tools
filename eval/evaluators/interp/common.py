"""Shared helpers for the interp evaluator."""
from __future__ import annotations

import os
import re
from pathlib import Path


PERM_INTERP_ROOT = Path(os.environ.get("PERM_INTERP_ROOT", "/home/ecm5702/perm/interp"))


def resolve_ckpt_id(checkpoint_path: str | os.PathLike, eval_config: dict | None = None) -> str:
    """Derive a stable ckpt_id used as the per-checkpoint dir under ~/perm/interp/.

    Priority:
      1) eval_config["ckpt_id"] if explicitly set
      2) <first-4-hex-chars-of-parent-dir>_<step//1000>k    (matches the
         convention we've been using: 59e4_300k, 85884ee7_189k, ...)
      3) fallback to the checkpoint filename stem
    """
    if eval_config and eval_config.get("ckpt_id"):
        return str(eval_config["ckpt_id"])

    p = Path(checkpoint_path)
    parent = p.parent.name
    # Try to extract a step number from the filename
    m = re.search(r"step[_-](\d+)", p.name)
    step_part = f"{int(m.group(1)) // 1000}k" if m else "unknown"

    if re.fullmatch(r"[0-9a-fA-F]{8,}", parent):
        prefix = parent[: 8] if len(parent) >= 8 else parent
        # Trim to 4 chars when long ids look hex-uuid-y (mirrors `59e4` style)
        short = prefix[:4]
        return f"{short}_{step_part}"
    return f"{parent}_{step_part}"


def perm_run_dir(ckpt_id: str) -> Path:
    return PERM_INTERP_ROOT / ckpt_id


def perm_run_dirs(ckpt_id: str) -> list[Path]:
    """All run dirs for this checkpoint: the base <ckpt_id> dir plus any
    case-study dirs (<ckpt_id>_humberto, <ckpt_id>_amazon_precip, ...)."""
    return sorted(p for p in PERM_INTERP_ROOT.glob(f"{ckpt_id}*") if p.is_dir())
