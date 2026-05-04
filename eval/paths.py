from __future__ import annotations

import os
from pathlib import Path

EVAL_ROOT_ENV = "EVAL_ROOT"
CANONICAL_EVAL_ROOT = "/home/ecm5702/scratch/eval"
LEGACY_EVAL_ROOT = "/home/ecm5702/perm/eval"


def default_eval_root() -> str:
    return os.environ.get(EVAL_ROOT_ENV, CANONICAL_EVAL_ROOT)


def resolve_eval_root(path: str | Path | None = None) -> Path:
    raw = path if path is not None else default_eval_root()
    return Path(raw).expanduser()


def default_scoreboard_dir() -> str:
    return str(resolve_eval_root() / "scoreboards")


def default_scoreboard_path(name: str) -> str:
    return str(resolve_eval_root() / "scoreboards" / name)


def default_slurm_dir() -> str:
    return str(resolve_eval_root() / "_slurm")


def default_slurm_output_pattern() -> str:
    return str(resolve_eval_root() / "_slurm" / "%x_%j.out")


DEFAULT_EVAL_ROOT = default_eval_root()
DEFAULT_SCOREBOARD_DIR = default_scoreboard_dir()
DEFAULT_SLURM_DIR = default_slurm_dir()
DEFAULT_SLURM_OUTPUT_PATTERN = default_slurm_output_pattern()
