"""Helpers shared by every stage of the pipeline.

Two project rules are enforced here rather than left to each caller.

First, nothing is ever deleted.  When a stage has to discard a file that turned
out to be wrong or incomplete, it calls move_aside(), which renames the file
into a dated sibling directory on the same filesystem.  The file still occupies
disk afterwards; that is deliberate, so that a mistake can always be examined.

Second, every artifact is written under a temporary name and renamed into place
only after it has been validated.  A reader that finds the final name can
therefore assume the file is complete, which is what makes the whole pipeline
safe to interrupt and restart.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys

DEFAULT_ROOT = "/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910"
CHECKPOINT = (
    "/home/ecm5702/scratch/eval/aifs_v1_vs_regen_20260909/regen/ckpt/aifs-ens-crps-2.0.ckpt"
)
LSM_PATH = "/home/ecm5702/scratch/eval/aifs_v1_vs_regen_20260909/regen/ckpt/lsm.grib"


def log(msg: str) -> None:
    """Print a timestamped line and flush, so SLURM logs stay readable live."""
    print(f"[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def move_aside(path: str, reason: str) -> str | None:
    """Move a file or directory out of the way instead of deleting it.

    The destination is a sibling directory named _aside_<YYYYMMDD-HHMMSS>, on
    the same filesystem, so the move is a rename and costs nothing.  The file
    keeps occupying disk; the caller is expected to say so in its report.
    """
    if not os.path.exists(path):
        return None
    stamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    aside = os.path.join(os.path.dirname(os.path.abspath(path)), f"_aside_{stamp}")
    os.makedirs(aside, exist_ok=True)
    dest = os.path.join(aside, os.path.basename(path))
    if os.path.exists(dest):
        dest = f"{dest}.{os.getpid()}"
    os.rename(path, dest)
    log(f"moved aside ({reason}): {path} -> {dest}  [still occupies disk]")
    return dest


def grib_count(path: str) -> int:
    """The number of GRIB messages in a file, or 0 if it is absent or unreadable."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    try:
        out = subprocess.run(
            ["grib_count", path], capture_output=True, text=True, timeout=1800
        )
        if out.returncode != 0:
            return 0
        return int(out.stdout.strip().split()[0])
    except Exception:
        return 0


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_json(path: str, obj) -> None:
    """Write JSON to a temporary name in the same directory, then rename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def run_cmd(cmd: list[str], timeout: int = 7200) -> tuple[int, str]:
    """Run a command, returning its exit code and its combined output."""
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def concat_files(sources: list[str], dest: str) -> None:
    """Concatenate GRIB files in the given order.  GRIB messages are self
    delimiting, so a plain byte concatenation is a valid multi-field file."""
    with open(dest, "wb") as out:
        for s in sources:
            with open(s, "rb") as f:
                shutil.copyfileobj(f, out, length=1 << 22)


def require(condition: bool, message: str) -> None:
    if not condition:
        log(f"FATAL {message}")
        sys.exit(1)
