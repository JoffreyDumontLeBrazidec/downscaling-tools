"""Launch manifest ``MANIFEST.json`` of a run (note s2, schema in ``preflight.py``), written by the launch step L from the
files on disk so that no value is typed by hand. B3, 19 Nov 2026.

    python -m eval.cli verdict --reader M --out RUN_DIR --row GL512_450k --ckpt PATH.ckpt --ema EMA-2000 \
        --sampler "heun, 30 steps" --precision bf16 --seed-base 1234 --code-sha SHA --input-stream-check PASS \
        --instrument MI-S25=/path/rows/GL512_450k --dates 20250901..20250930 --leads 12..240:12 --members 10 \
        --log /path/job.out=ALL_PASS

Counts: MI rows hold one ``pred_<date>_step<LLL>.npz`` per date and lead (10 members inside); XSET-1 and box rows hold
``predictions_*.nc``. ``files_expected`` = dates x leads unless given.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from eval.verdict.preflight import _md5


def expand_dates(spec: str) -> list[str]:
    out = []
    for part in spec.replace(",", " ").split():
        if ".." in part:
            a, b = (dt.datetime.strptime(x, "%Y%m%d").date() for x in part.split(".."))
            out += [(a + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range((b - a).days + 1)]
        else:
            out.append(part)
    return out


def expand_leads(spec: str) -> list[int]:
    if ".." in spec:
        rng, _, step = spec.partition(":")
        a, b = (int(x) for x in rng.split(".."))
        return list(range(a, b + 1, int(step or 12)))
    return [int(x) for x in spec.replace(",", " ").split()]


def build(row: str, ckpt: str, ema: str, sampler: str, precision: str, seed_base: int, code_sha: str,
          input_stream_check: str, instruments: dict[str, str], dates: list[str], leads: list[int], members: int,
          logs: dict[str, str], files_expected: int | None = None, hash_ckpt: bool = True) -> dict:
    p = Path(ckpt)
    return {
        "row": row,
        "checkpoint": {"path": str(p), "md5": _md5(p) if hash_ckpt and p.is_file() else "", "ema": ema},
        "sampler": sampler, "precision": precision, "seed_base": seed_base, "code_sha": code_sha,
        "input_stream_check": input_stream_check,
        "instruments": {n: {"pred_dir": d, "dates": dates, "leads": leads, "members": members,
                            "files_expected": files_expected if files_expected is not None else len(dates) * len(leads)}
                        for n, d in instruments.items()},
        "logs": logs,
    }


def write(path: Path, manifest: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    return path
