"""Preflight checks P1-P5 of note s4.1. Any miss = INVALID run verdict.

``MANIFEST.json`` (written by the launch step L, note s2). Keys read here:

    {"row": "FS245_100k",
     "checkpoint": {"path": "...ckpt", "md5": "...", "ema": "EMA-2000"},
     "sampler": {...}, "precision": "bf16", "seed_base": 1234, "code_sha": "...",
     "input_stream_check": "PASS",                       # P4: staged input stream == the checkpoint's anemoi.json
     "instruments": {"MI-S25": {"pred_dir": "...", "dates": ["20250901", ...], "leads": [12, ..., 240],
                                "members": 10, "files_expected": 600}},
     "logs": {"<path of a tool log>": "<the verdict line it must contain, e.g. ALL_PASS>"}}

P1 compares ``ema``, ``sampler``, ``precision`` and ``seed_base`` with the reference row's manifest (same instrument
construction) and, with ``check_md5``, hashes the checkpoint file. P2 counts prediction files and scored lines and
looks for NaN scores. P3 reads every listed log for its own verdict line and for GUARD FAILED / TENSOR CHECK FAIL
(an exit code alone is not a pass). P4 reads ``input_stream_check``. P5 checks that REF, ENS9 and INPUT cover every
(variable, lead, date) the RUN station file covers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from eval.verdict import stats as S
from eval.verdict.gates import GateLine

BAD = ("GUARD FAILED", "TENSOR CHECK FAIL")
P1_KEYS = ("sampler", "precision", "seed_base")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def run(manifest: Path, ref_manifest: Path | None = None, check_md5: bool = False,
        scores: dict[str, str] | None = None) -> list[GateLine]:
    out: list[GateLine] = []
    try:
        m = json.loads(Path(manifest).read_text())
    except Exception as e:
        return [GateLine(p, "manifest", True, S.FAIL, note=f"MANIFEST.json not readable: {e}") for p in ("P1", "P2", "P3", "P4", "P5")]
    # P1
    ck = m.get("checkpoint", {})
    miss = [k for k in ("md5", "ema") if not ck.get(k)] + [k for k in P1_KEYS if k not in m]
    notes, st = [], S.PASS
    if miss:
        st, notes = S.FAIL, [f"manifest lacks {', '.join(miss)}"]
    if ck.get("ema") and "2000" not in str(ck["ema"]):
        notes.append(f"EMA copy {ck['ema']} (standard EMA-2000)")
    if ref_manifest:
        r = json.loads(Path(ref_manifest).read_text())
        diff = [k for k in P1_KEYS if m.get(k) != r.get(k)]
        if ck.get("ema") != r.get("checkpoint", {}).get("ema"):
            diff.append("checkpoint.ema")
        if diff:
            st = S.FAIL
            notes.append(f"differs from the reference manifest: {', '.join(diff)}")
    else:
        notes.append("no reference manifest given: sampler/precision/seed not compared")
    if check_md5:
        p = Path(ck.get("path", ""))
        if not p.is_file():
            st = S.FAIL
            notes.append(f"checkpoint {p} not readable here")
        elif _md5(p) != ck.get("md5"):
            st = S.FAIL
            notes.append("checkpoint md5 differs from the manifest")
    out.append(GateLine("P1", "checkpoint, EMA, sampler, precision, seed", True, st, note="; ".join(notes)))
    # P2
    notes, st = [], S.PASS
    for name, inst in m.get("instruments", {}).items():
        d = Path(inst.get("pred_dir", ""))
        exp = inst.get("files_expected")
        if exp is None:
            exp = len(inst.get("dates", [])) * len(inst.get("leads", []))
        have = (len(list(d.glob("predictions_*.nc"))) + len(list(d.rglob("pred_*_step*.npz")))) if d.is_dir() else 0   # XSET/box rows, MI rows
        if have < exp:
            st = S.FAIL
        notes.append(f"{name}: {have}/{exp} prediction files")
    if scores and scores.get("RUN"):
        import pandas as pd
        df = pd.read_csv(scores["RUN"].partition("::")[0])
        nan = int(df["fcrps"].isna().sum()) if "fcrps" in df else 0
        if nan:
            st = S.FAIL
        notes.append(f"RUN station file: {len(df)} lines, {nan} NaN fair CRPS")
    if not m.get("instruments"):
        st, notes = S.FAIL, notes + ["manifest lists no instrument"]
    out.append(GateLine("P2", "counts complete, no NaN", True, st, note="; ".join(notes)))
    # P3
    notes, st = [], S.PASS
    logs = m.get("logs", {})
    for path, want in logs.items():
        try:
            text = Path(path).read_text(errors="replace")
        except OSError:
            st = S.FAIL
            notes.append(f"{path}: not readable")
            continue
        bad = [b for b in BAD if b in text]
        if bad or (want and want not in text):
            st = S.FAIL
            notes.append(f"{Path(path).name}: " + (", ".join(bad) if bad else f"no '{want}' line"))
    if not logs:
        st = S.FAIL
        notes.append("manifest lists no tool log")
    out.append(GateLine("P3", "tool verdict lines", True, st, note="; ".join(notes) or f"{len(logs)} logs clean"))
    # P4
    v = str(m.get("input_stream_check", "")).upper()
    out.append(GateLine("P4", "input stream == checkpoint anemoi.json", True, S.PASS if v in ("PASS", "TRUE") else S.FAIL,
                        note=f"input_stream_check = {m.get('input_stream_check', 'absent')}"))
    # P5
    if scores:
        out.append(_coverage(scores))
    else:
        out.append(GateLine("P5", "reference rows on the same dates", True, S.FAIL, note="no station files given"))
    return out


def _coverage(scores: dict[str, str]) -> GateLine:
    from eval.verdict.reader_a import load_scores
    try:
        fr = {r: load_scores(s, "input" if r == "INPUT" else "model") for r, s in scores.items() if s}
    except SystemExit as e:
        return GateLine("P5", "reference rows on the same dates", True, S.FAIL, note=str(e))
    if "RUN" not in fr:
        return GateLine("P5", "reference rows on the same dates", True, S.FAIL, note="no RUN station file")
    key = lambda df: set(map(tuple, df[["var", "lead", "date"]].astype(str).drop_duplicates().to_numpy()))
    want = key(fr["RUN"])
    notes, st = [], S.PASS
    for r in ("REF", "ENS9", "INPUT"):
        if r not in fr:
            st = S.FAIL
            notes.append(f"{r} absent")
            continue
        lack = want - key(fr[r])
        if lack:
            st = S.FAIL
            notes.append(f"{r} lacks {len(lack)} of {len(want)} (var, lead, date)")
    return GateLine("P5", "reference rows on the same dates", True, st,
                    note="; ".join(notes) or f"{len(want)} (var, lead, date) covered by REF, ENS9, INPUT")
