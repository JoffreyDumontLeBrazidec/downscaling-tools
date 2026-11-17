#!/usr/bin/env python3
"""QM baseline, phase 2 (Atos CPU): per-point quantile tables of the fit window centred on month M.

Reads OUT/samples/m<MM>_<var>_<in|t>.f16 of the three months of the window (rows marked done in m<MM>_index.json) and
writes, per variable and side, the quantiles at qm_lib.PROBS of every O1280 point:
    OUT/tables/w<MM>_<var>_<in|t>.npy     float16, shape (6,599,680, 33), point-major (one point = 66 contiguous bytes)
    OUT/tables/w<MM>.json                 samples per window, months, check values
Point blocks of BLOCK points; quantiles by numpy (linear interpolation between order statistics).

    python3 qm_fit.py --out DIR --month 9 [--block 200000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qm_lib import NQ, PROBS, VARS, window_months  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--block", type=int, default=200000)
    a = ap.parse_args()
    root = Path(a.out)
    tab = root / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    months = window_months(a.month)
    rows = {}
    for mo in months:
        ix = json.loads((root / "samples" / f"m{mo:02d}_index.json").read_text())
        rows[mo] = np.flatnonzero(np.asarray(ix["done"]))
        assert rows[mo].size > 0, f"month {mo}: no extracted rows"
    n_s = sum(r.size for r in rows.values())
    rep = dict(window=a.month, months=list(months), samples=int(n_s), per_month={int(k): int(v.size) for k, v in rows.items()},
               probs=PROBS.tolist(), checks={})
    t0 = time.time()
    for v in VARS:
        for side in ("in", "t"):
            dst = tab / f"w{a.month:02d}_{v}_{side}.npy"
            if dst.exists():
                print("have", dst.name, flush=True)
                continue
            src = {mo: np.load(root / "samples" / f"m{mo:02d}_{v}_{side}.f16", mmap_mode="r") for mo in months}
            npt = next(iter(src.values())).shape[1]
            part = Path(str(dst) + ".part.npy")
            T = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16, shape=(npt, NQ))
            for p0 in range(0, npt, a.block):
                p1 = min(npt, p0 + a.block)
                s = np.concatenate([np.asarray(src[mo][rows[mo], p0:p1], np.float32) for mo in months], axis=0)
                q = np.quantile(s, PROBS, axis=0)                      # (NQ, block)
                T[p0:p1] = q.T.astype(np.float16)
            T.flush()
            del T
            os.replace(part, dst)
            T = np.load(dst, mmap_mode="r")
            med = np.asarray(T[:, NQ // 2], np.float32)
            rep["checks"][f"{v}_{side}"] = dict(median_of_medians=float(np.median(med)), p998_max=float(np.asarray(T[:, -2], np.float32).max()),
                                                monotone=bool((np.diff(np.asarray(T[::997], np.float32), axis=1) >= 0).all()))
            print(f"  w{a.month:02d} {v} {side}: {json.dumps(rep['checks'][f'{v}_{side}'])} {time.time() - t0:.0f}s", flush=True)
    (tab / f"w{a.month:02d}.json").write_text(json.dumps(rep, indent=1))
    print(f"FIT_DONE w{a.month:02d}: {n_s} samples, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
