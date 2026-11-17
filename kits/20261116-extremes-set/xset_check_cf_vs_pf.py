#!/usr/bin/env python3
"""Check of the reference files of XSET-1: does the single 9 km control field (y, enfo cf) hold as many tail values as the
perturbed members (y_pred, enfo pf 1-10) on the same evaluation points? Station scores showed fewer exceedances for the
control (docs note s7.1). Prints, pooled over the given files, the share of points above each threshold for the control,
for each member, and the member mean and range; also the mean of each field (a units or accumulation fault would show).

    python3 xset_check_cf_vs_pf.py W/rows/ENS9 [--every 6]        (--every N: one file in N, default all 252)
"""
import argparse
from pathlib import Path

import netCDF4
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("pred")
ap.add_argument("--every", type=int, default=1)
a = ap.parse_args()
files = sorted(Path(a.pred).glob("predictions_*_step*.nc"))[::a.every]
WS, TP = (10.0, 15.0, 20.0, 25.0), (20.0, 50.0, 100.0)
cnt = {("ws", t): np.zeros(11) for t in WS} | {("tp", t): np.zeros(11) for t in TP}
tot, n = np.zeros((11, 2)), 0
for f in files:
    with netCDF4.Dataset(str(f)) as d:
        d.set_auto_mask(False)
        ws_ = [str(s) for s in d["weather_state"][:]]
        y, yp = np.asarray(d["y"][0], np.float64), np.asarray(d["y_pred"][0], np.float64)
    x = np.concatenate([y[None], yp])                                   # (11, points, states): 0 = control
    ws = np.hypot(x[..., ws_.index("10u")], x[..., ws_.index("10v")])
    tp = x[..., ws_.index("tp")] * 1000.0
    for t in WS:
        cnt[("ws", t)] += (ws > t).sum(1)
    for t in TP:
        cnt[("tp", t)] += (tp > t).sum(1)
    tot[:, 0] += ws.sum(1); tot[:, 1] += tp.sum(1); n += y.shape[0]
print(f"{len(files)} files, {n} point-times")
print("mean wind [m/s]: control %.4f, members %.4f (%.4f..%.4f)" % (tot[0, 0] / n, tot[1:, 0].mean() / n, tot[1:, 0].min() / n, tot[1:, 0].max() / n))
print("mean 12 h rain [mm]: control %.4f, members %.4f (%.4f..%.4f)" % (tot[0, 1] / n, tot[1:, 1].mean() / n, tot[1:, 1].min() / n, tot[1:, 1].max() / n))
for (v, t), c in cnt.items():
    m = c[1:]
    print(f"{v} > {t:g}: control {c[0] / n:.3e}, members mean {m.mean() / n:.3e} (min {m.min() / n:.3e}, max {m.max() / n:.3e}), "
          f"control / member mean {c[0] / m.mean():.3f}, members below control {int((m < c[0]).sum())} of 10")
