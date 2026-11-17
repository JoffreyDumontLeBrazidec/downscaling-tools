#!/usr/bin/env python3
"""Forcing-date check (17 Nov 2026): paired comparison of two XSET-1 rows that differ only in the date of the 5 time forcings.

Row A = GL512_450k (forcings at the valid time, as in training); row B = GL512_450k_FTINIT (forcings at the start date, which
emulates the anemoi-inference downscaling runner defect input_date = date - step). Same checkpoint, inputs, seeds and points.

Per variable and lead, over the evaluation points of every date given and the 10 draws:
  dmean  mean(B - A)          drms  rms(B - A)         (paired, same seed: what the forcing date alone changes)
  biasA / biasB   mean(ensemble mean - y)               rmseA / rmseB   rms(ensemble mean - y)
y = the single 9 km field (enfo cf) of the ENS9 template; it is a reference field, not observations (station scores: run
xset_score_stations.py on both rows).
Usage: python3 ft_compare.py --a ROWDIR_A --b ROWDIR_B --dates 20240109,... [--leads 12,24,36,48,60,72] [--vars skt,2t,2d]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4
import numpy as np


def load(path):
    with netCDF4.Dataset(str(path)) as d:
        d.set_auto_mask(False)
        st = [str(s) for s in d["weather_state"][:]]
        return st, np.asarray(d["y_pred"][0], np.float64), np.asarray(d["y"][0], np.float64), np.asarray(d["lon_hres"][:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--dates", required=True)
    ap.add_argument("--leads", default="12,24,36,48,60,72")
    ap.add_argument("--vars", default="skt,2t,2d")
    args = ap.parse_args()
    dates = [s for s in args.dates.split(",") if s]
    leads = [int(s) for s in args.leads.split(",")]
    vars_ = args.vars.split(",")
    print("| var | lead | dates | points | dmean | drms | biasA | biasB | rmseA | rmseB | rmse change % |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for v in vars_:
        for L in leads:
            acc = dict(d=[], ea=[], eb=[])
            nd = 0
            for date in dates:
                fa, fb = Path(args.a) / f"predictions_{date}_step{L:03d}.nc", Path(args.b) / f"predictions_{date}_step{L:03d}.nc"
                if not (fa.exists() and fb.exists()):
                    continue
                sa, pa, y, _ = load(fa)
                sb, pb, yb, _ = load(fb)
                if v not in sa or sa != sb:
                    continue
                assert pa.shape == pb.shape and np.array_equal(y, yb), f"rows not paired for {date} +{L}"
                j = sa.index(v)
                acc["d"].append((pb[..., j] - pa[..., j]).ravel())
                acc["ea"].append(pa[..., j].mean(0) - y[:, j])
                acc["eb"].append(pb[..., j].mean(0) - y[:, j])
                nd += 1
            if not nd:
                print(f"| {v} | {L} | 0 | - | - | - | - | - | - | - | - |")
                continue
            d = np.concatenate(acc["d"]); ea = np.concatenate(acc["ea"]); eb = np.concatenate(acc["eb"])
            ra, rb = np.sqrt((ea ** 2).mean()), np.sqrt((eb ** 2).mean())
            print(f"| {v} | {L} | {nd} | {ea.size} | {d.mean():+.3f} | {np.sqrt((d ** 2).mean()):.3f} | {ea.mean():+.3f} | "
                  f"{eb.mean():+.3f} | {ra:.3f} | {rb:.3f} | {100 * (rb - ra) / ra:+.1f} |")


if __name__ == "__main__":
    main()
