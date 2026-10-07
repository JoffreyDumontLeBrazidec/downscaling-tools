#!/usr/bin/env python3
"""Gate G1 of campaign T1d stage A (2026-10-07): compare two prediction files field by field.

Use: the one-draw gate of p12m_pw30_c0 (20230826, lead 24, member 1, base seed 756) run under the fresh patched
sandbox (anemoi-core 27391c1) and under the certified venv (~/dev/.ds-260612, the runtime of the 1.2M parent's own
reads). For every weather state of y_pred: max abs difference in physical units, the field's max abs value, their
ratio, and whether the two are bit-identical. Also checks that the truth (y), the input (x_interp when present) and the
coordinates are identical (same bundles, same box). PASS when every state is bit-identical or within --tol relative
(max |a - b| / max |b|, default 1e-3). Also usable for the member-1 identity check of the full run (--member).
Usage: python g1_compare.py <file A> <file B> [--member 1] [--tol 1e-3]
"""
import argparse
import sys

import netCDF4
import numpy as np


def states(d):
    return [str(s) for s in d["weather_state"][:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--member", type=int, default=None, help="compare this member label only (default: all common)")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    ok = True
    with netCDF4.Dataset(args.a) as da, netCDF4.Dataset(args.b) as db:
        da.set_auto_mask(False)
        db.set_auto_mask(False)
        for c in ("lat_hres", "lon_hres"):
            same = np.array_equal(da[c][:], db[c][:])
            ok &= same
            print(f"coord {c}: identical={same}")
        sa, sb = states(da), states(db)
        if sa != sb:
            print(f"FAIL weather states differ: {sa} vs {sb}")
            return 1
        ma = [int(m) for m in da["ensemble_member"][:]]
        mb = [int(m) for m in db["ensemble_member"][:]]
        mems = [m for m in ma if m in mb and (args.member is None or m == args.member)]
        if not mems:
            print(f"FAIL no common member (A {ma}, B {mb}, asked {args.member})")
            return 1
        for v in ("y", "x_interp"):
            if v in da.variables and v in db.variables:
                same = np.array_equal(da[v][:], db[v][:], equal_nan=True)
                ok &= same
                print(f"{v}: identical={same}")
        for att in ("sampling_config_json", "checkpoint_id", "checkpoint_path"):
            if att in da.ncattrs() and att in db.ncattrs():
                print(f"attr {att}: identical={da.getncattr(att) == db.getncattr(att)}")
        print(f"members compared: {mems}")
        print(f"{'state':10s} {'max|A-B|':>12s} {'max|B|':>12s} {'relative':>10s} {'bitwise':>8s}")
        for i, s in enumerate(sa):
            xa = np.stack([np.asarray(da["y_pred"][0, ma.index(m), :, i], dtype=np.float64) for m in mems])
            xb = np.stack([np.asarray(db["y_pred"][0, mb.index(m), :, i], dtype=np.float64) for m in mems])
            d = float(np.nanmax(np.abs(xa - xb)))
            ref = float(np.nanmax(np.abs(xb))) or 1.0
            bit = bool(np.array_equal(xa, xb, equal_nan=True))
            good = bit or d / ref <= args.tol
            ok &= good
            print(f"{s:10s} {d:12.6g} {ref:12.6g} {d / ref:10.3g} {str(bit):>8s}{'' if good else '   <-- above tol'}")
    print("G1 PASS" if ok else "G1 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
