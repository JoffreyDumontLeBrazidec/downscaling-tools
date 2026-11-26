#!/usr/bin/env python3
"""Extremes set XSET-1: are the first members of two prediction files equal bit for bit?

Used after a smoke of a defined row whose export also exists as a full row (GL512): the smoke file (2 members) must
equal members 1-2 of the row file of the same case, which proves that the row-definition path changes nothing.
Usage: python3 xset_compare_files.py <file A> <file B>      (netCDF4 + numpy; any login node)
"""
import sys

import netCDF4
import numpy as np


def load(p):
    with netCDF4.Dataset(p) as d:
        d.set_auto_mask(False)
        return np.asarray(d["y_pred"][0]), [str(s) for s in d["weather_state"][:]], np.asarray(d["point_index"][:])


def main():
    a, sa, pa = load(sys.argv[1])
    b, sb, pb = load(sys.argv[2])
    if sa != sb or not np.array_equal(pa, pb):
        print("XSET_COMPARE_FAIL variables or points differ", sa, sb, pa.size, pb.size)
        return 1
    n = min(a.shape[0], b.shape[0])
    diff = {v: float(np.abs(a[:n, :, j].astype(np.float64) - b[:n, :, j].astype(np.float64)).max()) for j, v in enumerate(sa)}
    ok = max(diff.values()) == 0.0
    print(f"XSET_COMPARE {'IDENTICAL' if ok else 'DIFFERENT'} first {n} members, {pa.size} points, max |A - B| per variable {diff}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
