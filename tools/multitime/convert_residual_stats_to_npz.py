#!/usr/bin/env python3
"""Convert a lane residual-statistics .npy into the .npz that Vera Gahlen's branch reads.

Our lane stores residual statistics (statistics of target - interpolated input, per
variable) as a pickled .npy holding ``{stat: {variable: float}}`` with stat in
``mean, stdev, maximum, minimum`` (e.g. ``o1280_dict_0_72.npy``: 68 variables, the
o1280 outputs without tp and cp).

Vera's ``feat/spatial_downscaler`` branch reads them with
``anemoi.training.data.residual_statistics.load_residual_statistics``: an .npz with keys
``mean``, ``minimum``, ``maximum``, ``stdev``, each a 0-d object array wrapping a
``{variable: float}`` mapping. Only the target's prognostic variables are looked up;
diagnostics (tp, cp in the multi-time E2 config) and forcings get identity statistics,
so they need no entry.

Usage:
  python convert_residual_stats_to_npz.py IN.npy OUT.npz [--check-variables v1,v2,...]
      [--check-residual-variables v1,...] [--compare OTHER.npy]

With --check-variables (the target dataset's full variable list, in its order) the
script reloads OUT.npz through load_residual_statistics (anemoi.training must import)
and reports the round-trip maximum absolute and relative error for every residual
variable; --compare reports the differences against another .npy on shared variables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

KEYS = ("mean", "minimum", "maximum", "stdev")


def read_npy(path: str) -> dict[str, dict[str, float]]:
    obj = np.load(path, allow_pickle=True)
    if obj.dtype != object or obj.shape != ():
        raise TypeError(f"{path}: expected a 0-d object array holding a dict, got dtype {obj.dtype} shape {obj.shape}")
    stats = obj.item()
    if not isinstance(stats, dict) or set(stats) != set(KEYS):
        raise KeyError(f"{path}: expected exactly the keys {sorted(KEYS)}, got {sorted(stats) if isinstance(stats, dict) else type(stats)}")
    variables = list(stats["mean"])
    for key in KEYS:
        if list(stats[key]) != variables:
            raise KeyError(f"{path}: statistic {key!r} does not list the same variables as 'mean'")
        for var, value in stats[key].items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{path}: {key}[{var}] = {value} is not finite")
    return {key: {var: float(val) for var, val in stats[key].items()} for key in KEYS}


def write_npz(stats: dict[str, dict[str, float]], path: str) -> None:
    arrays = {}
    for key in KEYS:
        holder = np.empty((), dtype=object)
        holder[()] = dict(stats[key])
        arrays[key] = holder
    np.savez(path, **arrays)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npy")
    parser.add_argument("npz")
    parser.add_argument("--check-variables", default=None, help="target dataset variables, comma-separated, in order")
    parser.add_argument("--check-residual-variables", default=None,
                        help="residual (prognostic) variables; default: every variable of the .npy")
    parser.add_argument("--compare", default=None, help="another .npy to compare on shared variables")
    args = parser.parse_args()

    stats = read_npy(args.npy)
    variables = list(stats["mean"])
    print(f"read {args.npy}: {len(variables)} variables, statistics {list(KEYS)}")
    if Path(args.npz).exists():
        print(f"refusing to overwrite {args.npz}; move it aside first")
        return 2
    write_npz(stats, args.npz)
    print(f"wrote {args.npz}")

    if args.check_variables:
        from anemoi.training.data.residual_statistics import load_residual_statistics

        target_vars = args.check_variables.split(",")
        residual_vars = args.check_residual_variables.split(",") if args.check_residual_variables else variables
        loaded = load_residual_statistics(args.npz, target_vars, residual_vars)
        worst_abs, worst_rel, n = 0.0, 0.0, 0
        for key in KEYS:
            for i, var in enumerate(target_vars):
                if var not in residual_vars:
                    continue
                ref = stats[key][var]
                got = float(loaded[key][i])
                worst_abs = max(worst_abs, abs(got - ref))
                worst_rel = max(worst_rel, abs(got - ref) / max(abs(ref), 1e-30))
                n += 1
        identity = {var: {key: float(loaded[key][i]) for key in KEYS}
                    for i, var in enumerate(target_vars) if var not in residual_vars}
        print(f"round trip through load_residual_statistics: {n} values ({n // len(KEYS)} variables x {len(KEYS)}), "
              f"max abs error {worst_abs:.3e}, max rel error {worst_rel:.3e} (float32 storage)")
        print(f"non-residual variables given identity statistics: {identity}")
        missing = sorted(set(residual_vars) - set(variables))
        print(f"residual variables missing from the file: {missing}")

    if args.compare:
        other = np.load(args.compare, allow_pickle=True).item()
        shared = [v for v in variables if v in other.get("mean", {})]
        diff = max(abs(float(other[key][v]) - stats[key][v]) for key in KEYS for v in shared)
        extra = sorted(set(other.get("mean", {})) - set(variables))
        print(f"compare with {args.compare}: {len(shared)} shared variables, max abs difference {diff:.3e}; "
              f"only in the other file: {extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
