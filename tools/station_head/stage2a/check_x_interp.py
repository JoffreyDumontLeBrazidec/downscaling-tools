"""Prove that a stage 2a prediction file was fed the right input.

The bundles of stage 2a are built from a MARS cache of GRIB files, while the arm A
training data is a set of anemoi zarr stores built from the same MARS request. If
the bundle route is right, the low-resolution input the model actually saw must be
the store's own numbers.

The check has two parts.

First, an exact one. The prediction file carries `x`, the low-resolution input on
the 421,120 points of the O320 grid, for each member and each of the ten saved
weather states. The AIFS ENS version 2 input store carries the same field at the
synthetic index of this case. Those two are compared value by value; agreement to
float32 rounding is what proves the route, and any real discrepancy would mean the
GRIB extraction picked the wrong date, time, member or lead.

Second, a consistency one for `x_interp`, which is the same input interpolated to
the 6,599,680 points of the O1280 output grid and which is the control the station
head is measured against. The harness interpolates with its own precomputed
operator, so this script does not try to reproduce it exactly. It checks instead
that x_interp is a genuine interpolation of the store field: at every output point
it must lie inside the range of the four nearest O320 values, which is true of any
convex interpolation and false of a wrong or shifted field, and it compares
x_interp with a nearest-neighbour reconstruction of the store field to show how far
a smooth interpolation sits from the nearest value.
"""
from __future__ import annotations

import argparse
import json

import netCDF4
import numpy as np
import zarr

INPUT_STORE = ("/home/ecm5702/scratch/data/anemoi_datasets_aifsens2_20260907/"
               "downscaling-ai-pf-enfo-0001-mars-o320-2026-2026-12h-6h-v1-aifsens2.zarr")


def _unit_xyz(lon_deg, lat_deg):
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", required=True)
    ap.add_argument("--synthetic-index", type=int, required=True)
    ap.add_argument("--interp-sample", type=int, default=200000,
                    help="how many O1280 points to test for the x_interp checks")
    args = ap.parse_args()

    ds = netCDF4.Dataset(args.prediction)
    states = [s.strip() for s in str(ds.getncattr("output_weather_states")).split(",")]
    members = np.asarray(ds.variables["ensemble_member"][:]).reshape(-1)
    print("prediction file: " + args.prediction)
    print("init_date=%s lead=%s members=%s" % (ds.getncattr("init_date"),
                                               ds.getncattr("lead_step_hours"),
                                               ",".join(str(m) for m in members)))
    print("weather states: " + ",".join(states))

    g = zarr.open(INPUT_STORE, mode="r")
    attrs = json.load(open(INPUT_STORE + "/.zattrs"))
    variables = attrs["variables"]
    fake = attrs.get("fake_forecasts", {})
    dates = list(fake.keys())
    if args.synthetic_index < len(dates):
        key = sorted(dates)[args.synthetic_index]
        print("store fake_forecasts entry for index %d: %s -> %s"
              % (args.synthetic_index, key, fake[key]))

    print("\n--- part one: x against the store, exact ---")
    worst = 0.0
    for vi, state in enumerate(states):
        if state not in variables:
            print("  %-6s not in the input store, skipped" % state)
            continue
        si_var = variables.index(state)
        for mi, member in enumerate(members):
            xp = np.asarray(ds.variables["x"][0, mi, :, vi], dtype=np.float64)
            xs = np.asarray(g["data"][args.synthetic_index, si_var, int(member) - 1, :],
                            dtype=np.float64)
            if xp.shape != xs.shape:
                print("  %-6s member %d SHAPE MISMATCH %s vs %s"
                      % (state, member, xp.shape, xs.shape))
                continue
            d = np.abs(xp - xs)
            scale = max(1.0, float(np.max(np.abs(xs))))
            rel = float(np.max(d)) / scale
            worst = max(worst, rel)
            print("  %-6s member %2d  max|x - store| = %.6g   relative to field scale = %.3g   "
                  "identical = %s" % (state, member, float(np.max(d)), rel,
                                      bool(np.array_equal(xp.astype(np.float32),
                                                          xs.astype(np.float32)))))
    print("worst relative difference over all states and members: %.3g" % worst)

    print("\n--- part two: x_interp as an interpolation of the store field ---")
    from scipy.spatial import cKDTree

    lon_l = np.asarray(ds.variables["lon_lres"][:])
    lat_l = np.asarray(ds.variables["lat_lres"][:])
    lon_h = np.asarray(ds.variables["lon_hres"][:])
    lat_h = np.asarray(ds.variables["lat_hres"][:])
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(lon_h.size, size=min(args.interp_sample, lon_h.size),
                                replace=False))
    tree = cKDTree(_unit_xyz(lon_l, lat_l))
    _, nn4 = tree.query(_unit_xyz(lon_h[sample], lat_h[sample]), k=4)

    for vi, state in enumerate(states):
        if state not in variables:
            continue
        si_var = variables.index(state)
        member = int(members[0])
        xs = np.asarray(g["data"][args.synthetic_index, si_var, member - 1, :], dtype=np.float64)
        xi = np.asarray(ds.variables["x_interp"][0, 0, :, vi], dtype=np.float64)[sample]
        neigh = xs[nn4]
        lo, hi = neigh.min(axis=1), neigh.max(axis=1)
        tol = 1e-6 * np.maximum(1.0, np.abs(hi))
        inside = float(np.mean((xi >= lo - tol) & (xi <= hi + tol)))
        nearest = neigh[:, 0]
        d = np.abs(xi - nearest)
        spread = np.median(hi - lo)
        print("  %-6s inside the range of the four nearest O320 values: %.4f of %d points; "
              "median |x_interp - nearest| = %.4g (median neighbourhood spread %.4g)"
              % (state, inside, sample.size, float(np.median(d)), float(spread)))
    ds.close()


if __name__ == "__main__":
    main()
