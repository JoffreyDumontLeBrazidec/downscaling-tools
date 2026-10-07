"""Cost matrices of one-step errors between dense reference levels (T1d, CPU).

For every draw (trajectory_states_s<seed>.npz from `trajectory --save-trajectory-states`)
and every ordered pair of dense levels i < j (sigma_i > sigma_j), the error of ONE step
from the reference state x_i to level j against the reference state x_j:

  euler:     x_hat = x_i + (sigma_j - sigma_i) * d_i,               d_i = (x_i - D_i) / sigma_i
  heun_ref:  x_hat = x_i + (sigma_j - sigma_i) * (d_i + d_j) / 2,   d_j = (x_j - D_j) / sigma_j

heun_ref is the trapezoid rule with the slope at the END taken on the REFERENCE state x_j.
The real Heun step evaluates D at the Euler-predicted point x_i + (sigma_j - sigma_i) d_i,
so its end slope carries the Euler error through the denoiser; heun_ref drops that term
(an oracle corrector). It therefore UNDER-estimates the true one-step Heun error, and more
so for long steps where the Euler predictor is poor (tests/toy_linear.py measures the
ratio on a linear-Gaussian toy).

Errors are split per variable and per band after resampling the box to the v3 regular
grid (0.07 deg, linear barycentric), a 2-D Hann window, and SHARP spectral cuts at 100 km
(the cost) and 300 km (reported): fine = wavelength < cut, coarse = the rest. Each
(variable, band) error energy is divided by the variance of the draw's FINAL reference
state in that (variable, band), so every term is dimensionless.

FIXED costs (weights set before any read, see README):
  C_L2   = mean over 10u, 10v, 2t, msl of  E_total / V_total
  C_band = mean over 10u, 10v, 2t, msl of  0.5 * E_fine100 / V_fine100 + 0.5 * E_coarse100 / V_coarse100

Outputs (in --out-dir):
  per_draw/cost_<bundle>_<seed>.npz   raw energies E[method][var][band] (N, N), V[var][band],
                                       C_L2_<method>, C_band_<method>, C_band300_<method>,
                                       trunc (start-truncation term per level), dense self-check
  cost_mean_<set>.npz                  the dimensionless matrices averaged over the draws of
                                       a set: all, and one set per date (the hold-out splits)
  cost_manifest.json                   draws, sets, grid, Nyquist, box size

  python -m scripts.t1d_sampler_20261007.dp.cost_matrix \
      --inputs '/path/diag/*/trajectory/trajectory_states_s*.npz' --out-dir /path/diag/cost
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (COST_VARS, W_COARSE, W_FINE, Trajectory, draw_date, save_json,  # type: ignore
                        splitter_for)
else:
    from .common import COST_VARS, W_COARSE, W_FINE, Trajectory, draw_date, save_json, splitter_for

LOG = logging.getLogger("t1d.cost_matrix")
METHODS = ("euler", "heun_ref")


def pair_energies(Dsp, dsp, s, splitter, method):
    """Dsp, dsp: (N, M) spectra of the reference denoiser outputs D_i and slopes
    d_i = (x_i - D_i) / sigma_i; s: (N,) sigmas. With x = D + sigma d the one-step errors are

      euler:    x_hat - x_j = (D_i - D_j) + sigma_j               * (d_i - d_j)
      heun_ref: x_hat - x_j = (D_i - D_j) + (sigma_i + sigma_j)/2 * (d_i - d_j)

    (algebraically identical to the forms in the module docstring, without the cancellation of
    terms of size sigma that the state form has at sigma 1e5). Returns {band: (N, N)} raw error
    energies, NaN on and below the diagonal."""
    N = len(s)
    out = {b: np.full((N, N), np.nan) for b in splitter.masks}
    for i in range(N - 1):
        j = np.arange(i + 1, N)
        if method == "euler":
            c = s[j]
        elif method == "heun_ref":
            c = 0.5 * (s[i] + s[j])
        else:
            raise ValueError(method)
        err = (Dsp[i][None, :] - Dsp[j]) + c[:, None] * (dsp[i][None, :] - dsp[j])
        en = splitter.energies(err)
        for b, e in en.items():
            out[b][i, j] = e
    return out


def draw_costs(tr: Trajectory, splitter, methods=METHODS, cost_vars=COST_VARS):
    s = tr.sigma
    N = len(s)
    res = {"sigma": s, "vars": list(tr.vars), "E": {}, "V": {}, "trunc": {}, "dense_step": {}}
    for v, name in enumerate(tr.vars):
        Dsp = splitter.spectrum(tr.D[:, v])               # (N, M) linear map of the D_i
        dsp = splitter.spectrum((tr.x[:, v] - tr.D[:, v]) / s[:, None])
        res["V"][name] = splitter.variance(tr.final[v])
        for m in methods:
            res["E"][(m, name)] = pair_energies(Dsp, dsp, s, splitter, m)
        # start truncation (paired seeds): starting at level j from fresh noise sigma_j * eps,
        # eps = x_0 / sigma_0 (the same draw), instead of the reference state x_j:
        # x_j - sigma_j x_0 / sigma_0 = (D_j - sigma_j/sigma_0 D_0) + sigma_j (d_j - d_0)
        res["trunc"][name] = splitter.energies((Dsp - (s[:, None] / s[0]) * Dsp[0][None, :])
                                               + s[:, None] * (dsp - dsp[0][None, :]))
        if len(tr.x2) == N - 1:                           # Euler-predicted points at level i+1
            res["dense_step"][name] = splitter.energies(splitter.spectrum(tr.x2[:, v] - tr.x[1:, v]))
    res["C"] = combine(res, cost_vars)
    return res


def combine(res, cost_vars=COST_VARS, methods=METHODS):
    """Dimensionless matrices from raw energies (per draw)."""
    C = {}
    V = res["V"]
    for m in methods:
        l2, band, band300 = [], [], []
        for name in cost_vars:
            E = res["E"][(m, name)]
            Vf, Vc = V[name]["fine100"], V[name]["coarse100"]
            l2.append((E["fine100"] + E["coarse100"]) / (Vf + Vc))
            band.append(W_FINE * E["fine100"] / Vf + W_COARSE * E["coarse100"] / Vc)
            band300.append(W_FINE * E["fine300"] / V[name]["fine300"]
                           + W_COARSE * E["coarse300"] / V[name]["coarse300"])
        C[f"C_L2_{m}"] = np.mean(l2, axis=0)
        C[f"C_band_{m}"] = np.mean(band, axis=0)
        C[f"C_band300_{m}"] = np.mean(band300, axis=0)
        for name in res["vars"]:
            E = res["E"][(m, name)]
            for b in E:
                C[f"n_{m}_{name}_{b}"] = E[b] / V[name][b]
    for name in res["vars"]:
        for b, t in res["trunc"][name].items():
            C[f"trunc_{name}_{b}"] = t / V[name][b]
        for b, t in res.get("dense_step", {}).get(name, {}).items():
            C[f"dense_step_{name}_{b}"] = t / V[name][b]
    # the two fixed costs applied to the start truncation, so pw16 (sigma_max 1e3) can be charged
    tl2, tband = [], []
    for name in cost_vars:
        T = {b: res["trunc"][name][b] for b in res["trunc"][name]}
        Vn = V[name]
        tl2.append((T["fine100"] + T["coarse100"]) / (Vn["fine100"] + Vn["coarse100"]))
        tband.append(W_FINE * T["fine100"] / Vn["fine100"] + W_COARSE * T["coarse100"] / Vn["coarse100"])
    C["trunc_C_L2"] = np.mean(tl2, axis=0)
    C["trunc_C_band"] = np.mean(tband, axis=0)
    return C


def save_draw(path, res, info):
    arrs = {"sigma": res["sigma"], "vars": np.asarray(res["vars"])}
    for (m, name), E in res["E"].items():
        for b, e in E.items():
            arrs[f"E_{m}_{name}_{b}"] = e.astype(np.float64)
    for name, Vb in res["V"].items():
        for b, v in Vb.items():
            arrs[f"V_{name}_{b}"] = np.float64(v)
    for k, v in res["C"].items():
        arrs[k] = np.asarray(v, dtype=np.float64)
    for k, v in info.items():
        arrs[f"info_{k}"] = np.asarray(v)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrs)


def _one_draw(job):
    """Worker: one draw -> per-draw npz; returns what the set means need."""
    p, out, method = Path(job[0]), Path(job[1]), job[2]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    tr = Trajectory(p)
    spl = splitter_for(tr, method=method)
    res = draw_costs(tr, spl)
    date = draw_date(tr, p)
    label = f"{p.parent.parent.name if p.parent.name == 'trajectory' else p.parent.name}_s{tr.seed}"
    info = {"source": str(p), "date": date, "seed": tr.seed, "grid": list(spl.sampler.shape),
            "dx_km": spl.dx, "dy_km": spl.dy, "nyquist_km": spl.nyquist_km,
            "box_km": list(spl.box_km), "n_cells": int(tr.lat.size), "stride": tr.stride}
    save_draw(out / "per_draw" / f"cost_{label}.npz", res, info)
    LOG.info("%s: N=%d levels, grid %s (dx %.1f km, Nyquist %.1f km), %.0f s", label,
             len(tr.sigma), spl.sampler.shape, spl.dx, spl.nyquist_km, time.time() - t0)
    return label, date, info, tr.sigma, res["C"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="trajectory_states_s*.npz files or glob patterns")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--method", default="linear", choices=["linear", "nearest"],
                    help="box -> regular grid resampling (v3 default: linear)")
    ap.add_argument("--max-draws", type=int, default=0, help="debug: stop after n draws")
    ap.add_argument("--workers", type=int, default=1, help="draws processed in parallel (processes)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    paths = []
    for pat in args.inputs:
        hits = sorted(glob.glob(pat))
        paths.extend(hits if hits else ([pat] if Path(pat).exists() else []))
    paths = [Path(p) for p in dict.fromkeys(paths)]
    if not paths:
        raise SystemExit("no trajectory_states files matched")
    if args.max_draws:
        paths = paths[: args.max_draws]
    out = Path(args.out_dir)
    sums, counts, manifest = {}, {}, {"draws": [], "sets": {}}
    sigma_ref = None
    jobs = [(str(p), str(out), args.method) for p in paths]
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(_one_draw, jobs)
    else:
        results = [_one_draw(j) for j in jobs]
    for label, date, info, sig, C in results:
        if sigma_ref is None:
            sigma_ref = sig
        elif len(sig) != len(sigma_ref) or np.max(np.abs(np.log(sig / sigma_ref))) > 1e-5:
            raise SystemExit(f"{label}: dense levels differ from the first draw's")
        for set_name in ("all", date):
            acc = sums.setdefault(set_name, {})
            for k, v in C.items():
                acc[k] = acc.get(k, 0.0) + v
            counts[set_name] = counts.get(set_name, 0) + 1
            manifest["sets"].setdefault(set_name, []).append(label)
        manifest["draws"].append(dict(label=label, **info))
    for set_name, acc in sums.items():
        n = counts[set_name]
        arrs = {k: v / n for k, v in acc.items()}
        arrs["sigma"] = sigma_ref
        arrs["n_draws"] = np.int32(n)
        np.savez_compressed(out / f"cost_mean_{set_name}.npz", **arrs)
        LOG.info("set %s: %d draws -> %s", set_name, n, out / f"cost_mean_{set_name}.npz")
    manifest["sigma"] = sigma_ref
    manifest["fixed_costs"] = {"vars": list(COST_VARS), "w_fine": W_FINE, "w_coarse": W_COARSE,
                               "cut_km": 100.0, "reported_cut_km": 300.0}
    save_json(out / "cost_manifest.json", manifest)


if __name__ == "__main__":
    main()
