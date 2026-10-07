"""Linear-Gaussian toy check of the cost matrix and the DP (T1d, CPU).

Four "variables" on a 96 x 96 periodic grid (lat/lon spacing 0.07 deg, as the v3 grid);
each Fourier mode is an independent Gaussian with std s(k) between 0.03 and 10 (power
laws in k: msl red, winds flatter), so the denoiser is linear and exact:
D(x, sigma) = IFFT[ FFT(x) * s^2 / (s^2 + sigma^2) ], and the probability-flow ODE has the
closed form X(sigma) = X(sigma_T) * sqrt((s^2 + sigma^2) / (s^2 + sigma_T^2)) per mode.

Steps (all with the fork's EDMHeunSampler, churn off, and the REAL capture/writer code of
interp/tools/trajectory.py):
 1. dense reference: 240 log-uniform levels 1e5 -> 0.03 (479 calls), captured into a
    trajectory_states npz, checked for call bookkeeping;
 2. cost matrices from that npz with cost_matrix.draw_costs;
 3. the approximate Heun cost (heun_ref, end slope on the reference) against the TRUE
    one-step Heun error (end slope at the Euler-predicted point, computable here because
    D is known): ratio per step length;
 4. DP schedules for K = 8 ... 30 on C_L2 and C_band; each is RUN with the fork sampler
    from the same initial noise and its final error against the exact converged solution
    (exact ODE to 0.03, then the same last step) is compared with log-uniform and the
    piecewise reference schedules at equal K.

  python scripts/t1d_sampler_20261007/dp/tests/toy_linear.py --out-dir /tmp/t1d_toy
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _harness import load_fork_samplers, load_trajectory_funcs  # noqa: E402
from common import (COST_VARS, W_COARSE, W_FINE, Trajectory, custom_scheduler_json,  # noqa: E402
                    dense_levels, piecewise_levels, splitter_for)
from cost_matrix import draw_costs  # noqa: E402
from dp_schedule import dp_path, path_cost  # noqa: E402

import torch  # noqa: E402

VAR_SPECS = {          # std at the largest scale, std at the grid-corner wavenumber
    "10u": (5.0, 0.05),
    "10v": (4.0, 0.04),
    "2t": (2.0, 0.03),
    "msl": (10.0, 0.03),
}


class ToyModel:
    def __init__(self, n=96, specs=VAR_SPECS):
        self.n = n
        self.names = list(specs)
        k = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n)))
        kmin, kmax = 1.0 / n, float(k.max())
        kk = np.maximum(k, kmin)
        s = []
        for name in self.names:
            hi, lo = specs[name]
            alpha = math.log(hi / lo) / math.log(kmax / kmin)
            s.append(hi * (kk / kmin) ** (-alpha))
        self.s = torch.tensor(np.stack(s, axis=-1), dtype=torch.float64)        # (n, n, V)

    def _spec(self, x):                       # (..., G, V) -> (..., n, n, V) spectrum
        g = x.reshape(x.shape[:-2] + (self.n, self.n, x.shape[-1]))
        return torch.fft.fft2(g, dim=(-3, -2), norm="ortho")

    def _real(self, F, shape):
        return torch.fft.ifft2(F, dim=(-3, -2), norm="ortho").real.reshape(shape)

    def denoise(self, x, sigma):
        s2 = self.s ** 2
        return self._real(self._spec(x) * s2 / (s2 + float(sigma) ** 2), x.shape)

    def exact(self, x_T, sigma_T, sigma):
        s2 = self.s ** 2
        f = torch.sqrt((s2 + float(sigma) ** 2) / (s2 + float(sigma_T) ** 2))
        return self._real(self._spec(x_T) * f, x_T.shape)


class ToyInner:
    """Stands in for the model: the sampler calls fwd_with_preconditioning(x, y, sigma, ...)."""

    def __init__(self, model):
        self.model = model

    def fwd_with_preconditioning(self, x, y, sigma, model_comm_group=None, grid_shard_sizes=None):
        sig = float(next(iter(sigma.values())).reshape(-1)[0])
        return {"out_hres": self.model.denoise(y["out_hres"], sig)}


def run_sampler(ds, inner, levels, x_T_unit, capture=None):
    """Fork Heun, churn off, on the positive `levels` (+ terminal 0); start noise
    x_T_unit * levels[0] (paired seeds)."""
    sig = torch.tensor(list(levels) + [0.0], dtype=torch.float64)
    y = {"out_hres": (x_T_unit * float(levels[0])).clone()}
    smp = ds.EDMHeunSampler(S_churn=0.0, S_noise=1.0, dtype=torch.float64)
    fn = inner.fwd_with_preconditioning
    return smp.sample({"in_lres": None}, y, sig, fn, dtype=torch.float64)["out_hres"]


def norm_err(spl, err_fields, ref_final):
    """(V, n) error fields -> (C_L2, C_band, per-var dict) with the fixed weights."""
    l2, band, per = [], [], {}
    for v, name in enumerate(COST_VARS):
        V = spl.variance(ref_final[v])
        E = spl.energies(spl.spectrum(err_fields[v]))
        l2.append((E["fine100"] + E["coarse100"]) / (V["fine100"] + V["coarse100"]))
        band.append(W_FINE * E["fine100"] / V["fine100"] + W_COARSE * E["coarse100"] / V["coarse100"])
        per[name] = {"fine100": E["fine100"] / V["fine100"], "coarse100": E["coarse100"] / V["coarse100"]}
    return float(np.mean(l2)), float(np.mean(band)), per


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp/t1d_toy")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--levels", type=int, default=240)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ds, ds_file = load_fork_samplers()
    fx = load_trajectory_funcs()
    capture_denoiser, write_states = fx["capture_denoiser"], fx["_write_trajectory_states"]

    model = ToyModel(args.n)
    inner = ToyInner(model)
    G, V = args.n * args.n, len(model.names)
    g = torch.Generator().manual_seed(args.seed)
    x_unit = torch.randn((1, 1, 1, G, V), generator=g, dtype=torch.float64)
    dense = dense_levels(args.levels)

    # 1. dense reference through the real capture
    calls = []

    def on_call(sigma, D, x_in=None):
        calls.append((sigma, x_in[0, 0, 0].T.numpy().astype(np.float32).copy(),
                      D[0, 0, 0].T.numpy().astype(np.float32).copy()))

    with capture_denoiser(inner, on_call, pass_input=True):
        final = run_sampler(ds, inner, dense, x_unit)
    n_calls = len(calls)
    sig_calls = np.array([c[0] for c in calls])
    book = {"n_calls": n_calls, "expected": 2 * len(dense) - 1,
            "first_eval_sigmas_match": bool(np.allclose(sig_calls[0::2], dense, rtol=1e-12)),
            "second_eval_at_next_level": bool(np.allclose(sig_calls[1::2], dense[1:], rtol=1e-12)),
            "call0_input_is_initial_noise": bool(np.allclose(calls[0][1], (x_unit[0, 0, 0] * dense[0]).T.numpy(),
                                                             rtol=1e-6))}
    assert book["n_calls"] == book["expected"] and book["first_eval_sigmas_match"] \
        and book["second_eval_at_next_level"] and book["call0_input_is_initial_noise"], book
    iy, ix = np.divmod(np.arange(G), args.n)
    lat, lon = 25.0 + 0.07 * iy, 290.0 + 0.07 * ix
    npz = out / "toy" / f"trajectory_states_s{args.seed}.npz"
    write_states(npz, calls, final[0, 0, 0].T.numpy(), np.zeros((V, G), np.float32), model.names,
                 list(range(V)), lat, lon, np.arange(G), 1, G,
                 meta={"seed": args.seed, "checkpoint": "toy", "bundle": "toy_date20230826",
                       "center_lat": 0.0, "center_lon": 0.0, "radius_km": 0.0, "local_scope": "",
                       "noise_scheduler": json.dumps(custom_scheduler_json(dense)), "sampler_params": "",
                       "num_steps": len(dense)})
    tr = Trajectory(npz)
    assert tr.x.shape == (len(dense), V, G) and tr.x2.shape == (len(dense) - 1, V, G)

    # exact converged solution: exact ODE to 0.03, then the same last step (x = D(x, 0.03))
    x_end = model.exact(x_unit * dense[0], dense[0], dense[-1])
    exact_final = model.denoise(x_end, dense[-1])[0, 0, 0].T.numpy()
    spl = splitter_for(tr)
    dense_err = norm_err(spl, final[0, 0, 0].T.numpy() - exact_final, exact_final)

    # 2. cost matrices (the same code path as the real data)
    res = draw_costs(tr, spl)
    C = res["C"]

    # 3. approximate vs true one-step Heun
    s = tr.sigma
    rows = []
    for i in range(0, len(s) - 1, 3):
        for m in (1, 2, 4, 8, 16, 32, 64, 128):
            j = i + m
            if j >= len(s):
                break
            xi = torch.from_numpy(tr.x[i].T)[None, None, None]                  # (1,1,1,G,V)
            di = (xi - model.denoise(xi, s[i])) / s[i]
            h = s[j] - s[i]
            xe = xi + h * di
            de = (xe - model.denoise(xe, s[j])) / s[j]
            xh = xi + h * 0.5 * (di + de)
            err = (xh[0, 0, 0].T.numpy() - tr.x[j])
            l2_true, band_true, _ = norm_err(spl, err, tr.final)
            rows.append({"i": i, "j": j, "ln_step": float(math.log(s[i] / s[j])), "sigma_i": float(s[i]),
                         "sigma_j": float(s[j]), "L2_true": l2_true, "band_true": band_true,
                         "L2_approx": float(C["C_L2_heun_ref"][i, j]), "band_approx": float(C["C_band_heun_ref"][i, j]),
                         "L2_euler": float(C["C_L2_euler"][i, j])})
    ratio = {}
    for lo, hi in ((0, 0.3), (0.3, 0.7), (0.7, 1.5), (1.5, 9.9)):
        sel = [r for r in rows if lo <= r["ln_step"] < hi and r["L2_true"] > 1e-12]
        if sel:
            rl = np.array([r["L2_approx"] / r["L2_true"] for r in sel])
            rb = np.array([r["band_approx"] / r["band_true"] for r in sel if r["band_true"] > 1e-12])
            ratio[f"ln_step {lo}-{hi}"] = {"n": len(sel), "L2_median": float(np.median(rl)),
                                           "L2_p10": float(np.quantile(rl, 0.1)), "L2_p90": float(np.quantile(rl, 0.9)),
                                           "band_median": float(np.median(rb)) if rb.size else None}

    # 4. DP schedules, run for real, vs baselines at equal K
    def run_err(levels):
        f = run_sampler(ds, inner, levels, x_unit)[0, 0, 0].T.numpy()
        return norm_err(spl, f - exact_final, exact_final)[:2]

    table = []
    for K in (8, 10, 12, 16, 20, 30):
        entry = {"K": K, "calls": 2 * K - 1}
        for cname in ("C_L2_heun_ref", "C_band_heun_ref"):
            path, val = dp_path(C[cname], K)
            lv = s[path]
            e = run_err(lv)
            entry[cname] = {"sigmas": [float("%.4g" % x) for x in lv], "pred_cost": val,
                            "run_L2": e[0], "run_band": e[1],
                            "max_ln_step": float(np.max(np.log(lv[:-1] / lv[1:])))}
        lu = np.exp(np.linspace(math.log(1e5), math.log(0.03), K))
        e = run_err(lu)
        idx = np.unique([int(np.argmin(np.abs(np.log(s) - math.log(x)))) for x in lu])
        entry["logu"] = {"run_L2": e[0], "run_band": e[1], "pred_L2": path_cost(C["C_L2_heun_ref"], list(idx)),
                         "pred_band": path_cost(C["C_band_heun_ref"], list(idx))}
        if K >= 4:
            nh = max(1, round(K / 3))
            pw = piecewise_levels(1e5, 0.03, nh, K - nh)
            e = run_err(pw)
            entry[f"pw_{nh}+{K - nh}_1e5"] = {"run_L2": e[0], "run_band": e[1]}
        table.append(entry)
    for name, lv in (("c0_30", piecewise_levels(1e5, 0.03, 10, 20)),
                     ("c0_pw16_s1k", piecewise_levels(1e3, 0.03, 5, 11))):
        e = run_err(lv)
        table.append({"reference": name, "K": len(lv), "calls": 2 * len(lv) - 1, "run_L2": e[0], "run_band": e[1]})

    # DP-path ratio approx / true along the K=16 band path
    p16, _ = dp_path(C["C_band_heun_ref"], 16)
    seg = []
    for a, b in zip(p16[:-1], p16[1:]):
        xi = torch.from_numpy(tr.x[a].T)[None, None, None]
        di = (xi - model.denoise(xi, s[a])) / s[a]
        h = s[b] - s[a]
        xe = xi + h * di
        de = (xe - model.denoise(xe, s[b])) / s[b]
        xh = xi + h * 0.5 * (di + de)
        l2t, bt, _ = norm_err(spl, xh[0, 0, 0].T.numpy() - tr.x[b], tr.final)
        seg.append({"sigma_from": float(s[a]), "sigma_to": float(s[b]), "band_true": bt,
                    "band_approx": float(C["C_band_heun_ref"][a, b]), "L2_true": l2t,
                    "L2_approx": float(C["C_L2_heun_ref"][a, b])})
    sum_ratio = {"band": sum(x["band_approx"] for x in seg) / max(sum(x["band_true"] for x in seg), 1e-300),
                 "L2": sum(x["L2_approx"] for x in seg) / max(sum(x["L2_true"] for x in seg), 1e-300)}

    report = {"fork_samplers": ds_file, "grid": [args.n, args.n], "levels": len(dense), "bookkeeping": book,
              "npz_bytes": npz.stat().st_size, "dense_reference_error_vs_exact": {"L2": dense_err[0], "band": dense_err[1]},
              "approx_over_true_heun_by_step": ratio, "approx_over_true_heun_along_dp16_band": sum_ratio,
              "dp16_band_segments": seg, "schedules": table, "pairs": rows, "wall_s": time.time() - t0}
    json.dump(report, open(out / "toy_report.json", "w"), indent=1)

    print(f"fork samplers: {ds_file}")
    print(f"bookkeeping: {book}")
    print(f"dense 240-level Heun vs exact: C_L2 {dense_err[0]:.3g}  C_band {dense_err[1]:.3g}")
    print("approx-Heun / true-Heun one-step cost, by ln step:")
    for k, v in ratio.items():
        print(f"  {k:>14}: n={v['n']:3d}  L2 median {v['L2_median']:.3f} (p10 {v['L2_p10']:.3f}, p90 {v['L2_p90']:.3f})"
              f"  band median {v['band_median']:.3f}")
    print(f"  along the DP K=16 band path: sum approx / sum true = band {sum_ratio['band']:.3f}, L2 {sum_ratio['L2']:.3f}")
    print("schedules run from the same noise, final error vs exact (C_L2 / C_band):")
    for e in table:
        if "reference" in e:
            print(f"  {e['reference']:>12} K={e['K']:2d} ({e['calls']} calls): {e['run_L2']:.3g} / {e['run_band']:.3g}")
            continue
        pw = [k for k in e if k.startswith("pw_")]
        print(f"  K={e['K']:2d} ({e['calls']:2d} calls): DP-L2 {e['C_L2_heun_ref']['run_L2']:.3g} / "
              f"{e['C_L2_heun_ref']['run_band']:.3g}   DP-band {e['C_band_heun_ref']['run_L2']:.3g} / "
              f"{e['C_band_heun_ref']['run_band']:.3g}   logu {e['logu']['run_L2']:.3g} / {e['logu']['run_band']:.3g}"
              + (f"   {pw[0]} {e[pw[0]]['run_L2']:.3g} / {e[pw[0]]['run_band']:.3g}" if pw else ""))
        print(f"        DP-band sigmas: {e['C_band_heun_ref']['sigmas']}")
    print(f"wall {time.time() - t0:.0f} s; report {out / 'toy_report.json'}")
    return report


if __name__ == "__main__":
    main()
