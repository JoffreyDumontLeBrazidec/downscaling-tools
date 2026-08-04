"""CRPS / spread / ens-mean probe for autoguidance: FIXED input, N diffusion seeds.

Per-draw RMSE necessarily degrades when a sampler is made sharper or wider (the
ensemble mean is the RMSE-optimal point forecast), so RMSE alone cannot judge a
distribution-level intervention. This probe fixes ONE input bundle, draws N seeds
per arm (ctrl / autoguided), and reports the probabilistic metrics against that
bundle's own target y:
  fair CRPS (Gneiting-Raftery unbiased), ens-mean RMSE, ens spread,
  spread/skill ratio, and per-draw RMSE — all cos-lat weighted, full field.

Usage (ds lineage env, cwd downscaling-tools):
  python -m eval.jobs.ag_crps_probe --checkpoint <59e4> --dbad <weak> --weight 1.5 \
      --date 20230826 --member 01 --step 072 --n-seeds 20 --out crps_w15.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from interp.core.data import collect_event_bundles
from interp.core.model import get_surface_target_indices, load_model, prepare_batch
from interp.tools.trajectory import (
    _autoguided_denoiser,
    _build_sampler,
    _seeded_sample,
    reconstruct_phys,
)

BUNDLE_DIR = "/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia"


def fair_crps(ens, obs, w):
    """ens (n, npts), obs (npts), w (npts) -> cos-lat weighted fair CRPS."""
    n = ens.shape[0]
    t1 = np.abs(ens - obs[None, :]).mean(axis=0)
    t2 = np.zeros_like(t1)
    for i in range(n):
        t2 += np.abs(ens[i][None, :] - ens).sum(axis=0)
    t2 /= (2.0 * n * (n - 1))
    return float((w * (t1 - t2)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dbad", required=True)
    ap.add_argument("--weight", type=float, default=1.5)
    ap.add_argument("--date", default="20230826")
    ap.add_argument("--member", default="01")
    ap.add_argument("--step", default="072")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = load_model(args.checkpoint, device=dev, precision="fp32")
    inner = b.inner_model
    weak = load_model(args.dbad, device=dev, precision="fp32")
    ag_fn = _autoguided_denoiser(inner.fwd_with_preconditioning,
                                 weak.inner_model.fwd_with_preconditioning,
                                 float(args.weight), 0.0, 1.0e9)
    sampler, s_min, s_max = _build_sampler(inner, torch.device(dev))
    tidx = get_surface_target_indices(b)

    eb = collect_event_bundles(b, BUNDLE_DIR, [args.date], [args.member], [args.step])
    prepared = prepare_batch(b, eb.x_lres[0:1], eb.x_hres[0:1], eb.y[0:1])
    x_i, x_h, y_res, xir = (prepared["x_interp"], prepared["x_hres"],
                            prepared["y_residual"], prepared["x_interp_raw"])
    _, _, lat, _ = eb.coords
    lat = np.asarray(lat)
    w = np.cos(np.deg2rad(lat)); w = w / w.sum()
    y_true = eb.y[0]
    yt = (y_true.reshape(-1, y_true.shape[-1]) if y_true.ndim > 2 else y_true).cpu().numpy()

    out = {"config": vars(args), "arms": {}}
    for arm, fn in (("ctrl", None), ("ag", ag_fn)):
        ens = {k: [] for k in tidx}
        for s in range(args.n_seeds):
            seed = 2000 + s
            torch.manual_seed(seed)
            yf = _seeded_sample(inner, sampler, args.num_steps, s_min, s_max,
                                x_i, x_h, y_res, s_max, seed, None, None,
                                free=True, denoise_fn=fn)
            with torch.no_grad():
                p = reconstruct_phys(b, xir, yf)[0, 0, 0].float().cpu().numpy()
            for name, ch in tidx.items():
                ens[name].append(p[:, ch])
            del yf
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  %s seed %d/%d" % (arm, s + 1, args.n_seeds), flush=True)
        res = {}
        for name, ch in tidx.items():
            E = np.asarray(ens[name], dtype=np.float64)
            obs = yt[:, ch].astype(np.float64)
            em = E.mean(axis=0)
            res[name] = {
                "crps": fair_crps(E, obs, w),
                "ens_mean_rmse": float(np.sqrt((w * (em - obs) ** 2).sum())),
                "spread": float(np.sqrt((w * E.var(axis=0, ddof=1)).sum())),
                "member_rmse": float(np.mean([np.sqrt((w * (E[i] - obs) ** 2).sum())
                                              for i in range(E.shape[0])])),
            }
            res[name]["spread_skill"] = res[name]["spread"] / res[name]["ens_mean_rmse"]
        out["arms"][arm] = res
        print(arm, json.dumps({k: {m: round(v, 4) for m, v in d.items()}
                               for k, d in res.items()}, indent=1), flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
