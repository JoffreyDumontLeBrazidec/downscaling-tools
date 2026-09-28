"""Paired ctrl-vs-autoguidance screen for ds-API checkpoints (59e4-class) on the
interp machinery — the eval predict bundle path is unified-only (IndexCollection
has no in_lres) and the ds model cannot cut_graph, so this script loops the full
(date, member, step) bundle set in ONE process, running the free sampler twice per
bundle with the SAME seed: ctrl and autoguided draws share the initial noise
(paired comparison). Scores registry event boxes (franklin/idalia) per draw and
dumps full msl/10u/10v fields for the first --dump-n bundles per arm (spectra gate).

Usage (inside the ds lineage env, cwd downscaling-tools):
  python -m eval.jobs.ag59e4_screen --checkpoint <59e4 training ckpt> \
      --dbad <cfec weak ckpt> --weight 1.5 --dates 20230826 [...] --out screen_<date>.json
"""
from __future__ import annotations

import argparse
import json
import os

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

EVENTS = {"franklin": (15.0, 38.0, -78.0, -58.0), "idalia": (10.0, 40.0, -100.0, -80.0)}
BUNDLE_DIR = "/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia"


def box_stats(vals_msl, vals_u, vals_v, masks):
    out = {}
    wind = np.hypot(vals_u, vals_v)
    for ev, m in masks.items():
        out[ev] = {
            "msl_min": float(vals_msl[m].min() / 100.0),
            "wind_max": float(wind[m].max()),
            "wind_p99": float(np.quantile(wind[m], 0.99)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dbad", required=True)
    ap.add_argument("--weight", type=float, default=1.5)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--members", nargs="+", default=[f"{m:02d}" for m in range(1, 11)])
    ap.add_argument("--steps", nargs="+", default=["024", "048", "072", "096", "120"])
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--dump-n", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_model(args.checkpoint, device=device, precision="fp32")
    inner = bundle.inner_model
    weak_bundle = load_model(args.dbad, device=device, precision="fp32")
    ag_fn = _autoguided_denoiser(inner.fwd_with_preconditioning,
                                 weak_bundle.inner_model.fwd_with_preconditioning,
                                 float(args.weight), 0.0, 1.0e9)
    sampler, sigma_min, sigma_max = _build_sampler(inner, torch.device(device))

    tidx = get_surface_target_indices(bundle)
    i_msl, i_u, i_v = tidx["msl"], tidx["10u"], tidx["10v"]

    eb = collect_event_bundles(bundle, BUNDLE_DIR, args.dates, args.members, args.steps)
    _, _, lat, lon = eb.coords
    lat = np.asarray(lat)
    lon = (np.asarray(lon) + 180.0) % 360.0 - 180.0
    masks = {ev: (lat >= b[0]) & (lat <= b[1]) & (lon >= b[2]) & (lon <= b[3])
             for ev, b in EVENTS.items()}

    rows, dumps = [], {}
    n = eb.x_lres.shape[0]
    for k in range(n):
        prepared = prepare_batch(bundle, eb.x_lres[k:k + 1], eb.x_hres[k:k + 1], eb.y[k:k + 1])
        x_i, x_h = prepared["x_interp"], prepared["x_hres"]
        y_res, xir = prepared["y_residual"], prepared["x_interp_raw"]
        y_true = eb.y[k]
        yt = y_true.reshape(-1, y_true.shape[-1]).cpu().numpy() if y_true.ndim > 2 else y_true.cpu().numpy()
        truth = box_stats(yt[:, i_msl], yt[:, i_u], yt[:, i_v], masks)
        seed = args.seed_base + k
        row = {"path": os.path.basename(eb.paths[k]), "seed": seed, "truth": truth}
        for arm, fn in (("ctrl", None), ("ag", ag_fn)):
            torch.manual_seed(seed)
            yf = _seeded_sample(inner, sampler, args.num_steps, sigma_min, sigma_max,
                                x_i, x_h, y_res, sigma_max, seed, None, None,
                                free=True, denoise_fn=fn)
            with torch.no_grad():
                phys = reconstruct_phys(bundle, xir, yf)
            p = phys[0, 0, 0].float().cpu().numpy()
            row[arm] = box_stats(p[:, i_msl], p[:, i_u], p[:, i_v], masks)
            if k < int(args.dump_n):
                for name, ch in (("msl", i_msl), ("10u", i_u), ("10v", i_v)):
                    dumps["%s_k%d_%s" % (arm, k, name)] = p[:, ch].astype(np.float32)
            del yf, phys
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if k < int(args.dump_n):
            for name, ch in (("msl", i_msl), ("10u", i_u), ("10v", i_v)):
                dumps["truth_k%d_%s" % (k, name)] = yt[:, ch].astype(np.float32)
        rows.append(row)
        print("[%d/%d] %s ctrl fr %.1f | ag fr %.1f" % (
            k + 1, n, row["path"], row["ctrl"]["franklin"]["msl_min"],
            row["ag"]["franklin"]["msl_min"]), flush=True)

    with open(args.out, "w") as f:
        json.dump({"weight": args.weight, "dbad": args.dbad, "checkpoint": args.checkpoint,
                   "num_steps": args.num_steps, "rows": rows}, f)
    if dumps:
        dumps["lat"], dumps["lon"] = lat.astype(np.float32), lon.astype(np.float32)
        np.savez_compressed(args.out.replace(".json", "_fields.npz"), **dumps)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
