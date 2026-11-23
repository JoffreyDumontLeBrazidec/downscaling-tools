#!/usr/bin/env python3
"""CorrDiff baseline, inference on an evaluation set (Leonardo GPU, one process per GPU; run from the pinned PhysicsNeMo
example folder examples/weather/corrdiff, tag v2.2.2, so `datasets` and `helpers` import).

For each start date (cdin_<date>.npz of cd_cut_inputs.py), lead and input member k: the coarse input -> E2v4Box's
own preprocessing (x4 upsample, invariants, [time inputs at valid time = date + lead when the config's dataset
block has time_inputs], normalisation) -> regression U-Net (the mean) + ONE draw of the patched
residual diffusion with the published generation defaults (stochastic EDM sampler, 18 steps, 448 x 448 patches,
overlap 4, boundary 2, hr_mean_conditioning) -> physical units. The draw's seed is rule 31's
derive_draw_seed(1234, date, lead, k-1), so member k of every row of the paper shares its seed. One realisation per
input member (rule 36).
Writes OUT/cdout_<date>_step<LLL>.npz: y_pred (10 members, evaluation points, 10 outputs) bilinear from the box grid,
NaN outside it; full (10 outputs, 864, 1728) float16 of member 1 (for spectra on the box grid); states.

    torchrun --nproc_per_node 4 cd_predict.py --cfg CONF.yaml --reg REG.mdlus --res RES.mdlus --inputs DIR --out DIR \
        [--sigma-max 800] [--sigma-min 0.002]
--sigma-max / --sigma-min go to stochastic_sampler (defaults = its own, the published 800 / 0.002).
"""
import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cd_grid as G  # noqa: E402

SEED_BASE = 1234


def derive_draw_seed(base, date, lead, member):
    """= xset_predict_global.derive_draw_seed (rule 31)."""
    return int(np.random.SeedSequence([int(base), int(date), int(lead), int(member)]).generate_state(1, dtype=np.uint32)[0])


def start_time(d):
    """Start date of cdin_<date>.npz: YYYYMMDD (00 UTC, the XSET sets) or YYYYMMDDHH."""
    d = str(d)
    return datetime.datetime.strptime(d, "%Y%m%d%H" if len(d) == 10 else "%Y%m%d")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True, help="the training config of the diffusion run (dataset block)")
    ap.add_argument("--reg", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-steps", type=int, default=18)
    ap.add_argument("--sigma-max", type=float, default=800.0)
    ap.add_argument("--sigma-min", type=float, default=0.002)
    a = ap.parse_args()
    from functools import partial

    import torch
    import yaml
    from helpers.train_helpers import set_patch_shape
    from physicsnemo import Module
    from physicsnemo.diffusion.generate import diffusion_step, regression_step
    from physicsnemo.diffusion.multi_diffusion import GridPatching2D
    from physicsnemo.diffusion.samplers import stochastic_sampler

    sys.path.insert(0, str(HERE))
    from cd_dataset import E2v4Box

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    torch.cuda.set_device(dev)
    dcfg = dict(yaml.safe_load(open(a.cfg))["dataset"])
    dcfg.pop("type")
    dcfg["split"] = "valid"
    dcfg["max_rows"] = 1
    ds = E2v4Box(**dcfg)
    img_shape = ds.image_shape()
    n_out = len(ds.output_channels())
    use_p, img_shape, patch_shape = set_patch_shape(img_shape, (448, 448))
    assert use_p
    patching = GridPatching2D(img_shape=img_shape, patch_shape=patch_shape, boundary_pix=2, overlap_pix=4)
    sampler_fn = partial(stochastic_sampler, patching=patching, num_steps=a.num_steps,
                         sigma_min=a.sigma_min, sigma_max=a.sigma_max)
    nets = []
    for p in (a.reg, a.res):
        m = Module.from_checkpoint(p, override_args={"use_apex_gn": False})
        m.use_fp16 = False
        m = m.eval().to(dev).to(memory_format=torch.channels_last)
        if hasattr(m, "amp_mode"):
            m.amp_mode = False
        nets.append(m)
    net_reg, net_res = nets
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.inputs).glob("cdin_*.npz"))
    jobs = [(f, li) for f in files for li in range(6)]
    for f, li in jobs[rank::world]:
        z = np.load(f)
        d, lead = str(z["date"]), int(z["leads"][li])
        name = out / f"cdout_{d}_step{lead:03d}.npz"
        if name.exists():
            continue
        t0 = time.time()
        vars_ = [str(v) for v in z["variables"]]
        order = [vars_.index(v) for v in ds.all_in]          # the export's 70-variable order
        bw = G.bilinear_weights(z["elat"], z["elon"])
        valid = start_time(d) + datetime.timedelta(hours=lead)
        preds, full1 = [], None
        for k in range(z["xc"].shape[1]):
            x = ds.coarse_input(z["xc"][li, k][order], valid)
            img_lr = torch.from_numpy(x[None]).to(dev).to(memory_format=torch.channels_last)
            mean = regression_step(net=net_reg, img_lr=img_lr, latents_shape=(1, n_out, *img_shape))
            seed = derive_draw_seed(SEED_BASE, int(d), lead, k)
            res = diffusion_step(net=net_res, sampler_fn=sampler_fn, img_shape=img_shape, img_out_channels=n_out,
                                 rank_batches=[torch.as_tensor([seed])], img_lr=img_lr, rank=0, device=dev,
                                 mean_hr=mean[0:1])
            y = ds.denormalize_output((mean + res)[0].float().cpu().numpy())
            preds.append(np.moveaxis(G.bilinear_apply(y, bw), 0, -1))   # (points, 10)
            if k == 0:
                full1 = y.astype(np.float16)
        y_pred = np.stack(preds).astype(np.float32)
        tp = ds.output_variables.index("tp")
        fin = np.isfinite(y_pred[..., 0]).all(0)
        np.savez(str(name) + ".part.npz", y_pred=y_pred, full=full1, states=np.array(ds.output_variables), date=d, lead=lead,
                 seeds=np.array([derive_draw_seed(SEED_BASE, int(d), lead, k) for k in range(len(preds))]))
        os.replace(str(name) + ".part.npz", name)
        print(f"CD {d} +{lead} h: {fin.sum()} of {fin.size} points in the box, tp max {np.nanmax(y_pred[..., tp]) * 1000:.1f} mm, "
              f"{time.time() - t0:.0f}s", flush=True)
    print(f"PREDICT_DONE rank {rank}", flush=True)


if __name__ == "__main__":
    main()
