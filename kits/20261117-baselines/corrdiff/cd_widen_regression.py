#!/usr/bin/env python3
"""CorrDiff baseline: widen a trained regression U-Net (.mdlus) by N input channels, so that its training continues
with extra inputs (the 5 time inputs of cd_dataset.py, appended after the 46) and its output at the start is the donor's.

Channel layout (PhysicsNeMo v2.2.2): CorrDiffRegressionUNet feeds SongUNetPosEmbd cat(zero latent [img_out_channels],
img_lr [img_in_channels - N_grid_channels]), and SongUNetPosEmbd appends its positional grid [N_grid_channels] last.
The dataset channels added at the end of img_lr therefore land at index img_out_channels + img_in_channels -
N_grid_channels of the network input (56 for 10 outputs, 46 dataset channels, 4 grid channels), before the grid
channels. Every weight whose input axis (dim 1) grows by N gets N zero slices there; every other tensor is copied; any
other shape difference stops the tool. The donor's saved constructor arguments are reused with img_in_channels + N.

Starting point of a new run: write the widened file as RUN/checkpoints_regression/CorrDiffRegressionUNet.0.0.mdlus in an
otherwise empty folder (--run-dir RUN). PhysicsNeMo's load_checkpoint then loads it as the newest model file (index 0),
finds no checkpoint.0.*.pt and returns 0: training starts at 0 samples with a fresh Adam state (selftest shows it).
train.py wraps the model load in try/except: pass, so a channel mismatch silently trains from random weights; the
training log must show "Loaded model state dictionary ...CorrDiffRegressionUNet.0.0.mdlus".

    python3 cd_widen_regression.py widen --donor D.mdlus --add 5 (--out W.mdlus | --run-dir RUN)
    python3 cd_widen_regression.py weights --donor D.mdlus --widened W.mdlus      # weight-level test, CPU, light
    python3 cd_widen_regression.py check --donor D.mdlus --widened W.mdlus --cfg CONF.yaml   # GPU: same output
    python3 cd_widen_regression.py selftest --work DIR     # tiny CPU model: widen, forward, load_checkpoint calls
"""
import argparse
import copy
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_mdlus(path, what):
    """'args.json' -> dict, 'model.pt' -> state_dict (CPU), from a zip or tar .mdlus."""
    import torch
    path = str(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            raw = z.read(what)
    else:
        with tarfile.open(path) as t:
            raw = t.extractfile(what).read()
    if what == "args.json":
        return json.loads(raw)
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)


def insert_index(args):
    a = args["__args__"]
    assert args["__name__"] == "CorrDiffRegressionUNet", args["__name__"]
    assert a.get("model_type", "SongUNetPosEmbd") == "SongUNetPosEmbd", a.get("model_type")
    return int(a["img_out_channels"]) + int(a["img_in_channels"]) - int(a.get("N_grid_channels", 4))


def widen_state(sd_old, sd_new, at, add):
    """sd_new filled from sd_old: equal shapes copied, dim-1 growth by `add` -> zero slices at `at`. Returns grown keys."""
    import torch
    assert set(sd_old) == set(sd_new), (sorted(set(sd_old) ^ set(sd_new)))[:10]
    grown = []
    for k, t_new in sd_new.items():
        t_old = sd_old[k]
        if t_old.shape == t_new.shape:
            sd_new[k] = t_old.clone()
            continue
        ok = (t_old.ndim == t_new.ndim >= 2 and t_new.shape[1] == t_old.shape[1] + add
              and t_new.shape[0] == t_old.shape[0] and t_new.shape[2:] == t_old.shape[2:] and at <= t_old.shape[1])
        if not ok:
            raise SystemExit(f"unexpected shape change {k}: {tuple(t_old.shape)} -> {tuple(t_new.shape)}")
        w = torch.zeros_like(t_new, dtype=t_old.dtype)
        w[:, :at] = t_old[:, :at]
        w[:, at + add:] = t_old[:, at:]
        sd_new[k] = w
        grown.append(k)
    return grown


def widen(donor, add, out):
    from physicsnemo import Module
    args = read_mdlus(donor, "args.json")
    at = insert_index(args)
    new_args = copy.deepcopy(args)
    new_args["__args__"]["img_in_channels"] = int(args["__args__"]["img_in_channels"]) + add
    sd_old = read_mdlus(donor, "model.pt")
    net = Module.instantiate(new_args)
    sd_new = net.state_dict()
    grown = widen_state(sd_old, sd_new, at, add)
    net.load_state_dict(sd_new, strict=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.save(str(out))
    print(f"WIDEN_DONE {donor} -> {out}: img_in_channels {args['__args__']['img_in_channels']} -> "
          f"{new_args['__args__']['img_in_channels']}, {add} zero input slices at network-input index {at}; grown tensors: "
          f"{[(k, tuple(sd_new[k].shape)) for k in grown]}", flush=True)
    return grown


def weights_test(donor, widened):
    """Shared tensors equal, the inserted slices zero, args equal except img_in_channels."""
    import torch
    a0, a1 = read_mdlus(donor, "args.json"), read_mdlus(widened, "args.json")
    add = int(a1["__args__"]["img_in_channels"]) - int(a0["__args__"]["img_in_channels"])
    assert add > 0, add
    d0 = {k: v for k, v in a0["__args__"].items() if k != "img_in_channels"}
    d1 = {k: v for k, v in a1["__args__"].items() if k != "img_in_channels"}
    assert d0 == d1 and a0["__name__"] == a1["__name__"], (d0, d1)
    at = insert_index(a0)
    s0, s1 = read_mdlus(donor, "model.pt"), read_mdlus(widened, "model.pt")
    assert set(s0) == set(s1)
    grown, n = [], 0
    for k in s0:
        if s0[k].shape == s1[k].shape:
            assert torch.equal(s0[k], s1[k]), k
            n += 1
        else:
            assert torch.equal(s1[k][:, :at], s0[k][:, :at]) and torch.equal(s1[k][:, at + add:], s0[k][:, at:]), k
            assert not s1[k][:, at:at + add].any(), f"{k}: new slices not zero"
            grown.append(k)
    print(f"WEIGHTS_OK {n} tensors equal, {len(grown)} grown with {add} zero input slices at {at}: {grown}", flush=True)


def check(donor, widened, cfg, seed=0):
    """GPU: donor(x46) vs widened(x46 + 5 random channels) on one validation row, fp32, TF32 off."""
    import numpy as np
    import torch
    import yaml
    from physicsnemo import Module
    from physicsnemo.diffusion.generate import regression_step
    sys.path.insert(0, str(HERE))
    from cd_dataset import E2v4Box
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dcfg = dict(yaml.safe_load(open(cfg))["dataset"])
    dcfg.pop("type")
    dcfg.update(split="valid", max_rows=1, time_inputs=False, crop=0)
    ds = E2v4Box(**dcfg)
    x46 = ds[0][1]
    add = int(read_mdlus(widened, "args.json")["__args__"]["img_in_channels"]) - \
        int(read_mdlus(donor, "args.json")["__args__"]["img_in_channels"])
    extra = np.random.default_rng(seed).standard_normal((add,) + x46.shape[1:]).astype(np.float32)
    outs = []
    for p, x in ((donor, x46), (widened, np.concatenate([x46, extra]))):
        m = Module.from_checkpoint(str(p), override_args={"use_apex_gn": False})
        m.use_fp16 = False
        m = m.eval().to(dev).to(memory_format=torch.channels_last)
        if hasattr(m, "amp_mode"):
            m.amp_mode = False
        img_lr = torch.from_numpy(x[None]).to(dev).to(memory_format=torch.channels_last)
        outs.append(regression_step(net=m, img_lr=img_lr, latents_shape=(1, len(ds.output_variables), *ds.image_shape()))
                    .float().cpu().numpy())
        del m
        torch.cuda.empty_cache()
    d = np.abs(outs[0] - outs[1])
    scale = np.abs(outs[0]).max()
    ok = d.max() <= 1e-4 * max(scale, 1.0)
    print(f"{'CHECK_OK' if ok else 'CHECK_FAIL'} donor vs widened (+{add} random channels) on {dev}: max |diff| "
          f"{d.max():.3e}, mean |diff| {d.mean():.3e}, max |donor output| {scale:.3f} (normalised units)", flush=True)
    return ok


def selftest(work):
    """Tiny model on CPU: widen + forward equality, then the load_checkpoint calls of train.py on a fresh run folder."""
    import logging
    import torch
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.models.diffusion_unets import CorrDiffRegressionUNet
    from physicsnemo.utils import load_checkpoint
    logging.basicConfig(level=logging.INFO)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    kw = dict(img_resolution=[32, 64], img_out_channels=3, model_type="SongUNetPosEmbd", N_grid_channels=4,
              embedding_type="zero", model_channels=16, channel_mult=[1, 2], attn_resolutions=[], checkpoint_level=0)
    nin, add = 6, 2
    donor = CorrDiffRegressionUNet(img_in_channels=nin + 4, **kw)
    with torch.no_grad():
        for p in donor.parameters():
            p.add_(0.05 * torch.randn_like(p))   # zero-initialised layers would hide a wrong slice
    dpath = work / "donor.mdlus"
    donor.save(str(dpath))
    run = work / "run"
    ckdir = run / "checkpoints_regression"
    assert not ckdir.exists() or not any(ckdir.iterdir()), f"{ckdir} not empty"
    wpath = ckdir / "CorrDiffRegressionUNet.0.0.mdlus"
    widen(dpath, add, wpath)
    weights_test(dpath, wpath)
    from physicsnemo import Module
    wide = Module.from_checkpoint(str(wpath)).eval()
    x = torch.randn(1, nin, 32, 64)
    extra = torch.randn(1, add, 32, 64)
    z = torch.zeros(1, 3, 32, 64)
    with torch.no_grad():
        y0 = donor.eval()(z, x)
        y1 = wide(z, torch.cat([x, extra], 1))
        y_wrong = wide(z, torch.cat([extra, x], 1))
    d = (y0 - y1).abs().max().item()
    print(f"SELFTEST forward donor vs widened: max |diff| {d:.2e} (extra channels appended); "
          f"control with the extra channels in front: {(y0 - y_wrong).abs().max().item():.2e}", flush=True)
    assert d < 1e-5, d
    # train.py: cur_nimg = load_checkpoint(path) ; load_checkpoint(path, models=model) ; load_checkpoint(path, optimizer)
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    fresh = CorrDiffRegressionUNet(img_in_channels=nin + add + 4, **kw)
    cur = load_checkpoint(path=str(ckdir))
    print(f"SELFTEST load_checkpoint(path) -> cur_nimg {cur}", flush=True)
    r = load_checkpoint(path=str(ckdir), models=fresh)
    same = all(torch.equal(a, b) for a, b in zip(fresh.state_dict().values(), wide.state_dict().values()))
    print(f"SELFTEST load_checkpoint(path, models=model) -> {r}; weights equal to the widened file: {same}", flush=True)
    opt = torch.optim.Adam(fresh.parameters(), lr=2e-4)
    r = load_checkpoint(path=str(ckdir), optimizer=opt, device="cpu")
    print(f"SELFTEST load_checkpoint(path, optimizer) -> {r}; Adam state entries after load: {len(opt.state)}", flush=True)
    assert cur == 0 and same and len(opt.state) == 0
    narrow = CorrDiffRegressionUNet(img_in_channels=nin + 4, **kw)
    try:
        load_checkpoint(path=str(ckdir), models=narrow)
        print("SELFTEST control: a model WITHOUT the extra channels loaded the widened file (unexpected)", flush=True)
    except Exception as e:  # train.py swallows this (try/except: pass) and trains from random weights
        print(f"SELFTEST control: a model without the extra channels fails to load it: {type(e).__name__}: "
              f"{str(e).splitlines()[0][:160]}", flush=True)
    print("SELFTEST_OK", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["widen", "weights", "check", "selftest"])
    ap.add_argument("--donor")
    ap.add_argument("--widened")
    ap.add_argument("--add", type=int, default=5)
    ap.add_argument("--out", help="widened .mdlus path")
    ap.add_argument("--run-dir", help="new run folder: writes RUN/checkpoints_regression/CorrDiffRegressionUNet.0.0.mdlus")
    ap.add_argument("--cfg", help="check: a training config (dataset block) of the kit")
    ap.add_argument("--work", help="selftest folder")
    a = ap.parse_args()
    if a.cmd == "widen":
        assert bool(a.out) != bool(a.run_dir), "give --out or --run-dir"
        out = Path(a.out) if a.out else Path(a.run_dir) / "checkpoints_regression" / "CorrDiffRegressionUNet.0.0.mdlus"
        old = [p.name for p in out.parent.glob("*") if p.suffix in (".mdlus", ".pt")] if out.parent.exists() else []
        if a.run_dir and old:
            raise SystemExit(f"{out.parent} already holds {old[:4]}: a new run folder must be empty")
        if out.exists():
            raise SystemExit(f"{out} exists")
        widen(a.donor, a.add, out)
        weights_test(a.donor, out)
    elif a.cmd == "weights":
        weights_test(a.donor, a.widened)
    elif a.cmd == "check":
        sys.exit(0 if check(a.donor, a.widened, a.cfg) else 1)
    else:
        selftest(a.work)


if __name__ == "__main__":
    main()
