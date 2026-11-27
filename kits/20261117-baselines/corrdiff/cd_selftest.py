#!/usr/bin/env python3
"""Self-test of the kit's opt-in options on a synthetic export (CPU, small; needs the PhysicsNeMo venv for the ladder
part). Checks: cd_dataset default unchanged (46 channels), time_inputs (51 channels, the 5 extra = time_fields at the
row's valid time, the first 46 unchanged, channel metadata), spread (rows evenly over the split, default = first
rows), cd_residual_ladder run (2 shards) + restart + report on tiny regression checkpoints with and without the time
inputs (a widened checkpoint must give its donor's residuals); 27 Nov: crop (the crop equals the slice of the full-box
sample bit for bit, origins cover every valid origin, visit counts), and with PhysicsNeMo the crop path of train_cd.py:
global_index selects the full-box positional grid at the crop (network input of a crop forward = slice of the network
input of the full-box forward, also after a save / load of the full-box model), CropRegressionLoss, EMA beta / save /
load, visit map.

    python3 cd_selftest.py --work DIR
"""
import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT10 = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "cp", "tp"]
INV = ["cos_latitude", "cos_local_time", "cos_longitude", "lsm", "sin_latitude", "sin_local_time", "sin_longitude", "z"]
NY, NX, F = 32, 64, 4
BOX = [10.0, 70.0, -80.0, 40.0]


def build(work):
    import zarr
    from cd_export import mk
    rng = np.random.default_rng(0)
    rows = []
    for split, dates in (("train", ["2010-01-04", "2012-04-02", "2015-07-06", "2019-10-07"]), ("valid", ["2022-06-29", "2022-12-01"])):
        for d in dates:
            for L in (12, 24, 36, 48, 60, 72):
                hc = dt.datetime.fromisoformat(d)
                rows.append([len(rows), f"1900-01-01T{len(rows) % 24:02d}:00:00", "2023-06-29T00:00:00", hc.isoformat(), L,
                             (hc + dt.timedelta(hours=L)).isoformat(), split])
    n = len(rows)
    dest = Path(work) / "syn.zarr"
    root = zarr.open_group(str(dest), mode="w")
    inv = [f"v{i}" for i in range(38)]
    a = mk(root, "input", (n, 38, NY // F, NX // F), (1, 38, NY // F, NX // F), "float32")
    a[:] = rng.standard_normal((n, 38, NY // F, NX // F)).astype(np.float32)
    a = mk(root, "output", (n, 10, NY, NX), (1, 10, NY, NX), "float32")
    a[:] = rng.standard_normal((n, 10, NY, NX)).astype(np.float32)
    a = mk(root, "invariant", (8, NY, NX), (8, NY, NX), "float32")
    a[:] = rng.standard_normal((8, NY, NX)).astype(np.float32)
    s, nn, w, e = BOX
    root.attrs.update(dict(output_variables=OUT10, input_variables=inv, invariant_variables=INV, box=BOX, ny=NY, nx=NX,
                           factor=F, lat0=nn - 0.5 * (nn - s) / NY, lon0=w + 0.5 * (e - w) / NX))
    with open(dest.with_suffix(".rows.csv"), "w", newline="") as f:
        c = csv.writer(f)
        c.writerow(["row", "synthetic", "reference", "hindcast", "step", "valid", "split"])
        c.writerows(rows)
    st = {"input": {v: {"mean": 0.1, "std": 2.0} for v in inv}, "output": {v: {"mean": 0.0, "std": 1.0} for v in OUT10},
          "invariant": {v: {"mean": 0.0, "std": 1.0} for v in INV}}
    dest.with_suffix(".stats.json").write_text(json.dumps(st))
    return dest


def test_dataset(dest):
    from cd_dataset import TIME_INPUTS, E2v4Box, spread_index, time_fields
    kw = dict(data_path=str(dest), stats_path=str(dest.with_suffix(".stats.json")))
    d0 = E2v4Box(**kw, split="valid")
    d1 = E2v4Box(**kw, split="valid", time_inputs=True)
    y0, x0 = d0[3]
    y1, x1 = d1[3]
    assert x0.shape == (46, NY, NX) and x1.shape == (51, NY, NX) and np.array_equal(y0, y1)
    assert np.array_equal(x1[:46], x0), "time_inputs changed the first 46 channels"
    assert np.allclose(x1[46:], time_fields(d1.valid[3], d1.latitude(), d1.longitude()), atol=0, rtol=0)
    assert [c.name for c in d1.input_channels()][-5:] == TIME_INPUTS and len(d0.input_channels()) == 46
    assert np.allclose(d1.denormalize_input(d1.normalize_input(x1))[46:], x1[46:], atol=1e-6)
    v = d1.valid[3]
    tf = time_fields(v, np.array([[40.0]]), np.array([[0.0]]))[:, 0, 0]
    assert abs(tf[2] - np.cos(v.hour / 24 * 2 * np.pi)) < 1e-6, tf      # local time at Greenwich = UTC hour
    first = E2v4Box(**kw, split="valid", max_rows=5)
    spr = E2v4Box(**kw, split="valid", max_rows=5, spread=True)
    assert list(first.rows) == list(d0.rows[:5]) and list(spr.rows) == [int(d0.rows[i]) for i in spread_index(12, 5)]
    assert spr.rows[0] == d0.rows[0] and spr.rows[-1] == d0.rows[-1]
    print(f"DATASET_OK 46 / 51 channels, time inputs at {v} (local time at lon 0: {tf[2]:.3f}), spread rows "
          f"{list(spr.rows)} vs first {list(first.rows)}", flush=True)


def test_crop(dest):
    import torch
    from cd_dataset import E2v4Box
    kw = dict(data_path=str(dest), stats_path=str(dest.with_suffix(".stats.json")), split="train", time_inputs=True)
    full = E2v4Box(**kw)
    cr = E2v4Box(**kw, crop=16)
    assert cr.image_shape() == full.image_shape() == (NY, NX)
    torch.manual_seed(3)
    seen = set()
    for i in range(400):
        idx = i % len(cr)
        y, x, o = cr[idx]
        oy, ox = int(o[0]), int(o[1])
        assert 0 <= oy <= NY - 16 and 0 <= ox <= NX - 16 and y.shape == (10, 16, 16) and x.shape == (51, 16, 16)
        seen.add((oy, ox))
        if i < 40:
            yf, xf = full[idx]
            assert np.array_equal(y, yf[:, oy:oy + 16, ox:ox + 16]) and np.array_equal(x, xf[:, oy:oy + 16, ox:ox + 16]), (i, oy, ox)
    corner = any(o[0] in (0, NY - 16) and o[1] in (0, NX - 16) for o in seen)
    print(f"CROP_OK 40 crops equal to the full-box slice bit for bit (51 channels incl. time inputs); {len(seen)} distinct "
          f"origins of {(NY - 15) * (NX - 15)} in 400 draws; a corner origin drawn: {corner}", flush=True)


def test_train_ext(work):
    import torch
    from physicsnemo import Module
    from physicsnemo.models.diffusion_unets import CorrDiffRegressionUNet
    import cd_train_ext as cdx
    torch.manual_seed(0)
    kw = dict(img_resolution=[NY, NX], img_out_channels=10, model_type="SongUNetPosEmbd", N_grid_channels=4,
              embedding_type="zero", model_channels=16, channel_mult=[1, 2], attn_resolutions=[], checkpoint_level=0)
    net = CorrDiffRegressionUNet(img_in_channels=51 + 4, **kw).eval()
    pe = net.model.pos_embd
    assert pe.shape == (4, NY, NX) and "pos_embd" not in net.state_dict(), "regression grid: non-persistent buffer"
    lin = lambda n: np.linspace(0, 2 * np.pi, n)
    ref = np.stack([np.tile(np.sin(lin(NX)), (NY, 1)), np.tile(np.sin(lin(NY))[:, None], (1, NX)),
                    np.tile(np.cos(lin(NX)), (NY, 1)), np.tile(np.cos(lin(NY))[:, None], (1, NX))])
    assert np.allclose(pe.numpy(), ref, atol=1e-6), "grid channels = sin x, sin y, cos x, cos y over the box"
    first = next(iter(net.model.enc.values()))
    seen = []
    h = first.register_forward_pre_hook(lambda m, a: seen.append(a[0].detach().clone()))
    xf = torch.randn(1, 51, NY, NX)
    with torch.no_grad():
        net(torch.zeros(1, 10, NY, NX), xf)
        full_in = seen.pop()
        origins = torch.tensor([[0, 0], [16, 48], [7, 23]])
        xc = torch.stack([xf[0, :, oy:oy + 16, ox:ox + 16] for oy, ox in origins.tolist()])
        gi = cdx.crop_global_index(origins, 16, 16, "cpu")
        net(torch.zeros(3, 10, 16, 16), xc, global_index=gi)
        crop_in = seen.pop()
        for b, (oy, ox) in enumerate(origins.tolist()):
            assert torch.equal(crop_in[b], full_in[0, :, oy:oy + 16, ox:ox + 16]), (b, oy, ox)
    h.remove()
    try:
        with torch.no_grad():
            net(torch.zeros(3, 10, 16, 16), xc)       # control: a crop without global_index cannot take the box grid
        ctrl = "ran (unexpected)"
    except Exception as e:
        ctrl = f"fails ({type(e).__name__})"
    print(f"EMBED_OK network input of a crop forward (data + 4 grid channels) = the full-box input at the crop, 3 crops "
          f"in one batch, each its own origin; control without global_index: {ctrl}", flush=True)
    p = Path(work) / "full_box.mdlus"
    net.save(str(p))
    re = Module.from_checkpoint(str(p)).eval()
    assert torch.equal(re.model.pos_embd, pe)
    print("EMBED_LOAD_OK a model saved at the box size rebuilds the same grid on load", flush=True)
    loss = cdx.CropRegressionLoss()(net=re, img_clean=torch.randn(3, 10, 16, 16), img_lr=xc, global_index=gi)
    assert loss.shape == (3, 10, 16, 16) and torch.isfinite(loss).all()
    ema = cdx.EMA(re, halflife_nimg=1000, rampup_ratio=0.05)
    assert ema.beta_at(0, 8) == 0.0 and abs(ema.beta_at(10 ** 6, 8) - 0.5 ** (8 / 1000)) < 1e-12
    assert abs(ema.beta_at(4000, 8) - 0.5 ** (8 / 200)) < 1e-12
    with torch.no_grad():
        for q in re.parameters():
            q.add_(1.0)
    b = ema.update(re, 10 ** 6, 8)
    q0, e0 = next(iter(re.parameters())), next(iter(ema.module.parameters()))
    ck = str(Path(work) / "ck" / "checkpoints_regression")
    path = ema.save(ck, 1234)
    assert path.endswith("checkpoints_regression_ema/CorrDiffRegressionUNet.0.1234.mdlus"), path
    e2 = cdx.EMA(re)
    assert e2.load(ck, 1234, "cpu") and torch.equal(next(iter(e2.module.parameters())), e0)
    assert e2.load(ck, 999, "cpu") is None
    assert not torch.equal(e0, q0) and torch.allclose(e0, q0 - 1.0 + (1 - b) * 1.0, atol=1e-6)
    v = cdx.CropVisits((NY, NX), 16)
    o = torch.tensor([[0, 0], [16, 48], [16, 48]])
    v.add(o)
    vis = cdx.CropVisits.visits_from(v.hist, 16, (NY, NX))
    brute = np.zeros((NY, NX), np.int64)
    for oy, ox in o.tolist():
        brute[oy:oy + 16, ox:ox + 16] += 1
    assert np.array_equal(vis, brute)
    e = v.expected()
    n, lo, hi = v.write(str(Path(work) / "visits.npz"), 0, 1)
    assert n == 3 and abs(e.sum() - 16 * 16) < 1e-6 and lo == e[0, 0] and hi == e.max()
    print(f"EXT_OK crop loss, EMA (beta {b:.6f} = 0.5^(8/1000), save/load to checkpoints_regression_ema), visit map "
          f"(expected visit probability per crop {lo:.4f} at the corners, {hi:.4f} at most)", flush=True)


def test_ladder(work, dest):
    import torch
    import yaml
    from physicsnemo.models.diffusion_unets import CorrDiffRegressionUNet
    torch.manual_seed(0)
    ck = Path(work) / "run" / "reg" / "checkpoints_regression"
    ck.mkdir(parents=True, exist_ok=True)
    kw = dict(img_resolution=[NY, NX], img_out_channels=10, model_type="SongUNetPosEmbd", N_grid_channels=4,
              embedding_type="zero", model_channels=16, channel_mult=[1, 2], attn_resolutions=[], checkpoint_level=0)
    for s in (100, 200):
        m = CorrDiffRegressionUNet(img_in_channels=46 + 4, **kw)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.05 * torch.randn_like(p))
        m.save(str(ck / f"CorrDiffRegressionUNet.0.{s}.mdlus"))
    py = sys.executable
    wide = ck.parent.parent / "reg_x" / "checkpoints_regression" / "CorrDiffRegressionUNet.0.200.w5.mdlus"
    subprocess.run([py, str(HERE / "cd_widen_regression.py"), "widen", "--donor", str(ck / "CorrDiffRegressionUNet.0.200.mdlus"),
                    "--add", "5", "--out", str(wide)], check=True)
    cfg = Path(work) / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"dataset": {"type": "x::E2v4Box", "data_path": str(dest),
                                               "stats_path": str(dest.with_suffix(".stats.json")), "time_inputs": True}}))
    out = Path(work) / "ladder"
    ckpts = f"{ck}/*.mdlus {wide}"
    for sh in ("0/2", "1/2"):
        subprocess.run([py, str(HERE / "cd_residual_ladder.py"), "run", "--cfg", str(cfg), "--out", str(out), "--ckpts", ckpts,
                        "--n", "12", "--shard", sh], check=True)
    subprocess.run([py, str(HERE / "cd_residual_ladder.py"), "report", "--out", str(out)], check=True, capture_output=True)
    r = subprocess.run([py, str(HERE / "cd_residual_ladder.py"), "run", "--cfg", str(cfg), "--out", str(out), "--ckpts", ckpts,
                        "--n", "12"], check=True, capture_output=True, text=True)
    assert "0 checkpoints to do" in r.stdout, r.stdout
    lad = json.load(open(out / "ladder.json"))["checkpoints"]
    assert len(lad) == 3, list(lad)
    don = next(v for k, v in lad.items() if k.endswith("CorrDiffRegressionUNet.0.200.mdlus"))
    wid = next(v for k, v in lad.items() if k.endswith(".w5.mdlus"))
    assert wid["time_inputs"] and not don["time_inputs"] and wid["dataset_channels"] == 51
    dmax = max(abs(don[s][v] - wid[s][v]) for s in ("train", "valid") for v in OUT10)
    assert dmax < 1e-4, dmax
    assert len(don["valid_by_lead"]) == 6 and len(don["train_rows"]) == 12 and len(don["valid_rows"]) == 12
    print(f"LADDER_OK 3 checkpoints, restart skips all, widened (time inputs) = donor within {dmax:.1e}; "
          f"{out / 'ladder.md'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--skip-ladder", action="store_true", help="dataset part only (no PhysicsNeMo)")
    a = ap.parse_args()
    Path(a.work).mkdir(parents=True, exist_ok=True)
    dest = build(a.work)
    test_dataset(dest)
    test_crop(dest)
    if not a.skip_ladder:
        test_train_ext(a.work)
        test_ladder(a.work, dest)
    print("SELFTEST_OK", flush=True)


if __name__ == "__main__":
    main()
