#!/usr/bin/env python3
"""Self-test of the kit's opt-in options on a synthetic export (CPU, small; needs the PhysicsNeMo venv for the ladder
part). Checks: cd_dataset default unchanged (46 channels), time_inputs (51 channels, the 5 extra = time_fields at the
row's valid time, the first 46 unchanged, channel metadata), spread (rows evenly over the split, default = first
rows), cd_residual_ladder run (2 shards) + restart + report on tiny regression checkpoints with and without the time
inputs (a widened checkpoint must give its donor's residuals).

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
    if not a.skip_ladder:
        test_ladder(a.work, dest)
    print("SELFTEST_OK", flush=True)


if __name__ == "__main__":
    main()
