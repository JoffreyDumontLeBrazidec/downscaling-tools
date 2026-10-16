#!/usr/bin/env python3
"""Downscale a whole forecast (leads 12-72 h of one reference date) with a multi-time model.

For a spatial downscaler trained on Vera Gahlen's `feat/spatial_downscaler` branch with
K time slots per sample (``task.input_offsets == task.output_offsets``, K entries), this
driver samples all L leads of one forecast block jointly:

- ``--stride 1`` (default): overlapping windows of K slots; at every denoiser call each
  slot's denoised estimate is the mean over the windows covering it (the owner's method:
  denoise (t, t+1) and (t+1, t+2), t+1 takes the average, for all 30 steps). Sampler
  ``heun_window`` of the fork branch.
- ``--stride K``: non-overlapping windows, one noise per slot, no averaging: the model's
  own "isolated windows" output, the within-model control of the averaging.
- ``--leads 12,24`` (any subset of consecutive leads of length K) reproduces the plain
  per-window prediction.

One noise field is drawn per slot of the trajectory (the sampled source has the time
length L), so the windows that share a slot denoise the SAME noisy state. The sampler is
the 30-step churn-off Heun (rule 6: S_churn 0, 59 calls per window), Karras schedule.

Inputs: the ``inference-*.ckpt`` of the run (the saved ``AnemoiModelInterface``), the three
stores and the lane's ``fake_hindcasts`` selection (test year 2023 by default), the block
index or the reference date. The target store is read only to write the truth beside the
prediction. Output: one ``.npz`` per block and member with arrays ``pred`` (L, grid, n_out),
``truth`` (L, grid, n_out), ``input_on_target`` is NOT written (use the interpolation matrix
of the lane if needed), ``variables``, ``leads_h``, ``lat``, ``lon``, ``refdate``.

Needs the fork's anemoi-datasets (fake_hindcasts) and the fork anemoi-core branch
``claude/project-thread-n1zaq3`` (window sampler). Run on one GPU; the E2 box fits in 40 GB
at K = 2 and L = 6 only because the network is called per window: memory is that of one
window plus the L-slot solver state.

Example (Atos AC, sandbox venv):
  python3 predict_sequence.py --ckpt .../inference-anemoi-by_step-...-step_020000.ckpt \
      --data-dir /ec/res4/scratch/ecm5702/o320_o1280_europe_e2_v4/data \
      --forcings /ec/res4/scratch/ecm5702/o320_o1280_europe_e2/data/in_hres_o1280_forcings.zarr \
      --select-file config/e2_select.json --year 2023 --block 10 --members 4 --stride 1 \
      --out /ec/res4/scratch/ecm5702/multitime/mt2_e2_20k/block010_stride1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

STEPS_H = [12, 24, 36, 48, 60, 72]


def _open(path: str, year: int | None, select: list[str] | None, steps: list[int]):
    from anemoi.datasets import open_dataset

    kwargs = {"fake_hindcasts": {"steps": steps}}
    if year is not None:
        kwargs["fake_hindcasts"]["start"] = year
        kwargs["fake_hindcasts"]["end"] = year
    if select:
        kwargs["select"] = select
    return open_dataset(path, **kwargs)


def _block_rows(block: int, k: int) -> list[int]:
    return list(range(block * k, (block + 1) * k))


def _to_batch(ds, rows: list[int]) -> torch.Tensor:
    x = np.asarray(ds[rows])  # (time, variables, ensemble, grid)
    x = x[:, :, 0, :]  # (time, variables, grid)
    x = np.transpose(x, (0, 2, 1))  # (time, grid, variables)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]  # (1, time, grid, variables)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="inference-*.ckpt of the multi-time run")
    parser.add_argument("--data-dir", required=True, help="folder with in_lres_*.zarr and out_hres_*.zarr")
    parser.add_argument("--lres-name", default="in_lres_o320_halo1p5.zarr")
    parser.add_argument("--hres-name", default="out_hres_o1280.zarr")
    parser.add_argument("--forcings", required=True, help="in_hres forcings store (E2's)")
    parser.add_argument("--select-file", default=None, help='JSON {"in_lres": [...], "in_hres": [...], "out_hres": [...]}')
    parser.add_argument("--year", type=int, default=2023, help="fake_hindcasts year (test 2023)")
    parser.add_argument("--steps", default="12,24,36,48,60,72")
    parser.add_argument("--block", type=int, default=0, help="forecast block index within the year's selection")
    parser.add_argument("--leads", default=None, help="subset of consecutive leads, e.g. 12,24 (default: all)")
    parser.add_argument("--members", type=int, default=1)
    parser.add_argument("--seed", type=int, default=756)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--sigma-max", type=float, default=None, help="default: the checkpoint's inference default")
    parser.add_argument("--sigma-min", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--out", required=True, help="output prefix; writes <out>_m<member>.npz")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    k_block = len(steps)
    select = json.load(open(args.select_file)) if args.select_file else {}

    data_dir = Path(args.data_dir)
    ds_lres = _open(str(data_dir / args.lres_name), args.year, select.get("in_lres"), steps)
    ds_forc = _open(args.forcings, args.year, select.get("in_hres"), steps)
    ds_hres = _open(str(data_dir / args.hres_name), args.year, select.get("out_hres"), steps)
    n_rows = int(ds_lres.shape[0])
    assert n_rows % k_block == 0, f"{n_rows} rows not whole blocks of {k_block}"
    assert int(ds_forc.shape[0]) == n_rows == int(ds_hres.shape[0]), "the three stores must share the axis"

    rows = _block_rows(args.block, k_block)
    if args.leads:
        wanted = [int(s) for s in args.leads.split(",")]
        rows = [rows[steps.index(s)] for s in wanted]
        leads = wanted
    else:
        leads = steps
    refdate = str(np.asarray(ds_lres.dates)[rows[0]])
    print(f"block {args.block}, view rows {rows}, leads {leads} h, synthetic first date {refdate}")

    interface = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = interface.model
    device = torch.device(args.device)
    interface.to(device).eval()
    n_slots_trained = int(interface.n_step_input)
    print(f"model trained on {n_slots_trained} slots; sampling {len(rows)} slots with window {n_slots_trained}, stride {args.stride}")

    batch = {
        "in_lres": _to_batch(ds_lres, rows).to(device),
        "in_hres": _to_batch(ds_forc, rows).to(device),
    }
    truth = _to_batch(ds_hres, rows)[0].numpy()  # (L, grid, vars)

    schedule_params = {"num_steps": args.num_steps}
    for key in ("sigma_max", "sigma_min", "rho"):
        value = getattr(args, key)
        if value is not None:
            schedule_params[key] = value
    sampler_params = {"sampler": "heun_window", "window": n_slots_trained, "stride": args.stride, "S_churn": 0.0}

    lat = np.asarray(ds_hres.latitudes)
    lon = np.asarray(ds_hres.longitudes)
    variables = list(ds_hres.variables)
    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for member in range(args.members):
        torch.manual_seed(args.seed + member)
        t0 = time.time()
        with torch.no_grad():
            pred = model.predict_step(
                batch=batch,
                pre_processors=interface.pre_processors,
                post_processors=interface.post_processors,
                n_step_input=len(rows),
                pre_processors_residual=interface.pre_processors_residual,
                post_processors_residual=interface.post_processors_residual,
                spatial_pre_processors=interface.spatial_pre_processors,
                schedule_params=dict(schedule_params),
                sampler_params=dict(sampler_params),
            )
        y = pred["out_hres"]  # (1, L, 1, grid, n_out) or (1, L, grid, n_out)
        y = y.reshape(len(rows), y.shape[-2], y.shape[-1]).float().cpu().numpy()
        path = out_prefix.parent / f"{out_prefix.name}_m{member:02d}.npz"
        np.savez_compressed(
            path,
            pred=y,
            truth=truth,
            variables=np.array(variables),
            leads_h=np.array(leads),
            lat=lat,
            lon=lon,
            refdate=refdate,
            stride=args.stride,
            window=n_slots_trained,
            num_steps=args.num_steps,
            seed=args.seed + member,
        )
        print(f"member {member}: {time.time() - t0:.1f} s, wrote {path}, pred std {y.std():.3f}, truth std {truth.std():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
