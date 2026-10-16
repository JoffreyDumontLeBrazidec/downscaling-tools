#!/usr/bin/env python3
"""Denoiser forward checks for the mt2_e2_20k warm start (our code vs Vera Gahlen's branch).

The same fixed input (normalised fields on the o1280 E2 grid, drawn once with a fixed seed and
saved to disk) is fed to the denoiser D(y; sigma) at sigma = 1 of:

  ours     our single-time model (do0 recipe, donor 12dcefea materialised with the 68 -> 70 output
           extension), through AnemoiD2ModelEncProcDec.fwd_with_preconditioning;
  vera     Vera's single-slot network (input_offsets ["0h"]) holding the transplanted weights,
           through AnemoiTransportSpatialDownscalerModelEncProcDec.forward (EDM preconditioning);
  vera2    Vera's two-slot network (input_offsets ["0h", "12h"]) holding the tiled weights, with the
           same state in both slots (both slots' outputs are saved).

Each side runs in its own venv and only needs the pickled AnemoiModelInterface (written by
materialise_do0_donor.py or build_vera_model.py). float32, TF32 off.

  make-input --grid N --out input.pt [--seed 20261016]
  run --side ours|vera|vera2 --interface iface.pt --input input.pt --out out.pt [--device cuda|cpu]
  compare --ref out_ours.pt --other out_vera.pt [--other ...] [--json summary.json]

Input layout (normalised space, batch 1, ensemble 1): x_in_lres (1,1,1,G,68) already on the o1280
grid, x_in_hres (1,1,1,G,11), y_noised (1,1,1,G,70) = y_clean + sigma * noise, sigma = 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

N_LRES, N_HRES, N_OUT = 68, 11, 70


def make_input(grid: int, out: str, seed: int, sigma: float) -> None:
    import torch

    g = torch.Generator().manual_seed(seed)
    x_lres = torch.randn(1, 1, 1, grid, N_LRES, generator=g)
    x_hres = torch.randn(1, 1, 1, grid, N_HRES, generator=g)
    y_clean = torch.randn(1, 1, 1, grid, N_OUT, generator=g)
    noise = torch.randn(1, 1, 1, grid, N_OUT, generator=g)
    y = y_clean + sigma * noise
    torch.save({"x_in_lres": x_lres, "x_in_hres": x_hres, "y_noised": y, "sigma": sigma, "seed": seed,
                "note": "standard normal fields in normalised space; in_lres already on the target grid"}, out)
    print(f"wrote {out}: grid {grid}, seed {seed}, sigma {sigma}")


def run(side: str, iface_path: str, input_path: str, out: str, device: str, backend: str | None,
        shard_strategy: str | None = None) -> None:
    import torch

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    t0 = time.time()
    iface = torch.load(iface_path, map_location="cpu", weights_only=False)
    net = iface.model
    if backend:
        n = 0
        for m in net.modules():
            for attr in ("graph_attention_backend", "attention_backend"):
                if hasattr(m, attr) and isinstance(getattr(m, attr), str):
                    setattr(m, attr, backend)
                    n += 1
        print(f"[{side}] graph attention backend set to {backend} on {n} attributes")
    if shard_strategy:
        # Compute-only switch: with one process the "heads" and "edges" strategies compute the same function;
        # only "edges" splits the mapper's edges into chunks (ANEMOI_INFERENCE_NUM_CHUNKS[_MAPPER]) at inference,
        # which keeps the E2 grid within the login node's memory.
        n = 0
        for m in net.modules():
            if isinstance(getattr(m, "shard_strategy", None), str) and m.shard_strategy != shard_strategy:
                m.shard_strategy = shard_strategy
                n += 1
        print(f"[{side}] shard strategy set to {shard_strategy} on {n} modules")
    if shard_strategy and not torch.distributed.is_initialized():
        # The "edges" mapper path builds its graph partition through torch.distributed collectives, which
        # need a process group even for one process: a single-rank gloo group (no communication happens).
        import os
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)
        print(f"[{side}] single-rank gloo process group initialised")
    net = net.to(dev).to(torch.float32).eval()
    inp = torch.load(input_path, map_location="cpu", weights_only=False)
    sig = torch.full((1, 1, 1, 1, 1), float(inp["sigma"]), dtype=torch.float32, device=dev)
    k = 2 if side == "vera2" else 1
    x = {"in_lres": inp["x_in_lres"].to(dev).repeat(1, k, 1, 1, 1),
         "in_hres": inp["x_in_hres"].to(dev).repeat(1, k, 1, 1, 1)}
    y = {"out_hres": inp["y_noised"].to(dev).repeat(1, k, 1, 1, 1)}
    s = {"out_hres": sig}
    print(f"[{side}] loaded in {time.time() - t0:.1f} s; device {dev}; slots {k}; "
          f"shapes {[tuple(v.shape) for v in (*x.values(), *y.values())]}", flush=True)
    t1 = time.time()
    with torch.no_grad():
        if side == "ours":
            pred = net.fwd_with_preconditioning(x, y, s)
        else:
            pred = net(x, y, s)
    o = pred["out_hres"].detach().float().cpu()
    print(f"[{side}] forward {time.time() - t1:.1f} s; output {tuple(o.shape)} mean {o.mean():.6f} std {o.std():.6f}; "
          f"peak memory {torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else float('nan'):.2f} GiB",
          flush=True)
    torch.save({"side": side, "interface": iface_path, "input": input_path, "output": o, "device": str(dev),
                "torch": torch.__version__}, out)


def compare(ref: str, others: list[str], json_out: str | None) -> None:
    import torch

    a = torch.load(ref, weights_only=False)
    xa = a["output"].double()[:, 0]  # single slot
    sd = xa.std().item()
    rows = {}
    for path in others:
        b = torch.load(path, weights_only=False)
        ob = b["output"].double()
        for slot in range(ob.shape[1]):
            d = ob[:, slot] - xa
            per_var = d.pow(2).mean(dim=(0, 1, 2)).sqrt() / xa.std(dim=(0, 1, 2)).clamp_min(1e-12)
            row = {"max_abs": d.abs().max().item(), "rms": d.pow(2).mean().sqrt().item(), "ref_std": sd,
                   "rms_over_std": d.pow(2).mean().sqrt().item() / sd, "max_abs_over_std": d.abs().max().item() / sd,
                   "worst_variable_index": int(per_var.argmax()), "worst_variable_rms_over_std": per_var.max().item()}
            rows[f"{Path(path).name}[slot {slot}]"] = row
            print(f"{Path(ref).name} vs {Path(path).name} slot {slot}: rms/std {row['rms_over_std']:.3e}, "
                  f"max abs {row['max_abs']:.3e} (max/std {row['max_abs_over_std']:.3e}), ref std {sd:.4f}, worst "
                  f"variable #{row['worst_variable_index']} rms/std {row['worst_variable_rms_over_std']:.3e}")
    if json_out:
        Path(json_out).write_text(json.dumps({"ref": ref, "rows": rows}, indent=1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("make-input")
    s.add_argument("--grid", type=int, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=20261016)
    s.add_argument("--sigma", type=float, default=1.0)
    s = sub.add_parser("run")
    s.add_argument("--side", choices=("ours", "vera", "vera2"), required=True)
    s.add_argument("--interface", required=True)
    s.add_argument("--input", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--backend", default=None, help="override the graph attention backend (pyg on CPU)")
    s.add_argument("--shard-strategy", default=None, help="override the mappers' shard strategy (edges = chunked)")
    s = sub.add_parser("compare")
    s.add_argument("--ref", required=True)
    s.add_argument("--other", action="append", required=True)
    s.add_argument("--json", default=None)
    a = p.parse_args()
    if a.cmd == "make-input":
        make_input(a.grid, a.out, a.seed, a.sigma)
    elif a.cmd == "run":
        run(a.side, a.interface, a.input, a.out, a.device, a.backend, a.shard_strategy)
    else:
        compare(a.ref, a.other, a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
