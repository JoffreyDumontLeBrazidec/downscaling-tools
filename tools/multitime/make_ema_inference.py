#!/usr/bin/env python3
"""Say which weights a Vera-branch inference checkpoint holds, and write its EMA copy.

On Vera Gahlen's branch the run has one EMA (``training.weight_averaging``, PyTorch Lightning
``WeightAveraging``). The anemoi checkpoint callback pickles the live network into
``inference-<name>.ckpt`` during training, while Lightning's ``on_save_checkpoint`` puts the
averaged parameters in the training checkpoint's ``state_dict`` and the live ones in
``current_model_state``. This tool compares the inference file's parameters with both, prints
the verdict (raw, EMA or neither), and with ``--out`` writes a copy of the inference interface
whose parameters are the EMA ones (buffers are kept from the inference file). The fork lane names
that file ``inference-ema<half-life>-<name>.ckpt``.

Usage:
  python make_ema_inference.py --train-ckpt <run>/anemoi-by_step-epoch_019-step_020000.ckpt \
      --inference <run>/inference-anemoi-by_step-epoch_019-step_020000.ckpt \
      [--out <work>/inference-ema2000-anemoi-by_step-epoch_019-step_020000.ckpt]
"""

from __future__ import annotations

import argparse
import json
import sys

import torch


def _strip(sd: dict, keys: set[str]) -> dict:
    """Map a Lightning-module state dict onto the interface's parameter names."""
    for prefix in ("model.", "module.model.", ""):
        mapped = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if keys <= set(mapped):
            return mapped
    missing = sorted(keys - set(sd))[:5]
    raise SystemExit(f"cannot map the state dict onto the interface parameters (e.g. {missing})")


def _diff(params: dict, other: dict) -> tuple[float, float]:
    num, den, mx = 0.0, 0.0, 0.0
    for k, v in params.items():
        o = other[k].to(v.dtype)
        d = (v - o).double()
        num += float((d * d).sum())
        den += float((v.double() ** 2).sum())
        mx = max(mx, float(d.abs().max()))
    return (num / max(den, 1e-30)) ** 0.5, mx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-ckpt", required=True)
    ap.add_argument("--inference", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ck = torch.load(args.train_ckpt, map_location="cpu", weights_only=False)
    print("training checkpoint keys:", sorted(ck.keys()))
    print("global_step:", ck.get("global_step"), "epoch:", ck.get("epoch"))
    for key in ("averaging_state",):
        if key in ck:
            st = ck[key]
            print(f"{key}:", {k: (v if not torch.is_tensor(v) else v.tolist()) for k, v in st.items()}
                  if isinstance(st, dict) else type(st))
    cbs = ck.get("callbacks", {})
    print("callbacks:", list(cbs.keys()) if isinstance(cbs, dict) else type(cbs))

    interface = torch.load(args.inference, map_location="cpu", weights_only=False)
    params = {k: v.detach() for k, v in interface.named_parameters()}
    keys = set(params)
    print(f"inference interface {type(interface).__name__}: {len(params)} parameter tensors, "
          f"{sum(v.numel() for v in params.values())} values")

    verdict = {}
    if "state_dict" in ck:
        ema = _strip(ck["state_dict"], keys)
        verdict["vs_state_dict"] = _diff(params, ema)
    if "current_model_state" in ck:
        raw = _strip(ck["current_model_state"], keys)
        verdict["vs_current_model_state"] = _diff(params, raw)
    if "state_dict" in ck and "current_model_state" in ck:
        verdict["state_dict_vs_current_model_state"] = _diff(_strip(ck["state_dict"], keys), raw)
    for k, (rel, mx) in verdict.items():
        print(f"{k}: relative RMS difference {rel:.3e}, max abs difference {mx:.3e}")

    holds = "unknown"
    if "vs_current_model_state" in verdict and verdict["vs_current_model_state"][1] == 0.0:
        holds = "raw"
    elif "vs_state_dict" in verdict and verdict["vs_state_dict"][1] == 0.0:
        holds = "ema"
    has_ema = "current_model_state" in ck and verdict.get("state_dict_vs_current_model_state", (0, 0))[1] > 0
    print(json.dumps({"inference_file_holds": holds, "training_ckpt_has_separate_ema": bool(has_ema)}))

    if args.out:
        if not has_ema:
            raise SystemExit("no separate EMA state in the training checkpoint: nothing to write")
        with torch.no_grad():
            for name, p in interface.named_parameters():
                p.copy_(ema[name].to(p.dtype))
        check = _diff({k: v.detach() for k, v in interface.named_parameters()}, ema)
        if check[1] != 0.0:
            raise SystemExit(f"EMA copy check FAILED: {check}")
        torch.save(interface, args.out)
        print(f"wrote {args.out}: parameters = training checkpoint state_dict (EMA), buffers from the inference file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
