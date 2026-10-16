#!/usr/bin/env python3
"""Transplant our single-time network weights into Vera Gahlen's single-slot downscaler.

The inverse of the name rule of
  /ec/res4/scratch/ecm5702/eval-vera-pair-20260922/scripts/transplant.py
which moved Vera's weights into OUR network (verified 18 Sep on o48 -> o96 and 23 Sep on
o96 -> o320, 499 network tensors, exact or rounding-level forward agreement). That rule maps an
interface-level key of ours to hers:

  encoder.out_hres.            -> encoder.0.
  decoder.out_hres.            -> decoder.0.
  .<mlp>.<i>.                  -> .<mlp>.mlp.<i>.   for <mlp> in node_dst_mlp, node_src_mlp, node_mlp,
                                                     mlp_dst, mlp_src

over the five network roots encoder, decoder, processor, noise_cond_mlp, noise_embedder. Here
the direction is ours -> hers: every network key of Vera's model is mapped back to ours by the
inverse rule (vera_to_ours_key), the forward rule is checked to undo it for every key, and each
tensor is taken from our checkpoint after a shape check.

Inputs
  --ours       our materialised single-time checkpoint (stage a: the donor 12dcefea with the do0
               68 -> 70 extension), Lightning-level keys model.model.<root>...
  --container  a checkpoint of Vera's SINGLE-SLOT model for the same E2 layout (build_vera_model.py,
               config mt2_e2_20k with task.input_offsets = output_offsets = ["0h"])
  --out        the container with every network tensor replaced (the stage b warm start)

Everything that is not a network tensor (normaliser statistics, graph buffers) keeps the
container's value: the branch computes those from the same stores and residual statistics. The
script lists, for information, the normaliser buffers whose names match on both sides and how far
they differ. It refuses to write if a network tensor of either side is left unmatched or a shape
differs.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

ROOTS = ("encoder", "decoder", "processor", "noise_cond_mlp", "noise_embedder")
MLPS = "node_dst_mlp|node_src_mlp|node_mlp|mlp_dst|mlp_src"
PREFIX = "model.model."  # Lightning module -> AnemoiModelInterface -> network


def ours_to_vera_key(name: str) -> str:
    """The rule of transplant.py (rename_net_key), on network-relative keys."""
    n = name.replace("encoder.out_hres.", "encoder.0.").replace("decoder.out_hres.", "decoder.0.")
    return re.sub(rf"\.({MLPS})\.(\d+)\.", r".\1.mlp.\2.", n)


def vera_to_ours_key(name: str) -> str:
    """Its inverse."""
    n = name.replace("encoder.0.", "encoder.out_hres.").replace("decoder.0.", "decoder.out_hres.")
    return re.sub(rf"\.({MLPS})\.mlp\.(\d+)\.", r".\1.\2.", n)


def is_net(key: str) -> bool:
    return key.startswith(PREFIX) and key[len(PREFIX):].split(".")[0] in ROOTS


def main() -> int:
    import torch

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ours", required=True)
    p.add_argument("--container", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if Path(args.out).exists():
        print(f"refusing: {args.out} exists (move it aside first)")
        return 2

    ours_ck = torch.load(args.ours, map_location="cpu", weights_only=False)
    ours = ours_ck["state_dict"]
    cont = torch.load(args.container, map_location="cpu", weights_only=False)
    vsd = cont["state_dict"]

    v_net = [k for k in vsd if is_net(k)]
    o_net = [k for k in ours if is_net(k)]
    new, unmatched, clashes, not_inverse = {}, [], [], []
    for vk in v_net:
        rel = vk[len(PREFIX):]
        ok_rel = vera_to_ours_key(rel)
        if ours_to_vera_key(ok_rel) != rel:
            not_inverse.append((vk, ok_rel))
        ok = PREFIX + ok_rel
        if ok not in ours:
            unmatched.append((vk, ok))
            continue
        if tuple(ours[ok].shape) != tuple(vsd[vk].shape):
            clashes.append((vk, ok, list(vsd[vk].shape), list(ours[ok].shape)))
            continue
        new[vk] = ours[ok].to(vsd[vk].dtype).clone()
    used = {PREFIX + vera_to_ours_key(k[len(PREFIX):]) for k in new}
    ours_left = [k for k in o_net if k not in used]
    n_el = sum(v.numel() for v in new.values())
    report = {"ours": args.ours, "container": args.container, "vera_network_tensors": len(v_net),
              "ours_network_tensors": len(o_net), "transplanted": len(new), "elements": n_el,
              "vera_unmatched": unmatched, "shape_clashes": clashes, "rule_not_inverse": not_inverse,
              "ours_network_tensors_unused": ours_left}

    # Normaliser buffers: information only (the container's values are kept).
    stats = []
    for k, v in vsd.items():
        if is_net(k) or not (k.endswith("_norm_mul") or k.endswith("_norm_add")):
            continue
        cand = [k, k.replace("pre_processors_residual", "pre_processors_tendencies").replace(
            "post_processors_residual", "post_processors_tendencies")]
        hit = next((c for c in cand if c in ours and tuple(ours[c].shape) == tuple(v.shape)), None)
        if hit is None:
            stats.append((k, "no same-shape twin in ours"))
            continue
        d = (ours[hit].double() - v.double()).abs()
        rel = (d / v.double().abs().clamp_min(1e-30)).max().item()
        stats.append((k, f"max rel diff {rel:.3e} vs {hit}"))
    report["normaliser_buffers"] = stats

    print(json.dumps({k: (v if not isinstance(v, list) or len(v) < 12 else f"{len(v)} items, first {v[:6]}")
                      for k, v in report.items() if k != "normaliser_buffers"}, indent=1, default=str))
    for row in stats:
        print("  normaliser", *row)
    if unmatched or clashes or not_inverse or ours_left or len(new) != len(v_net):
        print("TRANSPLANT INCOMPLETE: refusing to write")
        return 1

    out_sd = dict(vsd)
    out_sd.update(new)
    cont["state_dict"] = out_sd
    cont["provenance_transplant"] = {
        "note": "network weights of our materialised donor (12dcefea step 400000, do0 68->70 extension) in Vera's "
                "single-slot network through the inverse of the 2026-09-18 name rule",
        "ours": args.ours, "container": args.container, "n_transplanted": len(new),
        "date": datetime.datetime.utcnow().isoformat() + "Z"}
    torch.save(cont, args.out)
    Path(args.out + ".report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"TRANSPLANT OK: {len(new)} network tensors ({n_el} elements) written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
