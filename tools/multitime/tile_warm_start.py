#!/usr/bin/env python3
"""Warm-start a K-slot (multi-in multi-out) spatial downscaler from a single-slot one.

Vera Gahlen's `AnemoiTransportSpatialDownscalerModelEncProcDec` flattens the time slots
into the feature axis: the encoder's first linear layer sees
``[in_lres (K x n1) | in_hres (K x n2) | noised target (K x n3) | node attributes (n4)]``
columns in that order, time-major inside each block (slot 0's variables, then slot 1's),
and the decoder's last linear layer writes ``K x n3`` rows (slot 0's variables, then
slot 1's). Every other tensor has the same shape for any K, except any other layer that reads the same
assembled input vector: on the branch tip that is the decoder's destination-node embedding
(``decoder.0.emb_nodes_dst``), which is tiled exactly like the encoder's input embedding.

This script takes a single-slot state dict (K = 1; for the E2 lane that is the donor
`12dcefea` transplanted into Vera's network with the 2026-09-18 name map, 68 -> 70
variables extended by name beforehand) and a "container" checkpoint of the K-slot model
(any checkpoint of the K-slot configuration, e.g. one written by a 0-step dry run) and
writes the container's state dict with every tensor replaced by the single-slot one, the
two shape-changing layers tiled:

- ``--mode average`` (default): each slot block of the input columns is W_block / K, so at
  step 0 the model equals the single-slot model applied to the TIME MEAN of the slots'
  inputs and noised targets. Symmetric in the slots; not an exact warm start (the mean of
  two independent noise fields has std sigma / sqrt(2) while the network is told sigma).
- ``--mode first``: slot 0's block is W_block, the other slots' blocks are zero, so the
  model equals the single-slot model applied to slot 0. Exact for slot 0, blind to the
  others until training moves the zero columns.

The output rows are copied to every slot in both modes (every slot gets the single-slot
estimate). The smoke check after loading: the step-0 training loss must be finite and
far below the from-scratch value (about 1), and with ``--mode first`` the slot-0 output on a
batch must equal the single-slot model's output to float32 rounding.

Usage:
  python3 tile_warm_start.py --single single_slot_state.pt --container k_slot_init.ckpt \
      --widths 68,11,70 --out tiled_state.pt [--mode average|first]
``--widths`` are the per-slot widths n1,n2,n3 (in_lres input variables, in_hres input
variables, out_hres output channels) as the K-slot model sees them; the script checks that
K * (n1+n2+n3) + n4 matches the container and (n1+n2+n3) + n4 the single-slot layer.
"""

from __future__ import annotations

import argparse
import sys

import torch


def _state_dict(obj) -> dict:
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise TypeError(f"cannot read a state dict from {type(obj)}")


def _find_unique(state: dict, suffix: str, predicate) -> str:
    keys = [k for k in state if k.endswith(suffix) and predicate(state[k])]
    if len(keys) != 1:
        raise KeyError(f"expected exactly one key ending in {suffix!r} satisfying the shape test, found {keys}")
    return keys[0]


def tile_input_columns(w_single: torch.Tensor, widths: list[int], k: int, n_attr: int, mode: str) -> torch.Tensor:
    """Tile the (hidden, n_single) weight into (hidden, K * sum(widths) + n_attr)."""
    hidden, n_single = w_single.shape
    assert n_single == sum(widths) + n_attr, (n_single, widths, n_attr)
    blocks = []
    col = 0
    for width in widths:
        block = w_single[:, col : col + width]
        col += width
        if mode == "average":
            blocks.extend([block / k] * k)
        elif mode == "first":
            blocks.append(block)
            blocks.extend([torch.zeros_like(block)] * (k - 1))
        else:
            raise ValueError(mode)
    blocks.append(w_single[:, col:])  # node attributes, unchanged
    out = torch.cat(blocks, dim=1)
    assert out.shape == (hidden, k * sum(widths) + n_attr)
    return out


def tile_output_rows(w_single: torch.Tensor, n_out: int, k: int) -> torch.Tensor:
    """Repeat the (n_out, ...) output layer K times along dim 0 (time-major rows)."""
    assert w_single.shape[0] == n_out, (w_single.shape, n_out)
    return torch.cat([w_single] * k, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--single", required=True, help="single-slot state dict or checkpoint (.pt/.ckpt)")
    parser.add_argument("--container", required=True, help="K-slot checkpoint whose shapes to match")
    parser.add_argument("--widths", required=True, help="n1,n2,n3 per-slot widths (in_lres, in_hres, out_hres)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("average", "first"), default="average")
    parser.add_argument("--k", type=int, default=None, help="slots; inferred from the container when omitted")
    args = parser.parse_args()

    single = _state_dict(torch.load(args.single, map_location="cpu", weights_only=False))
    container_obj = torch.load(args.container, map_location="cpu", weights_only=False)
    container = _state_dict(container_obj)
    widths = [int(w) for w in args.widths.split(",")]
    n_in_single = sum(widths)
    n_out = widths[-1]

    # Shape-changing layers: the encoder input embedding and the decoder output layer.
    enc_key_c = _find_unique(container, "emb_nodes_src.weight", lambda t: t.ndim == 2 and t.shape[1] > n_in_single)
    enc_key_s = _find_unique(single, "emb_nodes_src.weight", lambda t: t.ndim == 2 and t.shape[1] >= n_in_single)
    n_attr = single[enc_key_s].shape[1] - n_in_single
    k = args.k or (container[enc_key_c].shape[1] - n_attr) // n_in_single
    if k * n_in_single + n_attr != container[enc_key_c].shape[1]:
        raise ValueError(
            f"container input width {container[enc_key_c].shape[1]} != K*{n_in_single}+{n_attr} for K={k}; check --widths"
        )
    dec_w_c = _find_unique(container, "node_data_extractor.1.weight", lambda t: t.shape[0] == k * n_out)
    dec_b_c = dec_w_c[: -len("weight")] + "bias"
    dec_w_s = _find_unique(single, "node_data_extractor.1.weight", lambda t: t.shape[0] == n_out)
    dec_b_s = dec_w_s[: -len("weight")] + "bias"

    # Every other 2-D weight that reads the same assembled input vector is tiled the same way. On the
    # branch tip (6587cc679) that is the decoder's destination-node embedding (decoder.0.emb_nodes_dst,
    # target_node_features: [encoded_data]), which sees the encoder's un-embedded source features.
    k_width = k * n_in_single + n_attr
    input_keys = {enc_key_c: enc_key_s}
    for key, tensor in container.items():
        if key in input_keys or key not in single:
            continue
        s_t = single[key]
        if (tensor.ndim == 2 and s_t.ndim == 2 and tensor.shape[0] == s_t.shape[0]
                and s_t.shape[1] == n_in_single + n_attr and tensor.shape[1] == k_width):
            input_keys[key] = key

    out = {}
    changed, copied, skipped = [], [], []
    for key, tensor in container.items():
        if key in input_keys:
            out[key] = tile_input_columns(single[input_keys[key]], widths, k, n_attr, args.mode)
            changed.append(key)
        elif key == dec_w_c:
            out[key] = tile_output_rows(single[dec_w_s], n_out, k)
            changed.append(key)
        elif key == dec_b_c:
            out[key] = tile_output_rows(single[dec_b_s], n_out, k)
            changed.append(key)
        elif key in single and single[key].shape == tensor.shape:
            out[key] = single[key].clone()
            copied.append(key)
        else:
            # Buffers the single-slot checkpoint lacks (graph buffers of the new container, statistics) keep
            # the container's values. Anything trainable here is a mapping error: list it.
            out[key] = tensor
            skipped.append((key, tuple(tensor.shape), tuple(single[key].shape) if key in single else None))

    print(f"K = {k}; per-slot widths {widths}; node-attribute columns {n_attr}; mode {args.mode}")
    print(f"tiled: {changed}")
    print(f"copied {len(copied)} tensors from the single-slot state")
    trainable_skipped = [s for s in skipped if not any(tag in s[0] for tag in ("pre_processors", "post_processors"))]
    print(f"kept from the container: {len(skipped)} tensors ({len(trainable_skipped)} outside the processors):")
    for key, c_shape, s_shape in trainable_skipped[:40]:
        print(f"  {key}: container {c_shape}, single {s_shape}")

    if isinstance(container_obj, dict) and "state_dict" in container_obj:
        container_obj["state_dict"] = out
        torch.save(container_obj, args.out)
    else:
        torch.save(out, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
