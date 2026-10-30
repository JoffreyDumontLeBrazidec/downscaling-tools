"""Find the GraphTransformer blocks of a state dict and give each a ROLE.

Every health metric is keyed by role (encoder, processor.<i>, decoder, mapper:<name>), never by tensor shape, so a
512-channel and a 1,024-channel model read on the same bands (review note s8, owner 30 Oct 2026).

Block layout (anemoi ``GraphTransformerBaseBlock``): lin_query / lin_key / lin_value / lin_self / lin_edge
(rows = heads x head width), projection (input columns = heads x head width), optional q_norm / k_norm (QK-norm gains
of one head width), layer_norm_attention, layer_norm_mlp_dst, node_dst_mlp = Sequential(W1, act, W2); mapper blocks
may add node_src_mlp. Works on the training state dict ('model.model.' prefix) and on pickled inference modules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MLP_NAMES = ("node_dst_mlp", "node_src_mlp")
HEAD_LEAVES_IN = ("lin_query", "lin_key", "lin_value", "lin_self", "lin_edge")


@dataclass
class Block:
    prefix: str  # full state-dict prefix, e.g. model.model.processor.proc.2
    role: str  # encoder | decoder | processor.<i> | mapper:<name>
    channels: int  # rows of lin_query = heads x head width
    heads: int | None
    head_dim: int | None
    mlps: dict = field(default_factory=dict)  # name -> hidden units

    def key(self, leaf: str) -> str:
        return f"{self.prefix}.{leaf}"


def _natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _short(prefix: str) -> str:
    for p in ("model.model.", "model."):
        if prefix.startswith(p):
            return prefix[len(p) :]
    return prefix


def _config_heads(config: dict, section: str):
    m = (config or {}).get("model", {}) or {}
    sec = m.get(section) or {}
    return sec.get("num_heads") if isinstance(sec, dict) else None


def find_blocks(state: dict, config: dict | None = None) -> list[Block]:
    """All GraphTransformer blocks, ordered encoder, processor.0..N, decoder, others; roles assigned."""
    prefixes = sorted(
        {k[: -len(".lin_query.weight")] for k in state if k.endswith(".lin_query.weight")}, key=_natural
    )
    prefixes = [p for p in prefixes if f"{p}.lin_key.weight" in state and f"{p}.projection.weight" in state]
    proc = [p for p in prefixes if "processor" in _short(p).split(".")[0]]
    blocks = []
    for p in prefixes:
        s = _short(p)
        head = s.split(".")[0]
        if head == "encoder" or head.startswith("encoder"):
            role, section = "encoder", "encoder"
        elif head == "decoder" or head.startswith("decoder"):
            role, section = "decoder", "decoder"
        elif p in proc:
            role, section = f"processor.{proc.index(p)}", "processor"
        else:
            role, section = f"mapper:{s}", None
        channels = int(state[f"{p}.lin_query.weight"].shape[0])
        heads = head_dim = None
        qn = state.get(f"{p}.q_norm.weight")
        if qn is not None and qn.shape and channels % int(qn.shape[0]) == 0:
            head_dim = int(qn.shape[0])
            heads = channels // head_dim
        elif section and _config_heads(config, section):
            heads = int(_config_heads(config, section))
            head_dim = channels // heads if channels % heads == 0 else None
            heads = heads if head_dim else None
        mlps = {}
        for name in MLP_NAMES:
            w1, w2 = state.get(f"{p}.{name}.0.weight"), state.get(f"{p}.{name}.2.weight")
            if w1 is not None and w2 is not None and w1.shape[0] == w2.shape[1]:
                mlps[name] = int(w1.shape[0])
        blocks.append(Block(p, role, channels, heads, head_dim, mlps))
    # disambiguate duplicate roles (two encoders of a multi-dataset model)
    seen = {}
    for b in blocks:
        seen[b.role] = seen.get(b.role, 0) + 1
    for b in blocks:
        if seen[b.role] > 1 and not b.role.startswith("processor"):
            b.role = f"{b.role}:{_short(b.prefix)}"
    order = {"encoder": 0, "processor": 1, "decoder": 2, "mapper": 3}
    return sorted(blocks, key=lambda b: (order.get(b.role.split(":")[0].split(".")[0], 3), _natural(b.role)))


def role_group(role: str) -> str:
    """encoder | processor | decoder | mapper (the level at which bands are set)."""
    return role.split(":")[0].split(".")[0]
