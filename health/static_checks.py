"""CPU checks on the checkpoint file alone (no forward pass, numpy only): review note s4 checks 1, 3 and 6.

1  zero-content heads and MLP units (exactly zero weights on both sides: the 512 -> 1,024 widening saddle); EMA
   ``*_dest`` alias mismatches; NaN/Inf in the model, the EMA shadows and the optimizer state.
3  gradient-dead units and heads from the checkpoint's OWN optimizer state: second moment exactly 0 on a unit's bias
   (and, where the optimizer keeps it per coordinate, on its W1 row and W2 column) = no gradient over the whole
   optimizer history (fresh warm start: since its step 1). First moment exactly 0 = none in the last ~1k steps.
6  scale ladders: QK-norm gains, edge-logit scale per head (s11 ladder), weight RMS over its nn.Linear init RMS,
   LayerNorm gains (an exactly zero gain is a dead channel), low-norm MLP units.

Every number is a per-block FRACTION or RATIO keyed by role (model_map.py), so models of other widths read on the
same bands. Unit and head INDICES are kept in the JSON for the revival command; the summary carries fractions.
"""
from __future__ import annotations

import math

import numpy as np

from .ckpt import Reader
from .model_map import Block, find_blocks, role_group


def _zero_rows(a: np.ndarray) -> np.ndarray:
    return ~np.any(a != 0, axis=tuple(range(1, a.ndim))) if a.ndim > 1 else a == 0


def _sn(a: np.ndarray) -> float:
    return float(np.linalg.svd(a, compute_uv=False)[0]) if a.size else 0.0


# ------------------------------------------------------------------------------------------------ check 1
def zero_content(r: Reader, blocks: list[Block]) -> dict:
    per_role, indices = {}, {}
    for b in blocks:
        m, idx = {}, {}
        if b.heads:
            d = b.head_dim
            q, k, p = (r.get(b.key(f"{n}.weight")) for n in ("lin_query", "lin_key", "projection"))
            others = [r.get(b.key(f"{n}.weight")) for n in ("lin_value", "lin_self", "lin_edge")]
            others = [o for o in others if o is not None]
            dead, partial = [], []
            for h in range(b.heads):
                rows = slice(h * d, (h + 1) * d)
                parts = [not np.any(q[rows]), not np.any(k[rows]), not np.any(p[:, rows])] + [not np.any(o[rows]) for o in others]
                if all(parts):
                    dead.append(h)
                elif any(parts):
                    partial.append(h)
            m["zero_head_frac"] = len(dead) / b.heads
            m["partial_head_frac"] = len(partial) / b.heads
            idx["zero_heads"], idx["partial_heads"] = dead, partial
        for name, hidden in b.mlps.items():
            w1, w2 = r.get(b.key(f"{name}.0.weight")), r.get(b.key(f"{name}.2.weight"))
            dead_u = np.where(_zero_rows(w1) & _zero_rows(w2.T))[0]
            m[f"{name}.zero_unit_frac"] = len(dead_u) / hidden
            idx[f"{name}.zero_units"] = dead_u.tolist()
        per_role[b.role] = m
        indices[b.role] = idx
    return {"per_role": per_role, "indices": indices}


def nonfinite_and_aliases(r: Reader) -> dict:
    """NaN/Inf counts in model, EMA shadows and optimizer state; *_dest alias mismatches in model and EMA."""
    out = {"nonfinite": {}, "alias_mismatch": {}}
    dicts = {"model": r.state} | r.ema_shadows()
    for label, sd in dicts.items():
        n_bad = n_tensors = 0
        for k, t in sd.items():
            if not t.is_float:
                continue
            a = r.array(t, dtype=None)
            n_tensors += 1
            n_bad += int(a.size - np.count_nonzero(np.isfinite(a)))
        out["nonfinite"][label] = {"values": n_bad, "tensors": n_tensors}
        pairs = bad = 0
        for k, t in sd.items():
            if "_dest." not in k:
                continue
            twin = k.replace("_dest.", ".")
            if twin not in sd:
                continue
            pairs += 1
            a, b = r.array(t), r.array(sd[twin])
            if a.shape != b.shape or not np.array_equal(a, b):
                bad += 1
        out["alias_mismatch"][label] = {"pairs": pairs, "mismatched": bad}
    opts = r.obj.get("optimizer_states") if isinstance(r.obj, dict) else None
    if opts:
        n_bad = 0
        for s in (opts[0].get("state") or {}).values():
            if not isinstance(s, dict):
                continue
            for v in s.values():
                if hasattr(v, "is_float") and v.is_float:
                    a = r.array(v, dtype=None)
                    n_bad += int(a.size - np.count_nonzero(np.isfinite(a)))
        out["nonfinite"]["optimizer"] = {"values": n_bad}
    return out


# ------------------------------------------------------------------------------------------------ check 3
def optimizer_dead(r: Reader, blocks: list[Block], zero: dict) -> dict:
    """Units and heads whose parameters got no gradient, from the optimizer moments. Zero-content units excluded."""
    view = r.optimizer()
    out = {
        "optimizer": view.kind,
        "states": view.n_states,
        "mapped": view.n_mapped,
        "mapping_complete": view.complete,
        "groups": view.groups,
        "per_role": {},
        "indices": {},
        "agreement": {},
    }
    if not view.n_states:
        out["skipped"] = "no optimizer state in this file (inference file or weights-only save)"
        return out

    def hist(key):
        s = view.by_name.get(key)
        return (None, None) if s is None else r.moment(s, "history")

    def recent(key):
        s = view.by_name.get(key)
        return (None, None) if s is None else r.moment(s, "recent")

    for b in blocks:
        m, idx, agree = {}, {}, {}
        zidx = zero["indices"].get(b.role, {})
        for name, hidden in b.mlps.items():
            zero_u = set(zidx.get(f"{name}.zero_units", []))
            _, vb = hist(b.key(f"{name}.0.bias"))
            _, mb = recent(b.key(f"{name}.0.bias"))
            if vb is None:
                continue
            dead = vb == 0
            # per-coordinate moments on the matrices (AdamW; not SOAP's rotated basis, not Muon's momentum alone)
            n1, v1 = hist(b.key(f"{name}.0.weight"))
            n2, v2 = hist(b.key(f"{name}.2.weight"))
            if view.kind == "adamw" and v1 is not None and v2 is not None:
                w1_dead, w2_dead = ~np.any(v1 != 0, axis=1), ~np.any(v2 != 0, axis=0)
                agree[name] = {
                    "bias": int(dead.sum()),
                    "w1_row": int(w1_dead.sum()),
                    "w2_col": int(w2_dead.sum()),
                    "all3": int((dead & w1_dead & w2_dead).sum()),
                }
                dead = dead & w1_dead & w2_dead
            grad_dead = [int(i) for i in np.where(dead)[0] if int(i) not in zero_u]
            m[f"{name}.grad_dead_unit_frac"] = len(grad_dead) / hidden
            idx[f"{name}.grad_dead_units"] = grad_dead
            if mb is not None:
                recent_dead = [int(i) for i in np.where(mb == 0)[0] if int(i) not in zero_u]
                m[f"{name}.recent_dead_unit_frac"] = len(recent_dead) / hidden
        if b.heads:
            d = b.head_dim
            zero_h = set(zidx.get("zero_heads", []))
            vs = [hist(b.key(f"{n}.bias"))[1] for n in ("lin_query", "lin_key", "lin_value")]
            vs = [v for v in vs if v is not None]
            if vs:
                dead_h = [h for h in range(b.heads) if h not in zero_h and all(not np.any(v[h * d : (h + 1) * d]) for v in vs)]
                m["grad_dead_head_frac"] = len(dead_h) / b.heads
                idx["grad_dead_heads"] = dead_h
        # dead-parameter fraction of the block: coordinates whose history moment is exactly zero (1-D tensors and,
        # for AdamW, matrices too)
        tot = zero_coords = 0
        for k in [k for k in view.by_name if k.startswith(b.prefix + ".")]:
            _, v = hist(k)
            if v is None or (view.kind != "adamw" and v.ndim > 1):
                continue
            tot += v.size
            zero_coords += int(v.size - np.count_nonzero(v))
        if tot:
            m["zero_moment_coord_frac"] = zero_coords / tot
        out["per_role"][b.role] = m
        out["indices"][b.role] = idx
        if agree:
            out["agreement"][b.role] = agree
    # parameters the optimizer never saw (frozen or never reached by a gradient)
    out["unmapped_float_tensors"] = len(view.skipped)
    out["unmapped_examples"] = view.skipped[:40]
    return out


# ------------------------------------------------------------------------------------------------ check 6
def _init_rms(fan_in: int) -> float:
    """RMS of nn.Linear's default init U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    return 1.0 / math.sqrt(3.0 * fan_in)


def _ln_gain(r: Reader, prefix: str):
    """Per-channel gain of a LayerNorm (weight) or ConditionalLayerNorm (1 + scale.bias, scale.weight)."""
    w = r.get(f"{prefix}.weight")
    if w is not None and w.ndim == 1:
        return w, "plain"
    sb = r.get(f"{prefix}.scale.bias")
    if sb is not None:
        sw = r.get(f"{prefix}.scale.weight")
        g = 1.0 + sb
        if sw is not None:  # a channel is dead for every condition only if its conditioning row is zero too
            g = np.where(np.any(sw != 0, axis=1), np.maximum(np.abs(g), 1e-30), g)
        return g, "conditional"
    return None, None


def scale_ladders(r: Reader, blocks: list[Block]) -> dict:
    per_role, detail = {}, {}
    for b in blocks:
        m, det = {}, {}
        for leaf in ("lin_query", "lin_key", "lin_value", "lin_self", "projection"):
            w = r.get(b.key(f"{leaf}.weight"))
            if w is not None:
                m[f"{leaf}.rms_over_init"] = float(np.sqrt(np.mean(w * w)) / _init_rms(w.shape[1]))
        for name in b.mlps:
            w1, w2 = r.get(b.key(f"{name}.0.weight")), r.get(b.key(f"{name}.2.weight"))
            m[f"{name}.w1.rms_over_init"] = float(np.sqrt(np.mean(w1 * w1)) / _init_rms(w1.shape[1]))
            m[f"{name}.w2.rms_over_init"] = float(np.sqrt(np.mean(w2 * w2)) / _init_rms(w2.shape[1]))
            rown = np.sqrt((w1 * w1).sum(1))
            live = rown[rown > 0]
            med = float(np.median(live)) if live.size else 0.0
            m[f"{name}.low_norm_unit_frac"] = float(((rown > 0) & (rown < 0.2 * med)).sum() / len(rown)) if med else 0.0
        gq, gk = r.get(b.key("q_norm.weight")), r.get(b.key("k_norm.weight"))
        if gq is not None:
            m["g_q.rms"] = float(np.sqrt(np.mean(gq * gq)))
            if gk is not None:
                m["g_k.rms"] = float(np.sqrt(np.mean(gk * gk)))
                m["g_q_g_k.max"] = float(np.abs(gq * gk).max())
        we = r.get(b.key("lin_edge.weight"))
        if we is not None and b.heads:
            d = b.head_dim
            g = gq if gq is not None else np.ones(d)
            sn_gain = r.get(b.key("edge_sn_gain"))
            geo, wsn = [], []
            for h in range(b.heads):
                wh = we[h * d : (h + 1) * d]
                if sn_gain is not None:  # edge_spectral_reparam: g_h W_h / sigma_max(W_h)
                    s = _sn(wh)
                    wh = wh * (float(sn_gain[h]) / s if s else 0.0)
                wsn.append(_sn(wh))
                geo.append(_sn(np.diag(g) @ wh) / math.sqrt(d))
            m["edge_logit_scale.max"] = float(max(geo))
            m["W_e.sn.max"] = float(max(wsn))
            det["edge_logit_scale"] = [round(x, 4) for x in geo]
        for ln in ("layer_norm_attention", "layer_norm_mlp_dst", "layer_norm_attention_src", "layer_norm_mlp_src"):
            g, kind = _ln_gain(r, b.key(ln))
            if g is None:
                continue
            a = np.abs(g)
            med = float(np.median(a))
            m[f"{ln}.zero_gain_frac"] = float((a == 0).sum() / a.size)
            m[f"{ln}.low_gain_frac"] = float((a < 0.01 * med).sum() / a.size) if med else 1.0
            m[f"{ln}.gain_mean_abs"] = float(a.mean())
        per_role[b.role] = m
        if det:
            detail[b.role] = det
    return {"per_role": per_role, "detail": detail}


# ------------------------------------------------------------------------------------------------ all CPU checks
def run_static(path: str, skip_nonfinite: bool = False) -> tuple[Reader, list[Block], dict]:
    r = Reader(path)
    blocks = find_blocks(r.state, r.config)
    z = zero_content(r, blocks)
    nf = None if skip_nonfinite else nonfinite_and_aliases(r)
    od = optimizer_dead(r, blocks, z)
    sl = scale_ladders(r, blocks)
    checks = {
        "c1_zero_content": z | ({"nonfinite": nf["nonfinite"], "alias_mismatch": nf["alias_mismatch"]} if nf else {}),
        "c3_optimizer_dead": od,
        "c6_scale": sl,
    }
    return r, blocks, checks


def capacity_summary(blocks: list[Block], checks: dict, group: str = "processor") -> dict:
    """Dead capacity of one role group: units and heads dead by content or by gradient, as fractions."""
    z = checks["c1_zero_content"]["indices"]
    o = checks["c3_optimizer_dead"].get("indices", {})
    units = dead_zero = dead_grad = heads = heads_zero = heads_grad = 0
    for b in blocks:
        if role_group(b.role) != group:
            continue
        for name, hidden in b.mlps.items():
            units += hidden
            dead_zero += len(z.get(b.role, {}).get(f"{name}.zero_units", []))
            dead_grad += len(o.get(b.role, {}).get(f"{name}.grad_dead_units", []))
        if b.heads:
            heads += b.heads
            heads_zero += len(z.get(b.role, {}).get("zero_heads", []))
            heads_grad += len(o.get(b.role, {}).get("grad_dead_heads", []))
    return {
        "group": group,
        "units": units,
        "units_dead_zero": dead_zero,
        "units_dead_grad": dead_grad,
        "unit_dead_frac": (dead_zero + dead_grad) / units if units else None,
        "heads": heads,
        "heads_dead_zero": heads_zero,
        "heads_dead_grad": heads_grad,
        "head_dead_frac": (heads_zero + heads_grad) / heads if heads else None,
        "optimizer_state": checks["c3_optimizer_dead"].get("states", 0) > 0,
    }
