"""CPU checks on the checkpoint file alone (no forward pass, numpy only): review note s4 checks 1, 3 and 6.

1  zero-content heads and MLP units (exactly zero weights on both sides: the 512 -> 1,024 widening saddle); EMA
   ``*_dest`` alias mismatches; NaN/Inf in the model, the EMA shadows and the optimizer state.
3  gradient-dead units and heads from the checkpoint's OWN optimizer state: second moment exactly 0 on a unit's bias
   (AdamW: also its whole W1 row and W2 column; SOAP: the GG diagonals) = no gradient for at least the WINDOW
   (~1.5k steps at beta2 0.95, ~70k at 0.999; reported per block), not necessarily for the whole run.
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
FP32_TINY = 1.4e-45  # smallest fp32 subnormal: a decaying moment below it is exactly 0


def _window(v_ref: float | None, beta: float | None) -> int | None:
    """Steps without gradient after which a moment of size v_ref underflows to exactly 0 under decay `beta`.

    So "moment == 0" means "no gradient for at least ~this many steps" (not "since the state was created"): with
    beta2 0.95 and v ~ 1e-12 that is ~1.5k steps; with beta2 0.999, ~70k."""
    if not v_ref or not beta or not (0 < beta < 1):
        return None
    return int(math.log(v_ref / FP32_TINY) / -math.log(beta))


def _beta2(view, soap_gg: bool = False) -> float | None:
    """The decay of the moment read: AdamW/Muon-aux exp_avg_sq -> betas[1]; SOAP's GG -> shampoo_beta (default -1,
    meaning betas[1], as in soap.py)."""
    for g in view.groups:
        b = g.get("betas")
        if isinstance(b, (list, tuple)) and len(b) == 2:
            sb = g.get("shampoo_beta")
            if soap_gg and sb is not None and float(sb) >= 0:
                return float(sb)
            return float(b[1])
    return None


def optimizer_dead(r: Reader, blocks: list[Block], zero: dict, trainable: list | None = None) -> dict:
    """Units and heads whose parameters got no gradient recently, from the optimizer moments.

    Criterion per MLP unit, by optimizer:
      AdamW  exp_avg_sq == 0 on the bias AND the whole W1 row AND the whole W2 column (the three must agree);
      SOAP   diag(GG[0]) == 0 for the W1 row and diag(GG[1]) == 0 for the W2 column (GG is kept in the original
             basis; exp_avg_sq of a matrix is in the rotated eigenbasis and says nothing per unit), and the bias'
             exp_avg_sq (unrotated unless precondition_1d);
      Muon   the bias' exp_avg_sq (aux AdamW group); the matrices' momentum is not a per-unit record.
    A zero moment means "no gradient for at least the window" (`dead_window_steps`, from the group's beta2 and the
    block's median live moment), not "for the whole run". Zero-content units (check 1) are not counted again.
    """
    view = r.optimizer(trainable)
    out = {
        "optimizer": view.kind,
        "states": view.n_states,
        "mapped": view.n_mapped,
        "entries": view.n_entries,
        "mapping": view.status,
        "mapping_complete": view.complete,
        "groups": view.groups,
        "excluded_frozen": view.excluded_frozen[:40],
        "unmapped": [[k, list(sh)] for k, sh in view.skipped[:60]],
        "per_role": {},
        "indices": {},
        "agreement": {},
        "window_steps": {},
    }
    if not view.n_states:
        out["skipped"] = "no optimizer state in this file (inference file or weights-only save)"
        return out
    beta2 = _beta2(view)
    beta_gg = _beta2(view, soap_gg=True)

    def st(key):
        return view.by_name.get(key)

    def leaf(key, name):
        s = st(key)
        t = s.get(name) if isinstance(s, dict) else None
        return r.array(t) if t is not None and hasattr(t, "is_float") else None

    def gg_diag(key, i):
        s = st(key)
        gg = s.get("GG") if isinstance(s, dict) else None
        if isinstance(gg, (list, tuple)) and len(gg) > i and hasattr(gg[i], "is_float"):
            return np.diagonal(r.array(gg[i])).copy()
        return None

    for b in blocks:
        m, idx, agree = {}, {}, {}
        zidx = zero["indices"].get(b.role, {})
        for name, hidden in b.mlps.items():
            zero_u = set(zidx.get(f"{name}.zero_units", []))
            kb, k1, k2 = (b.key(f"{name}.{x}") for x in ("0.bias", "0.weight", "2.weight"))
            vb = gg_diag(kb, 0) if view.kind == "soap" else None
            if vb is None:
                vb = leaf(kb, "exp_avg_sq")
            if vb is None:
                continue
            crit = {"bias": vb == 0}
            if view.kind == "adamw":
                v1, v2 = leaf(k1, "exp_avg_sq"), leaf(k2, "exp_avg_sq")
                if v1 is not None and v2 is not None:
                    crit["w1_row"], crit["w2_col"] = ~np.any(v1 != 0, axis=1), ~np.any(v2 != 0, axis=0)
            elif view.kind == "soap":
                d1, d2 = gg_diag(k1, 0), gg_diag(k2, 1)
                if d1 is not None and d1.shape[0] == hidden:
                    crit["w1_row"] = d1 == 0
                if d2 is not None and d2.shape[0] == hidden:
                    crit["w2_col"] = d2 == 0
            dead = np.logical_and.reduce(list(crit.values()))
            agree[name] = {k: int(v.sum()) for k, v in crit.items()} | {"all": int(dead.sum())}
            grad_dead = [int(i) for i in np.where(dead)[0] if int(i) not in zero_u]
            m[f"{name}.grad_dead_unit_frac"] = len(grad_dead) / hidden
            idx[f"{name}.grad_dead_units"] = grad_dead
            live = vb[vb > 0]
            win = _window(float(np.median(live)) if live.size else None, beta_gg if view.kind == "soap" else beta2)
            if win is not None:
                out["window_steps"][f"{b.role}.{name}"] = win
            mb = leaf(kb, "exp_avg")
            if mb is not None and view.kind != "soap":
                recent_dead = [int(i) for i in np.where(mb == 0)[0] if int(i) not in zero_u]
                m[f"{name}.recent_dead_unit_frac"] = len(recent_dead) / hidden
        if b.heads:
            d = b.head_dim
            zero_h = set(zidx.get("zero_heads", []))
            vs = []
            for n in ("lin_query", "lin_key", "lin_value"):
                v = gg_diag(b.key(f"{n}.bias"), 0) if view.kind == "soap" else None
                vs.append(v if v is not None else leaf(b.key(f"{n}.bias"), "exp_avg_sq"))
            vs = [v for v in vs if v is not None]
            if vs:
                dead_h = [h for h in range(b.heads) if h not in zero_h and all(not np.any(v[h * d : (h + 1) * d]) for v in vs)]
                m["grad_dead_head_frac"] = len(dead_h) / b.heads
                idx["grad_dead_heads"] = dead_h
        # coordinates whose second moment is exactly 0, as a fraction of the block (1-D tensors; AdamW matrices too)
        tot = zero_coords = 0
        for k in [k for k in view.by_name if k.startswith(b.prefix + ".")]:
            v = leaf(k, "exp_avg_sq")
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
    w = list(out["window_steps"].values())
    out["window_min_steps"] = min(w) if w else None
    return out


# ------------------------------------------------------------------------------------------------ check 3b
def _silu(x):
    return x / (1.0 + np.exp(-x))


def _cond_curve(r: Reader, cs: np.ndarray):
    """cond(c) for c = c_noise = log(sigma)/4 from noise_embedder.frequencies and noise_cond_mlp; None if absent."""
    fk = next((k for k in r.state if k.endswith("noise_embedder.frequencies")), None)
    l1 = next((k for k in r.state if k.endswith("noise_cond_mlp.linear1_no_gradscaling.weight")), None)
    if fk is None or l1 is None:
        return None
    base = l1[: -len("linear1_no_gradscaling.weight")]
    f = r.get(fk)
    W1, b1 = r.get(base + "linear1_no_gradscaling.weight"), r.get(base + "linear1_no_gradscaling.bias")
    W2, b2 = r.get(base + "linear2_no_gradscaling.weight"), r.get(base + "linear2_no_gradscaling.bias")
    out = cs[:, None] * f[None, :]
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=-1)
    return _silu(emb @ W1.T + b1) @ W2.T + b2


def saturation_bound(r: Reader, blocks: list[Block], cmin: float = -1.0, cmax: float = 1.0, n: int = 201) -> dict:
    """Check 3b (docs kit audit_saturated_units.py): a rigorous upper bound of each MLP unit's GELU input over every
    LayerNorm output and every noise level c_noise in [cmin, cmax] (default -1..1 = sigma 0.02..55, the training
    range). UB < -5.55: the unit can never fire (output exactly 0, so revival keeps the function); UB < -14: it can
    never get a gradient. The bound is adversarial, so a unit dead for data reasons usually has UB > 0: this is
    CPU evidence when it fires, never a count of all dead units."""
    cs = np.linspace(cmin, cmax, n)
    cond = _cond_curve(r, cs)
    per_role, indices = {}, {}
    for b in blocks:
        m, idx = {}, {}
        for name, hidden in b.mlps.items():
            W1, b1 = r.get(b.key(f"{name}.0.weight")), r.get(b.key(f"{name}.0.bias"))
            if b1 is None:
                b1 = np.zeros(W1.shape[0])
            C = W1.shape[1]
            ln = b.key("layer_norm_mlp_dst" if name == "node_dst_mlp" else "layer_norm_mlp_src")
            sw = r.get(f"{ln}.scale.weight")
            if sw is not None:
                sb, bw, bb = r.get(f"{ln}.scale.bias"), r.get(f"{ln}.bias.weight"), r.get(f"{ln}.bias.bias")
                if cond is None:
                    G, B = (1.0 + sb)[None, :], bb[None, :]
                else:
                    G, B = 1.0 + cond @ sw.T + sb[None, :], cond @ bw.T + bb[None, :]
            elif r.get(f"{ln}.weight") is not None:
                G = r.get(f"{ln}.weight")[None, :]
                lb = r.get(f"{ln}.bias")
                B = (lb if lb is not None else np.zeros(C))[None, :]
            else:
                G, B = np.ones((1, C)), np.zeros((1, C))
            ub = np.full(W1.shape[0], -np.inf)
            for i in range(G.shape[0]):
                A = W1 * G[i][None, :]
                A = A - A.mean(axis=1, keepdims=True)
                ub = np.maximum(ub, b1 + W1 @ B[i] + math.sqrt(C) * np.sqrt((A * A).sum(axis=1)))
            m[f"{name}.ub_never_fires_frac"] = float((ub < -5.55).mean())
            m[f"{name}.ub_no_grad_frac"] = float((ub < -14).mean())
            m[f"{name}.ub_max_below_zero_frac"] = float((ub < 0).mean())
            idx[f"{name}.never_fires_units"] = np.where(ub < -5.55)[0].tolist()
        per_role[b.role] = m
        indices[b.role] = idx
    return {"per_role": per_role, "indices": indices, "noise_cond_curve": cond is not None, "c_noise_range": [cmin, cmax]}


# ------------------------------------------------------------------------------------------------ check 6
def _init_rms(fan_in: int) -> float:
    """RMS of nn.Linear's default init U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    return 1.0 / math.sqrt(3.0 * fan_in)


def _ln_gain(r: Reader, prefix: str):
    """(per-channel gain, mask of channels whose gain is fixed) of a LayerNorm (weight) or a ConditionalLayerNorm
    (gain 1 + scale(cond): fixed only where the conditioning row scale.weight is zero; elsewhere it varies with
    the condition and is never read as zero or low)."""
    w = r.get(f"{prefix}.weight")
    if w is not None and w.ndim == 1:
        return w, np.ones(w.shape, bool)
    sb = r.get(f"{prefix}.scale.bias")
    if sb is not None:
        sw = r.get(f"{prefix}.scale.weight")
        fixed = ~np.any(sw != 0, axis=1) if sw is not None else np.ones(sb.shape, bool)
        return 1.0 + sb, fixed
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
            g, fixed = _ln_gain(r, b.key(ln))
            if g is None:
                continue
            a = np.abs(g)
            med = float(np.median(a))
            m[f"{ln}.zero_gain_frac"] = float(((a == 0) & fixed).sum() / a.size)
            m[f"{ln}.low_gain_frac"] = float(((a < 0.01 * med) & fixed).sum() / a.size) if med else 0.0
            m[f"{ln}.gain_mean_abs"] = float(a.mean())
        per_role[b.role] = m
        if det:
            detail[b.role] = det
    return {"per_role": per_role, "detail": detail}


# ------------------------------------------------------------------------------------------------ all CPU checks
def run_static(path: str, skip_nonfinite: bool = False, trainable: list | None = None) -> tuple[Reader, list[Block], dict]:
    r = Reader(path)
    blocks = find_blocks(r.state, r.config)
    z = zero_content(r, blocks)
    nf = None if skip_nonfinite else nonfinite_and_aliases(r)
    od = optimizer_dead(r, blocks, z, trainable)
    od["mapping_source"] = "loaded model (named_parameters, requires_grad)" if trainable is not None else "file (shapes, config)"
    sb = saturation_bound(r, blocks)
    sl = scale_ladders(r, blocks)
    z["blocks_without_heads"] = [b.role for b in blocks if not b.heads]
    checks = {
        "c1_zero_content": z | ({"nonfinite": nf["nonfinite"], "alias_mismatch": nf["alias_mismatch"]} if nf else {}),
        "c3_optimizer_dead": od,
        "c3b_saturation_bound": sb,
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
