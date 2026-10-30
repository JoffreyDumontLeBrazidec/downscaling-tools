"""Revive the dead units and heads an audit named, into a NEW file. By hand only: the audit prints this command,
never runs it (owner, 30 Oct 2026: no automatic revival, no loader change).

Same rule as anemoi-core's ``revive_dead_units`` (fork 18a5415af), extended to the units the optimizer state shows
gradient-dead with nonzero weights: the INPUT side (W1 row and bias; lin_query/key/value/self/edge rows of a head) is
redrawn from nn.Linear's default init U(-1/sqrt(fan_in), 1/sqrt(fan_in)) with bias 0, the OUTPUT side (W2 column;
projection input columns) is set to zero. A dead unit's output is exactly zero (GELU(x) = 0 for x <= -5.55, the
DB300 case) or its weights are zero, so the step-0 function is unchanged up to that exact zero; a gradient-dead unit is revived only with
check-4 evidence that it never fires (`--unproven` overrides, and says so). The EMA shadows get
the same values. The optimizer state is dropped (a weights-only warm start rebuilds it).

    python -m health revive SRC.ckpt DST.ckpt --audit AUDIT.json [--what grad,zero] [--seed 20261030]
"""
from __future__ import annotations

import json
import math

import torch

HEAD_IN = ("lin_query", "lin_key", "lin_value", "lin_self", "lin_edge")


def _uniform_(t: torch.Tensor, fan_in: int, gen: torch.Generator):
    bound = 1.0 / math.sqrt(fan_in)
    t.copy_((torch.rand(t.shape, generator=gen, dtype=torch.float64) * 2 - 1).mul_(bound).to(t.dtype))


def plan(audit: dict, what=("grad", "zero"), unproven: bool = False) -> tuple[dict, list]:
    """({block prefix: {"units": {mlp: [idx]}, "heads": [idx], "head_dim": d}}, warnings) from an audit JSON.

    A gradient-dead unit (nonzero weights) is revived only with evidence that its output is exactly zero, so that
    zeroing its W2 column keeps the function: check 4 `never_fires` (max pre-activation < -5.55 on every node of the
    GPU audit's cases). Without that evidence it is skipped unless `unproven` is set. Gradient-dead HEADS are never
    revived here (no saturation argument exists for them); zero-weight heads and units always qualify."""
    blocks = {b["role"]: b for b in audit["blocks"]}
    never = audit["checks"].get("c4_preact", {}).get("indices", {})
    warnings = []
    out = {}
    sources = []
    if "zero" in what:
        sources.append(("c1_zero_content", "zero"))
    if "grad" in what:
        sources.append(("c3_optimizer_dead", "grad_dead"))
    for cid, tag in sources:
        for role, idx in (audit["checks"].get(cid, {}).get("indices") or {}).items():
            b = blocks.get(role)
            if b is None:
                continue
            p = out.setdefault(b["prefix"], {"units": {}, "heads": set(), "head_dim": b.get("head_dim")})
            for k, v in idx.items():
                if k.endswith(f".{tag}_units"):
                    mlp = k.split(".")[0]
                    v = set(v)
                    if tag == "grad_dead" and not unproven:
                        proven = set(never.get(role, {}).get(f"{mlp}.never_fires_units", []))
                        if v - proven:
                            warnings.append(f"{role}.{mlp}: {len(v - proven)} gradient-dead units skipped: no check-4 "
                                            "evidence that their output is exactly 0 (run the audit with --gpu, or pass --unproven)")
                        v &= proven
                    p["units"].setdefault(mlp, set()).update(v)
                if k == "zero_heads" and tag == "zero":
                    p["heads"].update(v)
                if k == "grad_dead_heads" and v:
                    warnings.append(f"{role}: {len(v)} gradient-dead heads not revived (no function-preserving rule)")
    return out, warnings


def apply(state: dict, shadows: list, todo: dict, seed: int) -> int:
    gen = torch.Generator().manual_seed(seed)
    n = 0
    with torch.no_grad():
        for prefix, p in todo.items():
            for mlp, units in p["units"].items():
                w1, b1, w2 = (state[f"{prefix}.{mlp}.{s}"] for s in ("0.weight", "0.bias", "2.weight"))
                for u in sorted(units):
                    _uniform_(w1[u], w1.shape[1], gen)
                    b1[u] = 0
                    w2[:, u] = 0
                    n += 1
            d = p["head_dim"]
            for h in sorted(p["heads"]):
                rows = slice(h * d, (h + 1) * d)
                for leaf in HEAD_IN:
                    w = state.get(f"{prefix}.{leaf}.weight")
                    if w is None:
                        continue
                    _uniform_(w[rows], w.shape[1], gen)
                    b = state.get(f"{prefix}.{leaf}.bias")
                    if b is not None:
                        b[rows] = 0
                state[f"{prefix}.projection.weight"][:, rows] = 0
                n += 1
        touched = {k for k in state if any(k.startswith(pref + ".") for pref in todo)}
        for sh in shadows:
            for k in touched:
                sk = k[len("model."):] if k.startswith("model.") and k not in sh else k
                if sk in sh:
                    sh[sk].copy_(state[k].to(sh[sk].dtype))
    return n


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m health revive", description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--audit", required=True, help="audit.json of SRC (python -m health audit SRC)")
    ap.add_argument("--what", default="grad,zero", help="grad (optimizer-state dead), zero (zero weights), or both")
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--unproven", action="store_true",
                    help="also revive gradient-dead units without check-4 saturation evidence (may change the function)")
    a = ap.parse_args(argv)
    if a.src == a.dst:
        raise SystemExit("refusing to overwrite the source checkpoint")
    audit = json.load(open(a.audit))
    if audit["checks"].get("c3_optimizer_dead", {}).get("mapping") not in ("exact", None) and "grad" in a.what:
        raise SystemExit("optimizer mapping of this audit is not exact: gradient-dead indices may be shifted; refusing")
    todo, warnings = plan(audit, tuple(a.what.split(",")), a.unproven)
    for w in warnings:
        print("WARNING", w)
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    shadows = []
    for cb in (ck.get("callbacks") or {}).values():
        if isinstance(cb, dict) and isinstance(cb.get("shadows"), dict):
            shadows += [s for s in cb["shadows"].values() if isinstance(s, dict)]
    n = apply(ck["state_dict"], shadows, todo, a.seed)
    ck.pop("optimizer_states", None)
    torch.save(ck, a.dst)
    print(f"revived {n} units/heads in {len(todo)} blocks ({len(shadows)} EMA shadows patched); optimizer state dropped; wrote {a.dst}")
    return 0
