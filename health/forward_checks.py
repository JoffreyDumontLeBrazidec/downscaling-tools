"""GPU checks that need forward/backward passes (review note s4 checks 2, 4, 7, 9) and the graph audit (check 8).

The engine is model-agnostic: it hooks every GraphTransformer block of an ``nn.Module`` and runs a list of CASES
(one date x one sigma each) that a backend builds (backends.py). A case returns the loss, the output and the
conditioning inputs; the engine never touches data loading.

2  gradient-dead (GraMa, arxiv 2505.24061): per MLP hidden unit and per head, mean |dL/dh| over the nodes of each case
   (h = W1 output for units; lin_query/key/value outputs for heads). Dead = exactly 0 on all K cases; dormant = 0 or
   below 1e-3 of the layer mean on >= 3/4 of the cases but not all (owner default 4: 8 batches, dead on 6-7 = dormant).
4  pre-activation statistics per MLP unit over all nodes of all cases: max < -5.55 (erf-GELU output exactly 0 in
   PyTorch's fp32 opmath), max < -14 (its gradient exactly 0), min > 0 (always linear); thresholds from the DB300
   cause thread (optimiser note s11).
7  residual-stream budget per processor block: ||out - in|| / ||in|| with the attention and MLP parts separately
   (the MLP input LayerNorm sees the stream between them), max/median |x| of the block output (massive activations),
   and the Dirichlet energy of the block output on the processor edges over its mean squared norm.
9  information flow at sigma 0.2, 5, 80: every conditioning input variable gets a nonzero gradient from a random
   projection of the output, and every output variable moves when the conditioning inputs are perturbed.
8  graph: per edge type, destination nodes without an in-edge, degree range, edge-attribute range, connected
   components of the processor graph, and a checksum of each edge index (compare with the donor's on warm starts).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .model_map import find_blocks, role_group

GELU_ZERO_OUT = -5.55
GELU_ZERO_GRAD = -14.0
DORMANT_REL = 1e-3


@dataclass
class CaseResult:
    loss: torch.Tensor  # scalar with grad
    output: torch.Tensor  # (..., grid, V_out)
    inputs: dict = field(default_factory=dict)  # name -> leaf tensor (..., grid, V_in)
    input_vars: dict = field(default_factory=dict)  # name -> [variable names]
    output_vars: list = field(default_factory=list)


@dataclass
class Case:
    label: str  # e.g. 20230829_f024_s5.0
    sigma: float
    run: Callable  # run(input_grad: bool, perturb: float, seed: int) -> CaseResult


# ------------------------------------------------------------------------------------------------ accumulators
class _UnitStats:
    def __init__(self, n: int, device):
        self.n = n
        self.max = torch.full((n,), -math.inf, device=device, dtype=torch.float64)
        self.min = torch.full((n,), math.inf, device=device, dtype=torch.float64)
        self.sum = torch.zeros(n, device=device, dtype=torch.float64)
        self.sumsq = torch.zeros(n, device=device, dtype=torch.float64)
        self.count = 0
        self.grad_case = torch.zeros(n, device=device, dtype=torch.float64)
        self.grad_count = 0
        self.case_scores = []  # per case: mean |grad| per unit (cpu numpy)

    def observe(self, h: torch.Tensor):
        x = h.detach().reshape(-1, self.n).double()
        self.max = torch.maximum(self.max, x.max(0).values)
        self.min = torch.minimum(self.min, x.min(0).values)
        self.sum += x.sum(0)
        self.sumsq += (x * x).sum(0)
        self.count += x.shape[0]

    def observe_grad(self, g: torch.Tensor):
        x = g.detach().reshape(-1, self.n).double().abs()
        self.grad_case += x.sum(0)
        self.grad_count += x.shape[0]

    def close_case(self):
        if self.grad_count:
            self.case_scores.append((self.grad_case / self.grad_count).cpu().numpy())
        self.grad_case.zero_()
        self.grad_count = 0


class _HeadStats:
    def __init__(self, heads: int, head_dim: int, device):
        self.heads, self.d = heads, head_dim
        self.case = torch.zeros(heads, device=device, dtype=torch.float64)
        self.count = 0
        self.case_scores = []

    def observe_grad(self, g: torch.Tensor):
        x = g.detach().reshape(-1, self.heads, self.d).double().abs().sum(-1)
        self.case += x.sum(0)
        self.count += x.shape[0]

    def close_case(self):
        if self.count:
            self.case_scores.append((self.case / self.count).cpu().numpy())
        self.case.zero_()
        self.count = 0


class _StreamStats:
    def __init__(self):
        self.x_in = None
        self.x_mid = None
        self.edge_index = None
        self.rows = []

    def record(self, x_out: torch.Tensor, max_edges: int = 200_000):
        if self.x_in is None or x_out.shape != self.x_in.shape:
            return
        with torch.no_grad():
            xi, xo = self.x_in.float(), x_out.detach().float()
            n_in = xi.norm().item()
            row = {"update_ratio": (xo - xi).norm().item() / max(n_in, 1e-30)}
            if self.x_mid is not None and self.x_mid.shape == xi.shape:
                xm = self.x_mid.float()
                row["attn_ratio"] = (xm - xi).norm().item() / max(n_in, 1e-30)
                row["mlp_ratio"] = (xo - xm).norm().item() / max(n_in, 1e-30)
            a = xo.abs().reshape(-1)
            sub = a[torch.randint(0, a.numel(), (min(a.numel(), 1_000_000),), device=a.device)]
            row["massive_ratio"] = a.max().item() / max(sub.median().item(), 1e-30)
            ei = self.edge_index
            x2 = xo.reshape(-1, xo.shape[-1])
            if ei is not None and ei.dim() == 2 and ei.shape[0] == 2 and int(ei.max()) < x2.shape[0]:
                sel = ei[:, torch.randint(0, ei.shape[1], (min(ei.shape[1], max_edges),), device=ei.device)]
                de = (x2[sel[0]] - x2[sel[1]]).pow(2).sum(-1).mean().item()
                row["dirichlet"] = de / max(x2.pow(2).sum(-1).mean().item(), 1e-30)
            self.rows.append(row)
        self.x_in = self.x_mid = None


def _first_tensor(o):
    if torch.is_tensor(o):
        return o
    if isinstance(o, (tuple, list)):
        for v in o:
            if torch.is_tensor(v):
                return v
    return None


def _find_edge_index(args, kwargs):
    cands = list(args) + list(kwargs.values())
    for v in cands:
        if torch.is_tensor(v) and v.dim() == 2 and v.shape[0] == 2 and not v.is_floating_point():
            return v
    return None


# ------------------------------------------------------------------------------------------------ engine
class Probe:
    """Install hooks on every GraphTransformer block of ``root``; remove with close()."""

    def __init__(self, root: torch.nn.Module, device):
        state = {k: v for k, v in root.state_dict().items()}
        self.blocks = find_blocks(state)
        modules = dict(root.named_modules())
        self.units, self.heads, self.streams = {}, {}, {}
        self.handles = []
        self.capture_grads = True
        for b in self.blocks:
            mod = modules.get(b.prefix)
            if mod is None:
                continue
            for name, hidden in b.mlps.items():
                st = _UnitStats(hidden, device)
                self.units[(b.role, name)] = st
                self.handles.append(getattr(mod, name)[0].register_forward_hook(self._unit_hook(st)))
            if b.heads:
                hs = _HeadStats(b.heads, b.head_dim, device)
                self.heads[b.role] = hs
                for leaf in ("lin_query", "lin_key", "lin_value"):
                    self.handles.append(getattr(mod, leaf).register_forward_hook(self._head_hook(hs)))
            if role_group(b.role) == "processor":
                ss = _StreamStats()
                self.streams[b.role] = ss
                self.handles.append(mod.register_forward_pre_hook(self._pre_hook(ss), with_kwargs=True))
                self.handles.append(mod.register_forward_hook(self._post_hook(ss)))
                ln = getattr(mod, "layer_norm_mlp_dst", None)
                if ln is not None:
                    self.handles.append(ln.register_forward_pre_hook(self._mid_hook(ss)))

    def _unit_hook(self, st):
        def hook(_m, _inp, out):
            st.observe(out)
            if self.capture_grads and out.requires_grad:
                out.register_hook(lambda g: st.observe_grad(g))
        return hook

    def _head_hook(self, hs):
        def hook(_m, _inp, out):
            if self.capture_grads and out.requires_grad:
                out.register_hook(lambda g: hs.observe_grad(g))
        return hook

    @staticmethod
    def _pre_hook(ss):
        def hook(_m, args, kwargs):
            x = _first_tensor(args) if args else _first_tensor(list(kwargs.values()))
            ss.x_in = None if x is None else x.detach()
            ei = _find_edge_index(args, kwargs)
            if ei is not None:
                ss.edge_index = ei
        return hook

    @staticmethod
    def _mid_hook(ss):
        def hook(_m, args):
            x = _first_tensor(args)
            ss.x_mid = None if x is None else x.detach()
        return hook

    @staticmethod
    def _post_hook(ss):
        def hook(_m, _inp, out):
            x = _first_tensor(out)
            if x is not None:
                ss.record(x)
        return hook

    def close_case(self):
        for st in self.units.values():
            st.close_case()
        for hs in self.heads.values():
            hs.close_case()

    def remove(self):
        for h in self.handles:
            h.remove()


def _dead_dormant(scores: list[np.ndarray]):
    """(dead mask, dormant mask) from per-case unit scores."""
    S = np.stack(scores)  # (K, units)
    k = S.shape[0]
    zero = S == 0
    rel = S < DORMANT_REL * np.maximum(S.mean(1, keepdims=True), 1e-300)
    dead = zero.all(0)
    dormant = ((zero | rel).sum(0) >= math.ceil(0.75 * k)) & ~dead
    return dead, dormant


def run_grad_pass(root: torch.nn.Module, cases: list[Case], device) -> dict:
    """Checks 2, 4 and 7 over the cases (one forward + backward each)."""
    probe = Probe(root, device)
    losses = []
    try:
        for c in cases:
            root.zero_grad(set_to_none=True)
            res = c.run(input_grad=False, perturb=0.0, seed=0)
            res.loss.backward()
            losses.append(float(res.loss.detach()))
            probe.close_case()
            root.zero_grad(set_to_none=True)
    finally:
        probe.remove()
    c2 = {"per_role": {}, "indices": {}, "totals": {"K": len(cases), "dead_units": 0, "dormant_units": 0}}
    c4 = {"per_role": {}}
    for (role, name), st in probe.units.items():
        m2 = c2["per_role"].setdefault(role, {})
        i2 = c2["indices"].setdefault(role, {})
        if st.case_scores:
            dead, dormant = _dead_dormant(st.case_scores)
            m2[f"{name}.dead_unit_frac"] = float(dead.mean())
            m2[f"{name}.dormant_unit_frac"] = float(dormant.mean())
            i2[f"{name}.dead_units"] = np.where(dead)[0].tolist()
            i2[f"{name}.dormant_units"] = np.where(dormant)[0].tolist()
            c2["totals"]["dead_units"] += int(dead.sum())
            c2["totals"]["dormant_units"] += int(dormant.sum())
        if st.count:
            mx, mn = st.max.cpu().numpy(), st.min.cpu().numpy()
            mean = (st.sum / st.count).cpu().numpy()
            var = np.maximum((st.sumsq / st.count).cpu().numpy() - mean ** 2, 0)
            m4 = c4["per_role"].setdefault(role, {})
            m4[f"{name}.never_fires_frac"] = float((mx < GELU_ZERO_OUT).mean())
            m4[f"{name}.no_grad_frac"] = float((mx < GELU_ZERO_GRAD).mean())
            m4[f"{name}.linearised_frac"] = float((mn > 0).mean())
            m4[f"{name}.preact_mean.median"] = float(np.median(mean))
            m4[f"{name}.preact_std.median"] = float(np.median(np.sqrt(var)))
    for role, hs in probe.heads.items():
        if hs.case_scores:
            dead, dormant = _dead_dormant(hs.case_scores)
            c2["per_role"].setdefault(role, {})["dead_head_frac"] = float(dead.mean())
            c2["per_role"][role]["dormant_head_frac"] = float(dormant.mean())
            c2["indices"].setdefault(role, {})["dead_heads"] = np.where(dead)[0].tolist()
    c7 = {"per_role": {}}
    for role, ss in probe.streams.items():
        if ss.rows:
            keys = set().union(*ss.rows)
            c7["per_role"][role] = {k: float(np.mean([r[k] for r in ss.rows if k in r])) for k in sorted(keys)}
    c2["losses"] = losses
    c2["cases"] = [c.label for c in cases]
    return {"c2_grad_dead": c2, "c4_preact": c4, "c7_stream": c7}


def run_flow(root: torch.nn.Module, cases: list[Case], perturb: float = 0.1) -> dict:
    """Check 9 on the given cases (one per sigma): unread inputs, silent outputs."""
    per_sigma, unread_all, silent_all = {}, set(), set()
    n_in_total = n_out_total = 0
    for c in cases:
        root.zero_grad(set_to_none=True)
        res = c.run(input_grad=True, perturb=0.0, seed=0)
        gen = torch.Generator(device="cpu").manual_seed(0)
        R = (torch.randint(0, 2, res.output.shape, generator=gen) * 2 - 1).to(res.output.device, res.output.dtype)
        (res.output * R).sum().backward()
        unread = []
        n_in = 0
        for name, x in res.inputs.items():
            g = x.grad
            names = res.input_vars.get(name) or [f"{name}[{i}]" for i in range(x.shape[-1])]
            per_var = g.detach().abs().reshape(-1, x.shape[-1]).sum(0) if g is not None else torch.zeros(x.shape[-1])
            for i, v in enumerate(names):
                n_in += 1
                if float(per_var[i]) == 0.0:
                    unread.append(f"{name}:{v}")
        base = res.output.detach()
        with torch.no_grad():
            res2 = c.run(input_grad=False, perturb=perturb, seed=0)
        delta = (res2.output.detach() - base).abs().reshape(-1, base.shape[-1]).sum(0)
        outs = res.output_vars or [f"out[{i}]" for i in range(base.shape[-1])]
        silent = [v for i, v in enumerate(outs) if float(delta[i]) == 0.0]
        per_sigma[str(c.sigma)] = {"unread_inputs": unread, "silent_outputs": silent, "n_inputs": n_in, "n_outputs": len(outs)}
        unread_all |= set(unread)
        silent_all |= set(silent)
        n_in_total, n_out_total = n_in, len(outs)
        root.zero_grad(set_to_none=True)
    return {
        "c9_flow": {
            "per_role": {
                "model": {
                    "unread_input_frac": len(unread_all) / max(n_in_total, 1),
                    "silent_output_frac": len(silent_all) / max(n_out_total, 1),
                }
            },
            "per_sigma": per_sigma,
        }
    }


# ------------------------------------------------------------------------------------------------ check 8
def _components(n: int, src: np.ndarray, dst: np.ndarray) -> int:
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, d in zip(src.tolist(), dst.tolist()):
        rs, rd = find(s), find(d)
        if rs != rd:
            parent[rs] = rd
    return len({find(i) for i in range(n)})


def graph_audit(graph) -> dict:
    """Check 8 on a torch_geometric HeteroData (or anything with .edge_types and [edge_type].edge_index)."""
    out = {"per_role": {}, "detail": {}}
    for et in getattr(graph, "edge_types", []):
        src_t, _, dst_t = et
        store = graph[et]
        ei = store.edge_index.cpu().numpy()
        n_src = int(graph[src_t].num_nodes)
        n_dst = int(graph[dst_t].num_nodes)
        indeg = np.bincount(ei[1], minlength=n_dst)
        role = f"graph:{src_t}_to_{dst_t}"
        m = {
            "isolated_frac": float((indeg == 0).mean()),
            "indeg.min": float(indeg.min()),
            "indeg.median": float(np.median(indeg)),
            "indeg.max": float(indeg.max()),
            "edges_per_dst": float(ei.shape[1] / max(n_dst, 1)),
        }
        det = {"n_src": n_src, "n_dst": n_dst, "edges": int(ei.shape[1]), "sha1": hashlib.sha1(ei.tobytes()).hexdigest()}
        if src_t == dst_t and n_src <= 5_000_000:
            m["extra_components"] = float(_components(n_src, ei[0], ei[1]) - 1)
        attrs = {}
        for k in store.keys():
            v = store[k]
            if k != "edge_index" and torch.is_tensor(v) and v.is_floating_point():
                a = v.detach().float().cpu().numpy()
                attrs[k] = {"min": float(a.min()), "max": float(a.max()), "nonfinite": int((~np.isfinite(a)).sum())}
        det["edge_attrs"] = attrs
        out["per_role"][role] = m
        out["detail"][role] = det
    return {"c8_graph": out}
